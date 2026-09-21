// Adapted from AnyLanguageModel (https://github.com/huggingface/AnyLanguageModel).
// Copyright AnyLanguageModel contributors. Licensed under the Apache License, Version 2.0.

import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

enum HTTP {
    enum Method: String {
        case get = "GET"
        case post = "POST"
    }
}

#if canImport(FoundationNetworking)
    /// Serializes Linux URLSession operations to mitigate a FoundationNetworking race.
    ///
    /// On Linux, `FoundationNetworking` routes `URLSession` through a shared
    /// `_MultiHandle`, which can crash under concurrent access
    /// (`URLSession._MultiHandle.endOperation(for:)`).
    /// Requests hold this gate for the full request and response cycle,
    /// which serializes them on Linux.
    ///
    /// See: https://github.com/swiftlang/swift-corelibs-foundation/issues/4791
    actor LinuxURLSessionRequestGate {
        private struct Waiter {
            let id: UUID
            let continuation: CheckedContinuation<Void, Error>
        }

        static let shared = LinuxURLSessionRequestGate()

        private var isLocked = false
        private var waiters: [Waiter] = []

        func acquire() async throws {
            if Task.isCancelled {
                throw CancellationError()
            }

            if !isLocked {
                isLocked = true
                return
            }

            let waiterID = UUID()
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    waiters.append(Waiter(id: waiterID, continuation: continuation))
                }
            } onCancel: {
                Task {
                    await self.cancelWaiter(id: waiterID)
                }
            }
        }

        func release() {
            if waiters.isEmpty {
                isLocked = false
                return
            }

            let waiter = waiters.removeFirst()
            waiter.continuation.resume()
        }

        private func cancelWaiter(id: UUID) {
            guard let index = waiters.firstIndex(where: { $0.id == id }) else {
                return
            }

            let waiter = waiters.remove(at: index)
            waiter.continuation.resume(throwing: CancellationError())
        }
    }

    func withLinuxRequestLock<T>(
        _ operation: () async throws -> T
    ) async throws -> T {
        let gate = LinuxURLSessionRequestGate.shared
        try await gate.acquire()
        do {
            let result = try await operation()
            await gate.release()
            return result
        } catch {
            await gate.release()
            throw error
        }
    }
#endif

extension URLSession {
    /// Sends a request and returns the body and response.
    ///
    /// - Throws: ``DecisionError/requestFailed(statusCode:detail:headers:)``
    ///   for a status code outside 200 to 299, with the response headers,
    ///   ``DecisionError/invalidResponse(_:)`` for a non-HTTP response,
    ///   or `CancellationError` if the task is cancelled.
    func send(
        _ method: HTTP.Method,
        url: URL,
        headers: [String: String] = [:],
        body: Data? = nil
    ) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url)
        request.httpMethod = method.rawValue
        request.addValue("application/json", forHTTPHeaderField: "Accept")

        for (key, value) in headers {
            request.addValue(value, forHTTPHeaderField: key)
        }

        if let body {
            request.httpBody = body
            request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        let data: Data
        let response: URLResponse
        do {
            #if canImport(FoundationNetworking)
                (data, response) = try await withLinuxRequestLock {
                    try await self.data(for: request)
                }
            #else
                (data, response) = try await self.data(for: request)
            #endif
        } catch let error as URLError where error.code == .cancelled {
            throw CancellationError()
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw DecisionError.invalidResponse("The response is not an HTTP response.")
        }

        guard (200 ..< 300).contains(httpResponse.statusCode) else {
            let detail = String(data: data, encoding: .utf8) ?? "Invalid response"
            throw DecisionError.requestFailed(
                statusCode: httpResponse.statusCode,
                detail: detail,
                headers: httpResponse.stringHeaders
            )
        }

        return (data, httpResponse)
    }
}

extension HTTPURLResponse {
    /// The response headers with lowercase names.
    var stringHeaders: [String: String] {
        var headers: [String: String] = [:]
        for (key, value) in allHeaderFields {
            guard let key = key as? String else { continue }
            headers[key.lowercased()] = value as? String ?? String(describing: value)
        }
        return headers
    }
}
