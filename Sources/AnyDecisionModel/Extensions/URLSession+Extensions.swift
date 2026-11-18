// Adapted from AnyLanguageModel (https://github.com/huggingface/AnyLanguageModel).
// Copyright AnyLanguageModel contributors. Licensed under the Apache License, Version 2.0.

import Foundation
import HTTPTypes
import HTTPTypesFoundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

enum HTTP {
    enum Method: String {
        case get = "GET"
        case post = "POST"
    }

    /// Returns data with the credentials in request headers replaced.
    ///
    /// The credentials are the values of the headers in
    /// ``DecisionError/credentialHeaders``,
    /// and the token after the scheme in a value such as `Bearer token`.
    static func redactingCredentials(in data: Data, requestHeaders: [String: String]) -> Data {
        let names = Set(DecisionError.credentialHeaders.map(\.canonicalName))
        var secrets: Set<String> = []
        for (name, value) in requestHeaders where names.contains(name.lowercased()) {
            secrets.insert(value)
            if let space = value.firstIndex(of: " ") {
                secrets.insert(String(value[value.index(after: space)...]).trimmingCharacters(in: .whitespaces))
            }
        }
        // Replace longer values first, so a token inside a full header value doesn't split it.
        let replacement = Data(DecisionError.redactedValue.utf8)
        var data = data
        for secret in secrets.sorted(by: { $0.utf8.count > $1.utf8.count }) where !secret.isEmpty {
            let bytes = Data(secret.utf8)
            var start = data.startIndex
            while let range = data.range(of: bytes, in: start ..< data.endIndex) {
                data.replaceSubrange(range, with: replacement)
                start = range.lowerBound + replacement.count
            }
        }
        return data
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
    /// For a status code outside 200 to 299,
    /// the values of credential headers in `headers`, such as `Authorization`,
    /// are replaced in the error's body,
    /// and the values of credential headers in the error's response are replaced.
    /// Other response headers are kept as they are.
    ///
    /// A request that fails on the way, or that is cancelled,
    /// throws the URL session's own `URLError`.
    ///
    /// - Throws: A ``DecisionError`` with the response and body
    ///   for a status code outside 200 to 299,
    ///   as ``DecisionError/requestFailed(statusCode:detail:headers:)`` describes,
    ///   ``DecisionError/invalidResponse(_:)`` for a non-HTTP response,
    ///   or `URLError` if the request fails on the way or is cancelled.
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

        #if canImport(FoundationNetworking)
            let (data, response) = try await withLinuxRequestLock {
                try await self.data(for: request)
            }
        #else
            let (data, response) = try await self.data(for: request)
        #endif

        guard let httpResponse = response as? HTTPURLResponse else {
            throw DecisionError.invalidResponse("The response is not an HTTP response.")
        }

        guard (200 ..< 300).contains(httpResponse.statusCode) else {
            // A server can echo a credential, for example in an "invalid API key" message.
            let body = HTTP.redactingCredentials(in: data, requestHeaders: headers)
            guard let response = httpResponse.httpResponse else {
                throw DecisionError.requestFailed(
                    statusCode: httpResponse.statusCode,
                    detail: String(data: body, encoding: .utf8) ?? "Invalid response",
                    headers: [:]
                )
            }
            throw DecisionError(response: response, body: body)
        }

        return (data, httpResponse)
    }
}
