// Adapted from AnyLanguageModel (https://github.com/huggingface/AnyLanguageModel).
// Copyright AnyLanguageModel contributors. Licensed under the Apache License, Version 2.0.

import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// A `URLProtocol` that answers requests from queues of canned responses.
///
/// Each test uses its own host, so tests that run in parallel do not share queues.
final class StubURLProtocol: URLProtocol {
    struct Exchange: Sendable {
        var statusCode: Int = 200
        var body: Data
        var headers: [String: String] = [:]
        /// A delay before the response, in seconds.
        var delay: Double = 0
    }

    struct Recorded: Sendable {
        var body: Data
        var headers: [String: String]
    }

    private struct HostState: Sendable {
        var pending: [Exchange] = []
        var recorded: [Recorded] = []
    }

    private static let state = LockedState<[String: HostState]>([:])

    /// A stub host with its own response queue.
    struct Host: Sendable {
        let name: String

        init() {
            name = "stub-\(UUID().uuidString.lowercased()).invalid"
        }

        var baseURL: URL { URL(string: "https://\(name)")! }

        /// Queues one response for the next request to this host.
        func enqueue(json: String, statusCode: Int = 200, headers: [String: String] = [:], delay: Double = 0) {
            let exchange = Exchange(statusCode: statusCode, body: Data(json.utf8), headers: headers, delay: delay)
            StubURLProtocol.state.withLock { $0[name, default: HostState()].pending.append(exchange) }
        }

        /// The requests that this host received, in order.
        var recorded: [Recorded] {
            StubURLProtocol.state.withLock { $0[name]?.recorded ?? [] }
        }
    }

    /// A session that routes every request to this protocol.
    static func makeSession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        return URLSession(configuration: configuration)
    }

    private let stopped = LockedState(false)

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        // URLSession moves `httpBody` to `httpBodyStream` before the protocol sees the request.
        let body = request.httpBody ?? request.httpBodyStream.map(Self.readAll) ?? Data()
        let headers = request.allHTTPHeaderFields ?? [:]
        let host = request.url?.host ?? ""

        let exchange = Self.state.withLock { state -> Exchange? in
            state[host, default: HostState()].recorded.append(Recorded(body: body, headers: headers))
            guard state[host]?.pending.isEmpty == false else { return nil }
            return state[host]?.pending.removeFirst()
        }

        guard let exchange, let url = request.url else {
            client?.urlProtocol(self, didFailWithError: URLError(.resourceUnavailable))
            return
        }

        // FoundationNetworking does not treat URLProtocol subclasses as Sendable.
        let protocolBox = UncheckedSendable(value: self)
        let respond: @Sendable () -> Void = {
            let stub = protocolBox.value
            guard !stub.stopped.withLock({ $0 }) else { return }
            var fields = exchange.headers
            fields["Content-Type"] = "application/json"
            let response = HTTPURLResponse(
                url: url,
                statusCode: exchange.statusCode,
                httpVersion: "HTTP/1.1",
                headerFields: fields
            )!
            stub.client?.urlProtocol(stub, didReceive: response, cacheStoragePolicy: .notAllowed)
            stub.client?.urlProtocol(stub, didLoad: exchange.body)
            stub.client?.urlProtocolDidFinishLoading(stub)
        }

        if exchange.delay > 0 {
            DispatchQueue.global().asyncAfter(deadline: .now() + exchange.delay, execute: respond)
        } else {
            respond()
        }
    }

    override func stopLoading() {
        stopped.withLock { $0 = true }
    }

    private static func readAll(_ stream: InputStream) -> Data {
        stream.open()
        defer { stream.close() }

        var data = Data()
        let bufferSize = 4096
        var buffer = [UInt8](repeating: 0, count: bufferSize)
        while true {
            let read = stream.read(&buffer, maxLength: bufferSize)
            if read <= 0 { break }
            data.append(buffer, count: read)
        }
        return data
    }
}

/// Wraps a value that is used from one thread at a time.
struct UncheckedSendable<Value>: @unchecked Sendable {
    let value: Value
}

/// A minimal lock for test state.
final class LockedState<State>: @unchecked Sendable {
    private let lock = NSLock()
    private var state: State

    init(_ state: State) {
        self.state = state
    }

    func withLock<T>(_ body: (inout State) throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body(&state)
    }
}
