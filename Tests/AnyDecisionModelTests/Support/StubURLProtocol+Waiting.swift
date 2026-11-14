import Foundation

extension StubURLProtocol.Host {
    /// An error that stops a test when a stub host does not receive the expected requests in time.
    struct RequestTimeout: Error, CustomStringConvertible {
        let expected: Int
        let received: Int
        let timeout: Duration

        var description: String {
            "Expected \(expected) request(s) within \(timeout), but the stub host received \(received)."
        }
    }

    /// Waits until this host records at least `count` requests.
    ///
    /// Use this instead of a fixed sleep
    /// when a test must act after a request reaches the stub,
    /// because a slow runner can take longer than any fixed sleep
    /// to start the request.
    /// If the requests do not arrive before `timeout`,
    /// this method throws ``RequestTimeout``, which fails the test.
    func waitForRequests(count: Int = 1, timeout: Duration = .seconds(10)) async throws {
        let deadline = ContinuousClock.now.advanced(by: timeout)
        while recorded.count < count {
            guard ContinuousClock.now < deadline else {
                throw RequestTimeout(expected: count, received: recorded.count, timeout: timeout)
            }
            try await Task.sleep(for: .milliseconds(5))
        }
    }
}
