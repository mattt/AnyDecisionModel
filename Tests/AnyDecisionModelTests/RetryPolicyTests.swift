import Dispatch
import Foundation
import Testing

@testable import AnyDecisionModel

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

@Suite("RetryPolicy")
struct RetryPolicyTests {
    @Test func jevAliasIsTopLevelType() {
        #expect(JevDecisionModel.RetryPolicy.self == RetryPolicy.self)
        let policy: JevDecisionModel.RetryPolicy = RetryPolicy.never
        #expect(JevDecisionModel(apiKey: nil, retryPolicy: policy).retryPolicy == RetryPolicy.never)
    }

    @Test func constantBackoffStrategy() {
        let policy = RetryPolicy(
            strategy: .constant(duration: 1, jitter: 0),
            timeout: nil,
            maximumInterval: nil,
            maximumRetries: 5
        )

        #expect(Array(policy) == [1, 1, 1, 1, 1])
    }

    @Test func exponentialBackoffStrategy() {
        let policy = RetryPolicy(
            strategy: .exponential(base: 1, multiplier: 2, jitter: 0),
            timeout: nil,
            maximumInterval: 30,
            maximumRetries: 7
        )

        #expect(Array(policy) == [1, 2, 4, 8, 16, 30, 30])
    }

    @Test func defaultPolicyKeepsJevBehavior() {
        #expect(Array(RetryPolicy.default) == [0.5, 1, 2, 4, 8])
        #expect(RetryPolicy.default.timeout == nil)
        #expect(RetryPolicy.default.maximumInterval == 30)
        #expect(RetryPolicy.default.retryableStatusCodes == [429, 503, 529])
    }

    @Test func timeoutCreatesDeadline() {
        let timeout: TimeInterval = 300
        let timeoutNanoseconds = UInt64(timeout * 1e9)
        let before = DispatchTime.now().uptimeNanoseconds + timeoutNanoseconds
        let policy = RetryPolicy(
            strategy: .constant(),
            timeout: timeout,
            maximumInterval: nil,
            maximumRetries: nil
        )
        let deadline = policy.makeIterator().deadline
        let after = DispatchTime.now().uptimeNanoseconds + timeoutNanoseconds

        #expect(deadline != nil)
        #expect(deadline.map { $0.uptimeNanoseconds >= before } == true)
        #expect(deadline.map { $0.uptimeNanoseconds <= after } == true)
    }

    @Test func nilTimeoutHasNoDeadline() {
        let policy = RetryPolicy(
            strategy: .constant(),
            timeout: nil,
            maximumInterval: nil,
            maximumRetries: nil
        )

        #expect(policy.makeIterator().deadline == nil)
    }

    @Test func expiredDeadlineProducesNoDelay() {
        let policy = RetryPolicy(
            strategy: .constant(duration: 1),
            timeout: nil,
            maximumInterval: nil,
            maximumRetries: nil
        )
        var retrier = RetryPolicy.Retrier(
            policy: policy,
            deadline: DispatchTime(uptimeNanoseconds: 0)
        )

        #expect(retrier.next() == nil)
        #expect(retrier.retries == 0)
    }

    @Test func maximumIntervalCapsDelaysAfterJitter() {
        var expectedGenerator = SeededRandomNumberGenerator(seed: 42)
        let jitter = Double.random(in: -10 ... 10, using: &expectedGenerator)
        let maximumInterval = 30 + jitter - 0.1
        let policy = RetryPolicy(
            strategy: .constant(duration: 30, jitter: 20),
            timeout: nil,
            maximumInterval: maximumInterval,
            maximumRetries: 1
        )
        var retrier = RetryPolicy.Retrier(
            policy: policy,
            randomNumberGenerator: SeededRandomNumberGenerator(seed: 42),
            deadline: nil
        )

        #expect(retrier.next() == maximumInterval)
    }

    @Test func nilMaximumIntervalDoesNotCapDelays() {
        let policy = RetryPolicy(
            strategy: .constant(duration: 31, jitter: 0),
            timeout: nil,
            maximumInterval: nil,
            maximumRetries: 1
        )

        #expect(Array(policy) == [31])
    }

    @Test func nilMaximumRetriesDoesNotLimitSequence() {
        let policy = RetryPolicy(
            strategy: .constant(duration: 1),
            timeout: nil,
            maximumInterval: nil,
            maximumRetries: nil
        )
        var retrier = policy.makeIterator()

        #expect((0 ..< 100).allSatisfy { _ in retrier.next() == 1 })
        #expect(retrier.retries == 100)
    }

    @Test func zeroRetriesProducesEmptySequence() {
        #expect(Array(RetryPolicy(maximumRetries: 0)).isEmpty)
        #expect(Array(RetryPolicy.never).isEmpty)
    }

    @Test func iteratorsAreIndependent() {
        let policy = RetryPolicy(
            strategy: .exponential(base: 1, multiplier: 2, jitter: 0),
            maximumRetries: 2
        )
        var first = policy.makeIterator()
        var second = policy.makeIterator()

        #expect(first.next() == 1)
        #expect(first.next() == 2)
        #expect(first.next() == nil)
        #expect(second.next() == 1)
        #expect(second.retries == 1)
    }

    @Test func injectedGeneratorControlsJitter() {
        let policy = RetryPolicy(
            strategy: .constant(duration: 2, jitter: 2),
            timeout: nil,
            maximumInterval: nil,
            maximumRetries: 1
        )
        var expectedGenerator = SeededRandomNumberGenerator(seed: 42)
        let expected = 2 + Double.random(in: -1 ... 1, using: &expectedGenerator)
        var retrier = RetryPolicy.Retrier(
            policy: policy,
            randomNumberGenerator: SeededRandomNumberGenerator(seed: 42),
            deadline: nil
        )

        #expect(retrier.next() == expected)
    }

    @Test func delaysStayNonnegative() {
        let policy = RetryPolicy(
            strategy: .constant(duration: -10, jitter: 0),
            timeout: nil,
            maximumInterval: nil,
            maximumRetries: 1
        )

        #expect(Array(policy) == [0])
    }

    @Test func exponentialGrowthContinuesBeyondThirtyTwoRetries() {
        let policy = RetryPolicy(
            strategy: .exponential(base: 1e-9, multiplier: 2, jitter: 0),
            timeout: nil,
            maximumInterval: 3_600,
            maximumRetries: 101
        )
        let delays = Array(policy)

        #expect(delays[33] == delays[32] * 2)
        #expect(delays[41] == 1e-9 * pow(2, 41))
        #expect(delays[100] == 3_600)
    }

    @Test func retryAfterSecondsReplacesAndCapsGeneratedDelay() {
        let policy = RetryPolicy(maximumInterval: 8)
        let now = Date()

        #expect(policy.delay(0.5, retryAfter: DecisionError.retryAfter(from: "3", now: now)) == 3)
        #expect(policy.delay(0.5, retryAfter: DecisionError.retryAfter(from: "120", now: now)) == 8)
        #expect(policy.delay(0.5, retryAfter: DecisionError.retryAfter(from: "-1", now: now)) == 0)
        #expect(policy.delay(0.5, retryAfter: .milliseconds(1_500)) == 1.5)
    }

    @Test func retryAfterHTTPDateReplacesGeneratedDelay() {
        let policy = RetryPolicy(maximumInterval: 8)
        let now = Date(timeIntervalSince1970: 1_445_412_480)

        #expect(
            policy.delay(
                0.5,
                retryAfter: DecisionError.retryAfter(from: "Wed, 21 Oct 2015 07:28:05 GMT", now: now)
            ) == 5
        )
        #expect(
            policy.delay(
                0.5,
                retryAfter: DecisionError.retryAfter(from: "Wed, 21 Oct 2015 07:27:00 GMT", now: now)
            ) == 0
        )
    }

    @Test func invalidRetryAfterUsesGeneratedDelay() {
        let policy = RetryPolicy(maximumInterval: 8)

        #expect(DecisionError.retryAfter(from: "soon", now: Date()) == nil)
        #expect(policy.delay(1, retryAfter: nil) == 1)
    }

    @Test func retryAfterWithoutMaximumIntervalIsNotLimited() {
        let policy = RetryPolicy(maximumInterval: nil)

        #expect(policy.delay(0.5, retryAfter: .seconds(3_600)) == 3_600)
    }

    // MARK: - Retry decisions

    /// The retry decision for every status code matches version 0.9,
    /// which retried a `requestFailed` error whose status code is in `retryableStatusCodes`.
    @Test(
        arguments: [
            RetryPolicy.default,
            RetryPolicy.never,
            RetryPolicy(retryableStatusCodes: []),
            RetryPolicy(retryableStatusCodes: [408, 500, 502, 504]),
            RetryPolicy(retryableStatusCodes: Set(100 ..< 600)),
        ]
    )
    func statusCodeDecidesRetry(policy: RetryPolicy) {
        for statusCode in 100 ..< 600 {
            let error = DecisionError.requestFailed(
                statusCode: statusCode,
                detail: "",
                headers: ["Retry-After": "1"]
            )
            #expect(
                policy.shouldRetry(error) == policy.retryableStatusCodes.contains(statusCode),
                "status \(statusCode)"
            )
        }
    }

    /// The policy's status codes override the error's advice.
    @Test func statusCodesOverrideIsRetryable() {
        let timeout = DecisionError.requestFailed(statusCode: 504, detail: "", headers: [:])
        #expect(timeout.isRetryable)
        #expect(!RetryPolicy.default.shouldRetry(timeout))

        let serverError = DecisionError.requestFailed(statusCode: 500, detail: "", headers: [:])
        #expect(!serverError.isRetryable)
        #expect(RetryPolicy(retryableStatusCodes: [500]).shouldRetry(serverError))
    }

    @Test(
        arguments: [
            DecisionError(URLError(.timedOut)),
            DecisionError(URLError(.networkConnectionLost)),
            DecisionError(code: .rateLimited, message: "Slow down."),
            DecisionError.invalidResponse("Not JSON."),
        ]
    )
    func errorsWithoutStatusCodeAreNotRetried(error: DecisionError) {
        #expect(!RetryPolicy(retryableStatusCodes: Set(100 ..< 600)).shouldRetry(error))
    }
}

private struct SeededRandomNumberGenerator: RandomNumberGenerator {
    var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state = 2_862_933_555_777_941_757 &* state &+ 3_037_000_493
        return state
    }
}
