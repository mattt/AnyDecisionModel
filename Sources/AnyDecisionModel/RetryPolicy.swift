import Dispatch
import Foundation

/// Retry behavior for eligible HTTP failures.
///
/// A model retries a ``DecisionError`` whose ``DecisionError/response``
/// has a status code in ``retryableStatusCodes``.
/// The status code decides,
/// whatever the error's ``DecisionError/isRetryable`` value,
/// and an error without a response is not retried.
/// A `URLError`, such as a timeout in the URL session, is not retried either.
/// The error's ``DecisionError/retryAfter`` value replaces the generated delay,
/// and either delay is limited by ``maximumInterval``.
public struct RetryPolicy: Hashable, Sendable, Sequence {
    /// A strategy that determines how long to wait between retries.
    public enum Strategy: Hashable, Sendable {
        /// Waits for a constant interval, plus random jitter.
        ///
        /// The delay is `duration + R(-jitter / 2 ... jitter / 2)`.
        case constant(duration: TimeInterval = 2, jitter: Double = 0)

        /// Waits for an exponentially increasing interval, plus random jitter.
        ///
        /// For retry `n`, the delay is
        /// `base \cdot multiplier^n + R(-jitter / 2 ... jitter / 2)`.
        case exponential(
            base: TimeInterval = 2,
            multiplier: Double = 2,
            jitter: Double = 0.5
        )
    }

    /// The strategy used to determine how long to wait between retries.
    public let strategy: Strategy

    /// The total time during which the policy can produce delays.
    ///
    /// The timeout starts when an iterator is created. A value of `nil`
    /// disables this limit.
    public let timeout: TimeInterval?

    /// The longest delay between requests.
    ///
    /// This limit also applies to ``DecisionError/retryAfter``,
    /// which comes from a `Retry-After` header.
    /// A value of `nil` disables this limit.
    public let maximumInterval: TimeInterval?

    /// The largest number of retries after the first attempt.
    ///
    /// A value of `nil` disables this limit.
    public let maximumRetries: Int?

    /// The status codes to retry.
    public let retryableStatusCodes: Set<Int>

    /// Creates a retry policy.
    ///
    /// - Parameters:
    ///   - strategy: The strategy used to determine each delay.
    ///   - timeout: The total time during which the policy can produce delays.
    ///     This value must be greater than zero when specified.
    ///   - maximumInterval: The longest delay between requests.
    ///     This value must be greater than zero when specified.
    ///   - maximumRetries: The largest number of retries after the first attempt.
    ///     This value must be zero or greater when specified.
    ///   - retryableStatusCodes: The HTTP status codes to retry.
    public init(
        strategy: Strategy = .exponential(base: 0.5, multiplier: 2, jitter: 0),
        timeout: TimeInterval? = nil,
        maximumInterval: TimeInterval? = 30,
        maximumRetries: Int? = 5,
        retryableStatusCodes: Set<Int> = [429, 503, 529]
    ) {
        precondition(timeout.map { $0 > 0 && $0.isFinite } ?? true)
        precondition(maximumInterval.map { $0 > 0 && $0.isFinite } ?? true)
        precondition(maximumRetries.map { $0 >= 0 } ?? true)

        self.strategy = strategy
        self.timeout = timeout
        self.maximumInterval = maximumInterval
        self.maximumRetries = maximumRetries
        self.retryableStatusCodes = retryableStatusCodes
    }

    /// Five retries with exponential backoff from 0.5 seconds, capped at 30 seconds.
    public static let `default` = RetryPolicy()

    /// No retries.
    public static let never = RetryPolicy(maximumRetries: 0)

    /// One use of a retry policy.
    public struct Retrier: IteratorProtocol {
        /// The number of delay values already produced.
        public private(set) var retries = 0

        /// The retry policy.
        public let policy: RetryPolicy

        /// The time after which this iterator stops producing delays.
        public let deadline: DispatchTime?

        private var randomNumberGenerator: any RandomNumberGenerator

        init(
            policy: RetryPolicy,
            randomNumberGenerator: any RandomNumberGenerator = SystemRandomNumberGenerator(),
            deadline: DispatchTime?
        ) {
            self.policy = policy
            self.randomNumberGenerator = randomNumberGenerator
            self.deadline = deadline
        }

        /// Returns the next delay, or `nil` when a limit is reached.
        public mutating func next() -> TimeInterval? {
            guard policy.maximumRetries.map({ retries < $0 }) ?? true else {
                return nil
            }
            guard deadline.map({ DispatchTime.now() < $0 }) ?? true else {
                return nil
            }

            defer { retries += 1 }
            let delay: TimeInterval
            switch policy.strategy {
            case .constant(let duration, let jitter):
                delay =
                    duration
                    + Double.random(
                        jitter: jitter,
                        using: &randomNumberGenerator
                    )
            case .exponential(let base, let multiplier, let jitter):
                delay =
                    base * pow(multiplier, Double(retries))
                    + Double.random(jitter: jitter, using: &randomNumberGenerator)
            }

            return policy.clamp(delay)
        }
    }

    /// Returns a new, independent use of this retry policy.
    public func makeIterator() -> Retrier {
        Retrier(
            policy: self,
            deadline: timeout.map { timeout in
                let nanoseconds = timeout * 1e9
                guard nanoseconds < Double(UInt64.max) else {
                    return DispatchTime(uptimeNanoseconds: UInt64.max)
                }
                let result = DispatchTime.now().uptimeNanoseconds.addingReportingOverflow(
                    UInt64(nanoseconds)
                )
                return DispatchTime(
                    uptimeNanoseconds: result.overflow ? UInt64.max : result.partialValue
                )
            }
        )
    }

    /// Returns whether the policy retries an error.
    ///
    /// The policy retries an error with an HTTP status code
    /// in ``retryableStatusCodes``,
    /// whatever its ``DecisionError/isRetryable`` value.
    /// It does not retry an error without a status code,
    /// such as a response that can't be decoded.
    func shouldRetry(_ error: DecisionError) -> Bool {
        guard let statusCode = error.response?.status.code else { return false }
        return retryableStatusCodes.contains(statusCode)
    }

    /// Returns the delay before the next attempt, in seconds.
    ///
    /// A `Retry-After` delay replaces the generated delay.
    /// Either delay is limited to zero through ``maximumInterval``.
    ///
    /// - Parameters:
    ///   - generatedDelay: The delay from ``Retrier/next()``.
    ///   - retryAfter: The error's ``DecisionError/retryAfter`` value.
    func delay(_ generatedDelay: TimeInterval, retryAfter: Duration?) -> TimeInterval {
        guard let retryAfter else {
            return clamp(generatedDelay)
        }
        let (seconds, attoseconds) = retryAfter.components
        return clamp(Double(seconds) + Double(attoseconds) * 1e-18)
    }

    private func clamp(_ delay: TimeInterval) -> TimeInterval {
        guard !delay.isNaN else { return 0 }
        return Swift.min(
            Swift.max(delay, 0),
            maximumInterval ?? .greatestFiniteMagnitude
        )
    }
}

// MARK: -

private extension Double {
    static func random<T: RandomNumberGenerator>(
        jitter amount: Double,
        using generator: inout T
    ) -> Double {
        guard !amount.isZero else { return 0 }
        let halfAmount = abs(amount) / 2
        return Double.random(in: -halfAmount ... halfAmount, using: &generator)
    }
}
