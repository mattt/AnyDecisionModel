import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// A decision model that calls the TypeSafe Jev API.
///
/// The model sends each batch to `POST /v1/systemone`
/// and returns Jev's probabilities and confidence values unchanged.
///
/// By default, the API key comes from the `TYPESAFE_AI_API_KEY` environment variable,
/// or from `TYPESAFE_API_KEY` if the first is not set.
/// The key is sent only in the `Authorization` header
/// and is never included in descriptions or errors.
///
/// ```swift
/// let model = JevDecisionModel()
/// let session = DecisionSession(model: model, state: .text(ticket))
/// let urgent = try await session.probability(of: "Does this convey urgency?")
/// ```
public struct JevDecisionModel: DecisionModel, CustomStringConvertible {
    /// The default API base URL.
    public static let defaultBaseURL = URL(string: "https://api.typesafe.ai")!

    /// The environment variables that supply the API key, in order of preference.
    private static let apiKeyEnvironmentVariables = ["TYPESAFE_AI_API_KEY", "TYPESAFE_API_KEY"]

    /// The API base URL.
    public let baseURL: URL

    /// The Jev model identifier, such as `jev-latest`.
    public let modelID: String

    /// The retry behavior for eligible HTTP failures.
    public let retryPolicy: RetryPolicy

    private let tokenProvider: @Sendable () -> String?
    private let urlSession: URLSession

    /// Creates a Jev decision model.
    ///
    /// - Parameters:
    ///   - baseURL: The API base URL.
    ///   - apiKey: The API key, or a closure that returns it.
    ///     Defaults to the value of `TYPESAFE_AI_API_KEY` or `TYPESAFE_API_KEY`.
    ///   - modelID: The Jev model identifier.
    ///   - session: The URL session for requests.
    ///   - retryPolicy: The retry behavior for eligible HTTP failures.
    public init(
        baseURL: URL = defaultBaseURL,
        apiKey tokenProvider: @escaping @autoclosure @Sendable () -> String? = JevDecisionModel.environmentAPIKey(),
        modelID: String = "jev-latest",
        session: URLSession = URLSession(configuration: .default),
        retryPolicy: RetryPolicy = .default
    ) {
        self.baseURL = baseURL
        self.tokenProvider = tokenProvider
        self.modelID = modelID
        self.urlSession = session
        self.retryPolicy = retryPolicy
    }

    /// Returns the API key from the environment, or `nil` if none is set.
    ///
    /// - Parameter environment: The environment to read.
    public static func environmentAPIKey(
        from environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        for name in apiKeyEnvironmentVariables {
            if let value = environment[name]?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty {
                return value
            }
        }
        return nil
    }

    public var capabilities: DecisionCapabilities {
        DecisionCapabilities()
    }

    public var description: String {
        "JevDecisionModel(modelID: \(modelID), baseURL: \(baseURL.absoluteString))"
    }

    public func makeExecutor() -> any DecisionModelExecutor {
        Executor(model: self)
    }

    private struct Executor: DecisionModelExecutor {
        let model: JevDecisionModel

        func decide(_ questions: [Question], about state: DecisionState) async throws
            -> DecisionSession.Response
        {
            // Jev keys questions and answers by identifier. Use each question's position.
            let keys = questions.indices.map(String.init)
            let request = SystemOneRequest(
                model: model.modelID,
                state: state,
                questions: Dictionary(uniqueKeysWithValues: zip(keys, questions.map(SystemOneQuestion.init))),
                keepAlive: nil
            )
            let response = try await model.send(request)
            let answers = try keys.map { key in
                guard let answer = response.answers[key] else {
                    throw DecisionError.invalidResponse("No answer for the question at index \(key).")
                }
                return answer.answer
            }
            return DecisionSession.Response(
                modelID: response.model,
                answers: answers,
                usage: DecisionSession.Usage(
                    inputTokenCount: response.usage?.inputTokenCount ?? 0,
                    outputTokenCount: response.usage?.outputTokenCount ?? 0
                )
            )
        }
    }

    /// Sends a request and returns the decoded response.
    ///
    /// Eligible HTTP failures are retried according to ``retryPolicy``.
    private func send(_ request: SystemOneRequest) async throws -> SystemOneResponse {
        guard let apiKey = tokenProvider(), !apiKey.isEmpty else {
            throw DecisionError.missingCredentials
        }

        let url = baseURL.appendingPathComponent("v1").appendingPathComponent("systemone")
        let body = try JSONEncoder().encode(request)
        let headers = ["Authorization": "Bearer \(apiKey)"]

        var retrier = retryPolicy.makeIterator()
        while true {
            try Task.checkCancellation()
            do {
                let (data, _) = try await urlSession.send(.post, url: url, headers: headers, body: body)
                do {
                    return try JSONDecoder().decode(SystemOneResponse.self, from: data)
                } catch {
                    throw DecisionError.invalidResponse(String(describing: error))
                }
            } catch let error as DecisionError {
                guard case .requestFailed(let statusCode, _, let responseHeaders) = error,
                    retryPolicy.retryableStatusCodes.contains(statusCode),
                    let generatedDelay = retrier.next()
                else {
                    throw error
                }

                let delay = retryPolicy.delay(
                    generatedDelay,
                    retryAfter: responseHeaders["retry-after"]
                )
                try await Task.sleep(for: .seconds(delay))
            }
        }
    }
}

// MARK: - Retry policy

extension JevDecisionModel {
    /// Retry behavior for eligible HTTP failures.
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
        /// This limit also applies to `Retry-After` headers. A value of `nil`
        /// disables this limit.
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

        func delay(
            _ generatedDelay: TimeInterval,
            retryAfter: String?,
            now: Date = Date()
        ) -> TimeInterval {
            guard let retryAfter,
                let retryAfterDelay = Self.parseRetryAfter(retryAfter, now: now)
            else {
                return clamp(generatedDelay)
            }
            return clamp(retryAfterDelay)
        }

        private func clamp(_ delay: TimeInterval) -> TimeInterval {
            guard !delay.isNaN else { return 0 }
            return Swift.min(
                Swift.max(delay, 0),
                maximumInterval ?? .greatestFiniteMagnitude
            )
        }

        static func parseRetryAfter(_ value: String, now: Date) -> Double? {
            let trimmed = value.trimmingCharacters(in: .whitespaces)
            if let seconds = Double(trimmed), seconds.isFinite {
                return seconds
            }
            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.timeZone = TimeZone(identifier: "GMT")
            formatter.dateFormat = "EEE, dd MMM yyyy HH:mm:ss zzz"
            if let date = formatter.date(from: trimmed) {
                return date.timeIntervalSince(now)
            }
            return nil
        }
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
