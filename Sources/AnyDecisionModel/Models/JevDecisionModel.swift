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
    ///
    /// This is a type alias for the top-level ``AnyDecisionModel/RetryPolicy``.
    public typealias RetryPolicy = AnyDecisionModel.RetryPolicy
}
