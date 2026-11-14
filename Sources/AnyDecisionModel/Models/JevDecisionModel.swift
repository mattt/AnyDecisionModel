import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// A decision model that calls the TypeSafe Jev API.
///
/// The model sends each batch to `POST /v1/systemone`
/// and returns Jev's probabilities and confidence values unchanged.
/// It has the same configuration and behavior as
/// ``SystemOneDecisionModel/jev(baseURL:apiKey:modelID:session:retryPolicy:)``.
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

    private let base: SystemOneDecisionModel

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
        self.modelID = modelID
        self.retryPolicy = retryPolicy
        self.base = .jev(
            baseURL: baseURL,
            apiKey: tokenProvider(),
            modelID: modelID,
            session: session,
            retryPolicy: retryPolicy
        )
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
        base.capabilities
    }

    public var description: String {
        "JevDecisionModel(modelID: \(modelID), baseURL: \(baseURL.absoluteString))"
    }

    public func makeExecutor() -> any DecisionModelExecutor {
        base.makeExecutor()
    }
}

// MARK: - Retry policy

extension JevDecisionModel {
    /// Retry behavior for eligible HTTP failures.
    ///
    /// This is a type alias for the top-level ``AnyDecisionModel/RetryPolicy``.
    public typealias RetryPolicy = AnyDecisionModel.RetryPolicy
}
