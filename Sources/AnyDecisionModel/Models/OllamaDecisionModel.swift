import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// A decision model that calls a System One model in Ollama.
///
/// The model sends each batch to `POST /v1/systemone`
/// and returns Ollama's probabilities and confidence values unchanged.
/// It has the same configuration and behavior as
/// ``SystemOneDecisionModel/ollama(modelID:baseURL:apiKey:keepAlive:session:)``.
/// It needs Ollama v0.35.0 or later,
/// or v0.40.5 or later for cloud models.
///
/// Local requests need no API key.
/// By default, an API key comes from the `OLLAMA_API_KEY` environment variable
/// and is sent only when one is set,
/// for requests to Ollama's cloud at `https://ollama.com`.
/// The key is sent only in the `Authorization` header
/// and is never included in descriptions or errors.
///
/// ```swift
/// let model = OllamaDecisionModel(modelID: "nimble")
/// let session = DecisionSession(model: model, state: .text(ticket))
/// let urgent = try await session.probability(of: "Does this convey urgency?")
/// ```
public struct OllamaDecisionModel: DecisionModel, CustomStringConvertible {
    /// The default base URL for a local Ollama server.
    public static let defaultBaseURL = URL(string: "http://localhost:11434")!

    /// The environment variable that supplies the API key.
    private static let apiKeyEnvironmentVariable = "OLLAMA_API_KEY"

    /// The base URL for the Ollama server.
    public let baseURL: URL

    /// The identifier of a model trained for System One, such as `nimble`.
    ///
    /// Add `:cloud` to run a cloud model through a local server,
    /// or point `baseURL` at `https://ollama.com` and pass an API key.
    public let modelID: String

    /// How long the model stays loaded after a request, in seconds.
    ///
    /// Zero unloads the model after the request.
    /// A negative value keeps it loaded.
    /// `nil` uses the server's setting, which defaults to five minutes.
    public let keepAlive: TimeInterval?

    private let base: SystemOneDecisionModel

    /// Creates an Ollama decision model.
    ///
    /// - Parameters:
    ///   - modelID: The identifier of a model trained for System One, such as `nimble`.
    ///   - baseURL: The base URL for the Ollama server.
    ///   - apiKey: The API key for Ollama's cloud, or a closure that returns it.
    ///     Defaults to the value of `OLLAMA_API_KEY`.
    ///     Local requests do not need a key.
    ///   - keepAlive: How long the model stays loaded after a request, in seconds.
    ///   - session: The URL session for requests.
    public init(
        modelID: String,
        baseURL: URL = defaultBaseURL,
        apiKey tokenProvider: @escaping @autoclosure @Sendable () -> String? = OllamaDecisionModel.environmentAPIKey(),
        keepAlive: TimeInterval? = nil,
        session: URLSession = URLSession(configuration: .default)
    ) {
        self.modelID = modelID
        self.baseURL = baseURL
        self.keepAlive = keepAlive
        self.base = .ollama(
            modelID: modelID,
            baseURL: baseURL,
            apiKey: tokenProvider(),
            keepAlive: keepAlive,
            session: session
        )
    }

    /// Returns the API key from the environment, or `nil` if none is set.
    ///
    /// - Parameter environment: The environment to read.
    public static func environmentAPIKey(
        from environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        guard
            let value = environment[apiKeyEnvironmentVariable]?
                .trimmingCharacters(in: .whitespacesAndNewlines),
            !value.isEmpty
        else { return nil }
        return value
    }

    public var capabilities: DecisionCapabilities {
        base.capabilities
    }

    public var description: String {
        "OllamaDecisionModel(modelID: \(modelID), baseURL: \(baseURL.absoluteString))"
    }

    public func makeExecutor() -> any DecisionModelExecutor {
        base.makeExecutor(readsCustomOptions: false)
    }
}
