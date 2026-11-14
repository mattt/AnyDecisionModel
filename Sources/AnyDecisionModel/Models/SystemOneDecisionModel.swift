import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// A decision model that calls a server that implements the System One API.
///
/// The model sends each batch as a JSON `POST` request to ``path`` under ``baseURL``,
/// with the model identifier, the state,
/// the questions keyed by their positions,
/// and the fields in ``extraBody``.
/// The body's object keys are sorted,
/// so the same request always has the same bytes.
/// It returns the server's probabilities and confidence values unchanged.
///
/// Use ``jev(baseURL:apiKey:modelID:session:retryPolicy:)``
/// or ``ollama(modelID:baseURL:apiKey:keepAlive:session:)``
/// for a model that is configured for Jev or Ollama.
/// For another server, pass its configuration to the initializer:
///
/// ```swift
/// let model = try SystemOneDecisionModel(
///     baseURL: URL(string: "http://127.0.0.1:8000")!,
///     modelID: "my-model",
///     capabilities: DecisionCapabilities(maximumQuestions: 64)
/// )
/// let session = DecisionSession(model: model, state: .text(ticket))
/// let urgent = try await session.probability(of: "Does this convey urgency?")
/// ```
///
/// An API key is sent only in the `Authorization` header
/// and is never included in descriptions or errors.
/// The model sends text and JSON states only;
/// a state with attachments is rejected before any request is sent.
public struct SystemOneDecisionModel: DecisionModel, CustomStringConvertible {
    /// The default path of the System One endpoint.
    public static let defaultPath = "v1/systemone"

    /// The request body fields that the model writes.
    ///
    /// ``extraBody`` cannot contain these keys.
    public static let reservedBodyFields: Set<String> = ["model", "state", "questions"]

    /// The request headers that the model writes.
    ///
    /// ``headers`` cannot contain these names, in any letter case.
    public static let reservedHeaders: Set<String> = ["Authorization", "Content-Type"]

    /// The API base URL.
    public let baseURL: URL

    /// The path of the System One endpoint, relative to ``baseURL``.
    public let path: String

    /// The model identifier that the request sends.
    public let modelID: String

    /// Extra request headers.
    public let headers: [String: String]

    /// The limits and features of the server.
    public let capabilities: DecisionCapabilities

    /// Extra top-level fields that the model adds to each request body.
    public let extraBody: [String: JSONValue]

    /// The retry behavior for eligible HTTP failures.
    public let retryPolicy: RetryPolicy

    private let tokenProvider: @Sendable () -> String?
    private let urlSession: URLSession

    /// Whether a request without an API key throws ``DecisionError/missingCredentials``
    /// before it is sent.
    ///
    /// Only the Jev configuration sets this,
    /// to keep the behavior of ``JevDecisionModel``.
    let requiresAPIKey: Bool

    /// Creates a model for a server that implements the System One API.
    ///
    /// `extraBody` cannot set a field in ``reservedBodyFields``,
    /// and `headers` cannot set a header in ``reservedHeaders``.
    ///
    /// - Parameters:
    ///   - baseURL: The API base URL.
    ///   - path: The path of the System One endpoint, relative to `baseURL`.
    ///   - modelID: The model identifier that the request sends.
    ///   - apiKey: The API key, or a closure that returns it.
    ///     The model sends the key as a bearer token.
    ///     If the key is `nil` or empty, the model sends no `Authorization` header.
    ///   - headers: Extra request headers.
    ///   - capabilities: The limits and features of the server.
    ///     The default has no limits and an unspecified probability source.
    ///   - extraBody: Extra top-level fields to add to each request body.
    ///   - session: The URL session for requests.
    ///   - retryPolicy: The retry behavior for eligible HTTP failures.
    /// - Throws: ``ConfigurationError`` if `extraBody` sets a reserved field
    ///   or `headers` sets a reserved header.
    public init(
        baseURL: URL,
        path: String = defaultPath,
        modelID: String,
        apiKey tokenProvider: @escaping @autoclosure @Sendable () -> String? = nil,
        headers: [String: String] = [:],
        capabilities: DecisionCapabilities = DecisionCapabilities(),
        extraBody: [String: JSONValue] = [:],
        session: URLSession = URLSession(configuration: .default),
        retryPolicy: RetryPolicy = .default
    ) throws {
        if let key = Self.reservedBodyField(in: extraBody) {
            throw ConfigurationError(kind: .reservedBodyField, key: key)
        }
        if let name = Self.reservedHeader(in: headers) {
            throw ConfigurationError(kind: .reservedHeader, key: name)
        }
        self.init(
            baseURL: baseURL,
            path: path,
            modelID: modelID,
            tokenProvider: tokenProvider,
            requiresAPIKey: false,
            headers: headers,
            capabilities: capabilities,
            extraBody: extraBody,
            session: session,
            retryPolicy: retryPolicy
        )
    }

    /// Creates a model without checking for reserved keys.
    ///
    /// The Jev and Ollama configurations use this initializer,
    /// because they never set a reserved key.
    /// The request encoder also skips reserved body fields.
    init(
        baseURL: URL,
        path: String,
        modelID: String,
        tokenProvider: @escaping @Sendable () -> String?,
        requiresAPIKey: Bool,
        headers: [String: String],
        capabilities: DecisionCapabilities,
        extraBody: [String: JSONValue],
        session: URLSession,
        retryPolicy: RetryPolicy
    ) {
        assert(Self.reservedBodyField(in: extraBody) == nil && Self.reservedHeader(in: headers) == nil)

        self.baseURL = baseURL
        self.path = path
        self.modelID = modelID
        self.tokenProvider = tokenProvider
        self.requiresAPIKey = requiresAPIKey
        self.headers = headers
        self.capabilities = capabilities
        self.extraBody = extraBody
        self.urlSession = session
        self.retryPolicy = retryPolicy
    }

    /// Returns a key in a request body that the model writes, or `nil` if there is none.
    static func reservedBodyField(in body: [String: JSONValue]) -> String? {
        body.keys.sorted().first { reservedBodyFields.contains($0) }
    }

    /// Returns a header name that the model writes, or `nil` if there is none.
    ///
    /// HTTP header names are case-insensitive,
    /// so the comparison ignores letter case.
    static func reservedHeader(in headers: [String: String]) -> String? {
        let reserved = Set(reservedHeaders.map { $0.lowercased() })
        return headers.keys.sorted().first { reserved.contains($0.lowercased()) }
    }

    /// The URL of the System One endpoint.
    ///
    /// Each component of ``path`` is appended to ``baseURL`` in turn,
    /// so leading, trailing, and repeated slashes have no effect.
    var endpoint: URL {
        path.split(separator: "/").reduce(baseURL) { url, component in
            url.appendingPathComponent(String(component))
        }
    }

    public var description: String {
        "SystemOneDecisionModel(modelID: \(modelID), baseURL: \(baseURL.absoluteString), path: \(path))"
    }

    public func makeExecutor() -> any DecisionModelExecutor {
        Executor(model: self)
    }

    private struct Executor: DecisionModelExecutor {
        let model: SystemOneDecisionModel

        func decide(_ questions: [Question], about state: DecisionState) async throws
            -> DecisionSession.Response
        {
            // System One keys questions and answers by identifier. Use each question's position.
            let keys = questions.indices.map(String.init)
            let request = SystemOneRequest(
                model: model.modelID,
                state: state,
                questions: Dictionary(uniqueKeysWithValues: zip(keys, questions.map(SystemOneQuestion.init))),
                extraBody: model.extraBody
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
        // Version 0.8's Jev backend read the API key before it encoded the body
        // and checked for cancellation before every attempt.
        // Its Ollama backend encoded the body first
        // and sent the first attempt without a check.
        // The Jev configuration keeps the first order;
        // every other configuration keeps the second.
        var apiKey: String?
        if requiresAPIKey {
            apiKey = readAPIKey()
            guard apiKey != nil else {
                throw DecisionError.missingCredentials
            }
        }

        let url = endpoint
        // `JSONEncoder` writes object keys in an order that can change between encodings.
        // Sorted keys make the body the same for the same request,
        // which helps caching and comparison.
        // Key order has no meaning in JSON, so servers read the same request.
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let body = try encoder.encode(request)

        if !requiresAPIKey {
            apiKey = readAPIKey()
        }
        var headers = self.headers
        if let apiKey {
            headers["Authorization"] = "Bearer \(apiKey)"
        }

        var retrier = retryPolicy.makeIterator()
        var isRetry = false
        while true {
            if requiresAPIKey || isRetry {
                try Task.checkCancellation()
            }
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
                isRetry = true
            }
        }
    }

    /// Returns the API key, or `nil` if it is missing or empty.
    private func readAPIKey() -> String? {
        guard let apiKey = tokenProvider(), !apiKey.isEmpty else { return nil }
        return apiKey
    }
}

// MARK: - Configurations

extension SystemOneDecisionModel {
    /// Returns a model for the TypeSafe Jev API.
    ///
    /// The model has the same configuration and behavior as ``JevDecisionModel``.
    /// A request without an API key throws ``DecisionError/missingCredentials``
    /// before it is sent.
    ///
    /// - Parameters:
    ///   - baseURL: The API base URL.
    ///   - apiKey: The API key, or a closure that returns it.
    ///     Defaults to the value of `TYPESAFE_AI_API_KEY` or `TYPESAFE_API_KEY`.
    ///   - modelID: The Jev model identifier.
    ///   - session: The URL session for requests.
    ///   - retryPolicy: The retry behavior for eligible HTTP failures.
    public static func jev(
        baseURL: URL = JevDecisionModel.defaultBaseURL,
        apiKey tokenProvider: @escaping @autoclosure @Sendable () -> String? = JevDecisionModel.environmentAPIKey(),
        modelID: String = "jev-latest",
        session: URLSession = URLSession(configuration: .default),
        retryPolicy: RetryPolicy = .default
    ) -> SystemOneDecisionModel {
        SystemOneDecisionModel(
            baseURL: baseURL,
            path: defaultPath,
            modelID: modelID,
            tokenProvider: tokenProvider,
            requiresAPIKey: true,
            headers: [:],
            capabilities: DecisionCapabilities(),
            extraBody: [:],
            session: session,
            retryPolicy: retryPolicy
        )
    }

    /// Returns a model for a System One model in Ollama.
    ///
    /// The model has the same configuration and behavior as ``OllamaDecisionModel``.
    /// It sends an API key only when one is set,
    /// sends `keep_alive` only when `keepAlive` is not `nil`,
    /// and does not retry failed requests.
    ///
    /// - Parameters:
    ///   - modelID: The identifier of a model trained for System One, such as `nimble`.
    ///   - baseURL: The base URL for the Ollama server.
    ///   - apiKey: The API key for Ollama's cloud, or a closure that returns it.
    ///     Defaults to the value of `OLLAMA_API_KEY`.
    ///     Local requests do not need a key.
    ///   - keepAlive: How long the model stays loaded after a request, in seconds.
    ///     Zero unloads the model after the request,
    ///     and a negative value keeps it loaded.
    ///     `nil` uses the server's setting.
    ///   - session: The URL session for requests.
    public static func ollama(
        modelID: String,
        baseURL: URL = OllamaDecisionModel.defaultBaseURL,
        apiKey tokenProvider: @escaping @autoclosure @Sendable () -> String? = OllamaDecisionModel.environmentAPIKey(),
        keepAlive: TimeInterval? = nil,
        session: URLSession = URLSession(configuration: .default)
    ) -> SystemOneDecisionModel {
        var extraBody: [String: JSONValue] = [:]
        if let keepAlive {
            extraBody["keep_alive"] = .number(keepAlive)
        }
        return SystemOneDecisionModel(
            baseURL: baseURL,
            path: defaultPath,
            modelID: modelID,
            tokenProvider: tokenProvider,
            requiresAPIKey: false,
            headers: [:],
            capabilities: DecisionCapabilities(probabilitySource: .unspecified),
            extraBody: extraBody,
            session: session,
            retryPolicy: .never
        )
    }
}

// MARK: - Configuration errors

extension SystemOneDecisionModel {
    /// An error in the configuration of a System One model.
    ///
    /// ``SystemOneDecisionModel/init(baseURL:path:modelID:apiKey:headers:capabilities:extraBody:session:retryPolicy:)``
    /// throws this error before it creates the model.
    public struct ConfigurationError: Error, Hashable, Sendable, LocalizedError, CustomStringConvertible {
        /// A kind of configuration error.
        #if hasAttribute(nonexhaustive)
            @nonexhaustive
        #endif
        public enum Kind: Hashable, Sendable {
            /// `extraBody` sets a field in ``SystemOneDecisionModel/reservedBodyFields``.
            case reservedBodyField

            /// `headers` sets a header in ``SystemOneDecisionModel/reservedHeaders``.
            case reservedHeader
        }

        /// The kind of error.
        public let kind: Kind

        /// The body field or header name that is not allowed.
        public let key: String

        /// Creates a configuration error.
        ///
        /// - Parameters:
        ///   - kind: The kind of error.
        ///   - key: The body field or header name that is not allowed.
        public init(kind: Kind, key: String) {
            self.kind = kind
            self.key = key
        }

        public var description: String {
            switch kind {
            case .reservedBodyField:
                return "extraBody cannot set the reserved field \"\(key)\"."
            case .reservedHeader:
                return "headers cannot set the reserved header \"\(key)\"."
            }
        }

        public var errorDescription: String? { description }
    }
}
