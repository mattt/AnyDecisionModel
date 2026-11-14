/// One batch of questions about a state, with the options for the batch.
///
/// A ``DecisionSession`` creates a request for each batch
/// and passes it to ``DecisionModelExecutor/decide(_:)``.
/// A model that wraps other models gets the request as a value,
/// so it can change ``options`` before it passes the request on.
public struct DecisionRequest: Hashable, Sendable {
    /// The state that every question is about.
    public var state: DecisionState

    /// The questions, in order.
    public var questions: [Question]

    /// The options for the batch.
    public var options: DecisionOptions

    /// Creates a request.
    ///
    /// - Parameters:
    ///   - state: The state that every question is about.
    ///   - questions: The questions, in order.
    ///   - options: The options for the batch.
    public init(state: DecisionState, questions: [Question], options: DecisionOptions = DecisionOptions()) {
        self.state = state
        self.questions = questions
        self.options = options
    }
}

/// Options for a batch of decision questions.
///
/// A common option is a hint that a backend can ignore
/// without a change to an answer.
/// No backend in this package reads ``cacheKey`` or ``userID`` yet.
///
/// An option that changes answers or routing,
/// such as a provider's routing preferences,
/// is specific to one backend.
/// Set it with the ``subscript(custom:)`` subscript:
///
/// ```swift
/// var options = DecisionOptions(userID: "user-42")
/// options[custom: SystemOneDecisionModel.self] = .init(
///     extraBody: ["session_id": "triage-7"]
/// )
/// let session = DecisionSession(model: model, state: .text(ticket), options: options)
/// ```
public struct DecisionOptions: Hashable, Sendable {
    /// A key that groups requests which share a prompt prefix,
    /// so that a provider can reuse its cache for them.
    ///
    /// No backend in this package reads this value yet.
    public var cacheKey: String?

    /// A stable identifier for the end user,
    /// which a provider can use to detect abuse.
    ///
    /// Use a value that does not identify the person,
    /// such as a hash of an account identifier.
    /// No backend in this package reads this value yet.
    public var userID: String?

    /// The custom options of each model type.
    private var customOptions: [ObjectIdentifier: AnyCustomDecisionOptions] = [:]

    /// Creates options.
    ///
    /// - Parameters:
    ///   - cacheKey: A key that groups requests which share a prompt prefix.
    ///   - userID: A stable identifier for the end user.
    public init(cacheKey: String? = nil, userID: String? = nil) {
        self.cacheKey = cacheKey
        self.userID = userID
    }

    /// Accesses the options for one model type.
    ///
    /// Only an executor of `Model` reads these options.
    /// Two models of the same type read the same options,
    /// even when they are configured for different servers.
    ///
    /// ```swift
    /// var options = DecisionOptions()
    /// options[custom: SystemOneDecisionModel.self] = .init(
    ///     extraHeaders: ["traceparent": traceparent]
    /// )
    /// ```
    ///
    /// - Parameter model: The model type.
    /// - Returns: The options for `model`, or `nil` if none are set.
    public subscript<Model: DecisionModel>(custom model: Model.Type) -> Model.CustomOptions? {
        get {
            customOptions[ObjectIdentifier(model)]?.value as? Model.CustomOptions
        }
        set {
            customOptions[ObjectIdentifier(model)] = newValue.map(AnyCustomDecisionOptions.init)
        }
    }
}

/// Options that are specific to one type of decision model.
///
/// A model declares its options as its ``DecisionModel/CustomOptions`` type,
/// and callers set them with ``DecisionOptions/subscript(custom:)``:
///
/// ```swift
/// extension MyDecisionModel {
///     public struct CustomOptions: CustomDecisionOptions {
///         public var temperature: Double?
///     }
/// }
/// ```
///
/// A model without options uses `Never`.
public protocol CustomDecisionOptions: Hashable, Sendable {}

extension Never: CustomDecisionOptions {}

/// A type-erased custom options value.
///
/// `AnyHashable` is not `Sendable`,
/// so this wrapper keeps the value as a `Sendable` existential
/// and compares and hashes it through `AnyHashable`.
private struct AnyCustomDecisionOptions: Hashable, Sendable {
    let value: any CustomDecisionOptions

    init(_ value: some CustomDecisionOptions) {
        self.value = value
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        AnyHashable(lhs.value) == AnyHashable(rhs.value)
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(AnyHashable(value))
    }
}
