import Foundation

/// A configured decision backend.
///
/// A model holds configuration only.
/// It creates an executor for each ``DecisionSession``,
/// and the executor does the work.
public protocol DecisionModel: Sendable {
    /// The model identifier reported in responses.
    var modelID: String { get }

    /// The limits and features of the backend.
    var capabilities: DecisionCapabilities { get }

    /// Creates an executor for one session.
    ///
    /// Executors can keep per-session state, such as a cache for the session state.
    /// Creating an executor must be cheap; load resources lazily.
    func makeExecutor() -> any DecisionModelExecutor
}

/// Evaluates batches of decision questions about one state.
///
/// Executors do not need a causal language model or a KV cache.
/// Each question must be answered independently of the other questions in the batch.
public protocol DecisionModelExecutor: Sendable {
    /// Loads resources and precomputes work for a state.
    func prewarm(for state: DecisionState) async throws

    /// Answers a batch of questions about a state.
    ///
    /// - Parameters:
    ///   - questions: The questions.
    ///   - state: The state that the questions are about.
    /// - Returns: A response with one answer for each question, in the order of the questions.
    func decide(_ questions: [Question], about state: DecisionState) async throws
        -> DecisionSession.Response
}

extension DecisionModelExecutor {
    public func prewarm(for state: DecisionState) async throws {}
}

/// The limits and features of a decision backend.
public struct DecisionCapabilities: Hashable, Sendable {
    /// The largest number of options in a choice question, or `nil` for no local limit.
    public var maximumChoiceOptions: Int?

    /// The largest number of levels in a score question, or `nil` for no local limit.
    public var maximumScoreLevels: Int?

    /// Creates a capabilities description.
    public init(
        maximumChoiceOptions: Int? = nil,
        maximumScoreLevels: Int? = nil
    ) {
        self.maximumChoiceOptions = maximumChoiceOptions
        self.maximumScoreLevels = maximumScoreLevels
    }

    /// Checks a question against the limits.
    ///
    /// - Throws: ``DecisionError/unsupportedQuestion(index:reason:)`` if the question exceeds a limit.
    func check(_ question: Question, index: Int) throws {
        switch question {
        case .binary:
            break
        case .choice(_, let options):
            if let maximum = maximumChoiceOptions, options.count > maximum {
                throw DecisionError.unsupportedQuestion(
                    index: index,
                    reason: "The model supports at most \(maximum) choice options, not \(options.count)."
                )
            }
        case .score(_, let levels):
            if let maximum = maximumScoreLevels, levels.count > maximum {
                throw DecisionError.unsupportedQuestion(
                    index: index,
                    reason: "The model supports at most \(maximum) score levels, not \(levels.count)."
                )
            }
        }
    }
}
