import Foundation

/// A question about a decision state.
///
/// Each case matches one Jev primitive.
/// Questions are independent:
/// the answer to one question never becomes input to another.
public enum Question: Hashable, Sendable {
    /// A yes-or-no question that returns the probability of "yes".
    case binary(instructions: String, criteria: BinaryCriteria? = nil)

    /// A question that selects one option from a set.
    case choice(instructions: String, options: [ChoiceOption])

    /// A question that places the state on an ordinal scale.
    ///
    /// Levels are ordered from lowest (index 0) to highest.
    case score(instructions: String, levels: [String])

    /// The question instructions.
    var instructions: String {
        switch self {
        case .binary(let instructions, _),
            .choice(let instructions, _),
            .score(let instructions, _):
            return instructions
        }
    }

    /// Checks that the question is well formed.
    ///
    /// - Parameter index: The position of the question in its batch, used in the error.
    /// - Throws: ``DecisionError/invalidQuestion(index:reason:)``
    ///   if the question has no instructions, no options, duplicate option names,
    ///   or fewer than two score levels.
    func validate(index: Int) throws {
        guard !instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DecisionError.invalidQuestion(index: index, reason: "The instructions are empty.")
        }

        switch self {
        case .binary:
            break
        case .choice(_, let options):
            guard !options.isEmpty else {
                throw DecisionError.invalidQuestion(index: index, reason: "A choice needs at least one option.")
            }
            var seen = Set<String>()
            for option in options {
                guard !option.name.isEmpty else {
                    throw DecisionError.invalidQuestion(index: index, reason: "An option name is empty.")
                }
                guard seen.insert(option.name).inserted else {
                    throw DecisionError.invalidQuestion(
                        index: index,
                        reason: "The option name \"\(option.name)\" occurs more than once."
                    )
                }
            }
        case .score(_, let levels):
            guard levels.count >= 2 else {
                throw DecisionError.invalidQuestion(index: index, reason: "A score needs at least two levels.")
            }
        }
    }
}

/// Descriptions of what makes a binary question true or false.
public struct BinaryCriteria: Hashable, Sendable {
    /// The description of a true answer.
    public var whenTrue: String

    /// The description of a false answer.
    public var whenFalse: String

    /// Creates binary criteria.
    ///
    /// - Parameters:
    ///   - whenTrue: The description of a true answer.
    ///   - whenFalse: The description of a false answer.
    public init(whenTrue: String, whenFalse: String) {
        self.whenTrue = whenTrue
        self.whenFalse = whenFalse
    }
}

/// One option of a choice question.
public struct ChoiceOption: Hashable, Sendable {
    /// The option name, which is also the key in the answer probabilities.
    public var name: String

    /// An optional description of when to select the option.
    public var description: String?

    /// Creates a choice option.
    ///
    /// - Parameters:
    ///   - name: The option name.
    ///   - description: An optional description of when to select the option.
    public init(_ name: String, description: String? = nil) {
        self.name = name
        self.description = description
    }
}

// MARK: - Typed options

/// A type whose values are the options of a choice question.
///
/// Conform an enumeration to this protocol
/// to ask for a typed choice with ``DecisionSession/choice(_:from:)``.
/// String-backed enumerations use their raw values as option names.
///
/// ```swift
/// enum Department: String, Choosable {
///     case billing, technical, sales
///
///     var optionDescription: String? {
///         switch self {
///         case .billing: "payments, charges, refunds, invoices"
///         case .technical: "bugs, outages, errors, login problems"
///         case .sales: "pricing questions, upgrades, new purchases"
///         }
///     }
/// }
/// ```
public protocol Choosable: CaseIterable, Hashable, Sendable {
    /// The option name sent to the model. It must be unique among all cases.
    var optionName: String { get }

    /// An optional description of when to select the option.
    var optionDescription: String? { get }
}

extension Choosable {
    public var optionDescription: String? { nil }
}

extension Choosable where Self: RawRepresentable, RawValue == String {
    public var optionName: String { rawValue }
}
