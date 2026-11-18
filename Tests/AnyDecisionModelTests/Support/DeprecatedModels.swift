@testable import AnyDecisionModel

// Version 0.11 deprecates `JevDecisionModel` and `OllamaDecisionModel`,
// and version 1.0 removes them.
// These aliases shadow the library's types in the test module,
// so that their tests show one deprecation warning for each type
// instead of one for each use.
// Remove the aliases and the tests that use them with the types.

/// The deprecated Jev model, for its tests.
typealias JevDecisionModel = AnyDecisionModel.JevDecisionModel

/// The deprecated Ollama model, for its tests.
typealias OllamaDecisionModel = AnyDecisionModel.OllamaDecisionModel
