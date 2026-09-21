// Adapted from AnyLanguageModel (https://github.com/huggingface/AnyLanguageModel).
// Copyright AnyLanguageModel contributors. Licensed under the Apache License, Version 2.0.

import Foundation

/// Protects shared mutable state behind an `NSLock`.
final class Locked<State> {
    private let lock = NSLock()
    private var state: State

    /// Creates a locked container with the given initial state.
    init(_ state: State) {
        self.state = state
    }

    /// Executes `body` while holding the lock.
    ///
    /// Keep critical sections small and synchronous.
    func withLock<T>(_ body: (inout State) throws -> T) rethrows -> T {
        try lock.withLock { try body(&self.state) }
    }
}

// MARK: - Sendable

extension Locked: @unchecked Sendable where State: Sendable {}
