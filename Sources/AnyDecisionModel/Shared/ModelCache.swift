import Foundation

/// Loads each model once and shares the result across sessions.
///
/// Callers that ask for a key while it loads wait for the same load.
/// A failed load is not kept, so the next caller tries again.
final class ModelCache<Value: Sendable>: Sendable {
    private let entries = Locked<[String: Task<Value, Error>]>([:])
    private let loaded = Locked<Set<String>>([])

    /// Returns the value for a key, loading it if needed.
    ///
    /// - Parameters:
    ///   - key: The key that identifies the model.
    ///   - loader: A function that loads the value. It runs at most once at a time for each key.
    func value(
        for key: String,
        loader: @escaping @Sendable () async throws -> Value
    ) async throws -> Value {
        let task = entries.withLock { entries in
            if let task = entries[key] {
                return task
            }
            let task = Task { try await loader() }
            entries[key] = task
            return task
        }

        do {
            // Waiting callers do not cancel the shared load.
            let value = try await task.value
            _ = loaded.withLock { $0.insert(key) }
            try Task.checkCancellation()
            return value
        } catch {
            if !(error is CancellationError) || task.isCancelled {
                entries.withLock { entries in
                    if entries[key] == task { entries[key] = nil }
                }
            }
            throw error
        }
    }

    /// Returns whether the value for a key has finished loading.
    func isLoaded(_ key: String) -> Bool {
        loaded.withLock { $0.contains(key) }
    }

    /// Removes the value for a key.
    ///
    /// Callers that already hold the value keep it.
    func remove(_ key: String) {
        entries.withLock { $0[key] = nil }
        _ = loaded.withLock { $0.remove(key) }
    }

    /// Removes every value.
    func removeAll() {
        entries.withLock { $0.removeAll() }
        loaded.withLock { $0.removeAll() }
    }
}
