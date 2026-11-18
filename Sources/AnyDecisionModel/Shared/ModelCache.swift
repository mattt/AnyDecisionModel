import Foundation

/// Loads each model once and shares the result across sessions.
///
/// Callers that ask for a key while it loads wait for the same load.
/// A failed load is not kept, so the next caller tries again.
final class ModelCache<Value: Sendable>: Sendable {
    /// Entries and loaded keys share one lock so removal cannot interleave with a completed load.
    private struct Storage: Sendable {
        var entries: [String: Task<Value, Error>] = [:]
        var loaded: Set<String> = []
    }

    private let storage = Locked(Storage())

    /// Returns the value for a key, loading it if needed.
    ///
    /// - Parameters:
    ///   - key: The key that identifies the model.
    ///   - loader: A function that loads the value. It runs at most once at a time for each key.
    func value(
        for key: String,
        loader: @escaping @Sendable () async throws -> Value
    ) async throws -> Value {
        let task = storage.withLock { storage in
            if let task = storage.entries[key] {
                return task
            }
            let task = Task { try await loader() }
            storage.entries[key] = task
            return task
        }

        do {
            // Waiting callers do not cancel the shared load.
            let value = try await task.value
            storage.withLock { storage in
                if storage.entries[key] == task { _ = storage.loaded.insert(key) }
            }
            try Task.checkCancellation()
            return value
        } catch {
            if !(error is CancellationError) || task.isCancelled {
                storage.withLock { storage in
                    if storage.entries[key] == task { storage.entries[key] = nil }
                }
            }
            throw error
        }
    }

    /// Returns whether the value for a key has finished loading.
    func isLoaded(_ key: String) -> Bool {
        storage.withLock { $0.loaded.contains(key) }
    }

    /// Removes the value for a key.
    ///
    /// Callers that already hold the value keep it.
    func remove(_ key: String) {
        let removed = storage.withLock { storage in
            _ = storage.loaded.remove(key)
            return storage.entries.removeValue(forKey: key)
        }
        // Release the value after the lock,
        // because its deinit can call into a native library,
        // which can call back into this process.
        _ = removed
    }

    /// Removes every value.
    func removeAll() {
        let removed = storage.withLock { storage in
            storage.loaded.removeAll()
            defer { storage.entries.removeAll() }
            return storage.entries
        }
        // Release the values after the lock, as in `remove(_:)`.
        _ = removed
    }
}
