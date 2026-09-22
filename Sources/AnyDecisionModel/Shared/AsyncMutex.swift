import Foundation

/// A lock that an async caller can hold across suspension points.
///
/// Waiting callers get the lock in the order in which they asked for it.
/// Cancellation removes a waiting caller without waiting for the current holder.
actor AsyncMutex {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, any Error>
    }

    private var isLocked = false
    private var waiters: [Waiter] = []

    var waitingCount: Int { waiters.count }

    func withLock<T: Sendable>(_ body: @Sendable () async throws -> T) async throws -> T {
        try Task.checkCancellation()
        if isLocked {
            let id = UUID()
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                    if Task.isCancelled {
                        continuation.resume(throwing: CancellationError())
                    } else {
                        waiters.append(Waiter(id: id, continuation: continuation))
                    }
                }
            } onCancel: {
                Task { await self.cancelWaiter(id) }
            }
        }
        isLocked = true
        defer {
            if waiters.isEmpty {
                isLocked = false
            } else {
                // The lock stays locked for the next caller.
                waiters.removeFirst().continuation.resume()
            }
        }
        // Cancellation can race with a handoff, so check again after arranging to release the lock.
        try Task.checkCancellation()
        return try await body()
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        waiters.remove(at: index).continuation.resume(throwing: CancellationError())
    }
}
