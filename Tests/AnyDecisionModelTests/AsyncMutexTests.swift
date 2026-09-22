import Testing

@testable import AnyDecisionModel

@Suite("AsyncMutex", .timeLimit(.minutes(1)))
struct AsyncMutexTests {
    private actor Signal {
        private(set) var isOpen = false
        private var waiters: [CheckedContinuation<Void, Never>] = []

        func wait() async {
            if isOpen { return }
            await withCheckedContinuation { waiters.append($0) }
        }

        func open() {
            isOpen = true
            let pending = waiters
            waiters.removeAll()
            for waiter in pending { waiter.resume() }
        }
    }

    private func waitForQueue(_ mutex: AsyncMutex, count: Int) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while await mutex.waitingCount != count, ContinuousClock.now < deadline {
            await Task.yield()
        }
        #expect(await mutex.waitingCount == count)
    }

    @Test func canceledWaiterStopsBeforeHolderReleases() async throws {
        let mutex = AsyncMutex()
        let entered = Signal()
        let release = Signal()
        let holder = Task {
            try await mutex.withLock {
                await entered.open()
                await release.wait()
            }
        }
        await entered.wait()
        // Release the holder even if cancellation is broken, so the test can report failure.
        let watchdog = Task {
            try await Task.sleep(for: .seconds(5))
            await release.open()
        }
        defer { watchdog.cancel() }

        let canceled = Task {
            try await mutex.withLock { _ = Issue.record("A canceled waiter entered the lock.") }
        }
        try await waitForQueue(mutex, count: 1)
        let next = Task { try await mutex.withLock { 42 } }
        try await waitForQueue(mutex, count: 2)

        canceled.cancel()
        await #expect(throws: CancellationError.self) { try await canceled.value }
        #expect(await !release.isOpen)
        #expect(await mutex.waitingCount == 1)

        await release.open()
        try await holder.value
        #expect(try await next.value == 42)
        #expect(await mutex.waitingCount == 0)
    }

    @Test func alreadyCanceledTaskDoesNotAcquireLock() async throws {
        let mutex = AsyncMutex()
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            try await mutex.withLock { _ = Issue.record("A canceled task acquired the lock.") }
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(try await mutex.withLock { 42 } == 42)
    }

    @Test func throwingHolderReleasesNextWaiter() async throws {
        enum Failure: Error { case expected }
        let mutex = AsyncMutex()
        let entered = Signal()
        let release = Signal()
        let holder = Task {
            try await mutex.withLock {
                await entered.open()
                await release.wait()
                throw Failure.expected
            }
        }
        await entered.wait()
        let next = Task { try await mutex.withLock { 42 } }
        try await waitForQueue(mutex, count: 1)
        await release.open()
        await #expect(throws: Failure.self) { try await holder.value }
        #expect(try await next.value == 42)
    }
}
