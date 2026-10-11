import Testing

@testable import AnyDecisionModel

@Suite("ModelCache", .timeLimit(.minutes(1)))
struct ModelCacheTests {
    private actor Counter {
        private(set) var count = 0

        func increment() -> Int {
            count += 1
            return count
        }
    }

    private struct LoadError: Error {}

    private actor SuspendedLoader {
        private var started = false
        private var startWaiter: CheckedContinuation<Void, Never>?
        private var loadWaiter: CheckedContinuation<Void, Never>?

        func load() async -> Int {
            started = true
            startWaiter?.resume()
            startWaiter = nil
            await withCheckedContinuation { loadWaiter = $0 }
            return 1
        }

        func waitUntilStarted() async {
            if started { return }
            await withCheckedContinuation { startWaiter = $0 }
        }

        func finish() {
            loadWaiter?.resume()
            loadWaiter = nil
        }
    }

    @Test func concurrentCallersShareOneLoad() async throws {
        let cache = ModelCache<Int>()
        let counter = Counter()
        let values = try await withThrowingTaskGroup(of: Int.self) { group in
            for _ in 0 ..< 8 {
                group.addTask {
                    try await cache.value(for: "model") {
                        try await Task.sleep(for: .milliseconds(50))
                        return await counter.increment()
                    }
                }
            }
            return try await group.reduce(into: []) { $0.append($1) }
        }
        #expect(values == Array(repeating: 1, count: 8))
        #expect(await counter.count == 1)
        #expect(cache.isLoaded("model"))
        #expect(!cache.isLoaded("other"))
    }

    @Test func removingAKeyLoadsItAgain() async throws {
        let cache = ModelCache<Int>()
        let counter = Counter()
        let load: @Sendable () async throws -> Int = { await counter.increment() }

        #expect(try await cache.value(for: "model", loader: load) == 1)
        cache.remove("model")
        #expect(!cache.isLoaded("model"))
        #expect(try await cache.value(for: "model", loader: load) == 2)

        cache.removeAll()
        #expect(!cache.isLoaded("model"))
        #expect(try await cache.value(for: "model", loader: load) == 3)
    }

    @Test(arguments: [false, true])
    func removedLoadDoesNotMarkKeyLoaded(removeAll: Bool) async throws {
        let cache = ModelCache<Int>()
        let loader = SuspendedLoader()
        let first = Task { try await cache.value(for: "model") { await loader.load() } }
        await loader.waitUntilStarted()

        if removeAll { cache.removeAll() } else { cache.remove("model") }
        #expect(!cache.isLoaded("model"))
        await loader.finish()
        #expect(try await first.value == 1)
        #expect(!cache.isLoaded("model"))
        #expect(try await cache.value(for: "model") { 2 } == 2)
        #expect(cache.isLoaded("model"))
    }

    @Test func failedLoadsAreNotCached() async throws {
        let cache = ModelCache<Int>()
        await #expect(throws: LoadError.self) {
            _ = try await cache.value(for: "model") { throw LoadError() }
        }
        #expect(!cache.isLoaded("model"))
        #expect(try await cache.value(for: "model") { 7 } == 7)
    }
}
