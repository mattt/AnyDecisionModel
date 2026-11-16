import Foundation
import Testing

@testable import AnyDecisionModel

#if Llama
    /// A session that a global variable holds for the life of the process.
    nonisolated(unsafe) private var globalExitTestSession: DecisionSession?

    /// Tests that exit a child process with a model loaded.
    ///
    /// Without the exit handler,
    /// llama.cpp's Metal backend stops such a process with an assertion
    /// when a model that a session holds has layers on the GPU,
    /// and an exit during a decision crashes on the GPU and on the CPU.
    @Suite(
        "LlamaDecisionModel at exit",
        .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"] != nil)
    )
    struct LlamaDecisionModelExitTests {
        static var modelPath: String {
            ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"] ?? ""
        }

        /// Returns a session after a first decision,
        /// so that the model and the session's context are loaded.
        static func usedSession(gpuLayers: Int32, path: String = modelPath) async throws -> DecisionSession {
            let model = LlamaDecisionModel(modelPath: path, gpuLayers: gpuLayers)
            let session = DecisionSession(model: model, state: .text("The checkout page returns 500 errors."))
            _ = try await session.probability(of: "Is this about a software error?")
            return session
        }

        @Test("A process exits normally while a session holds a model", arguments: [Int32(-1), 0])
        func exitWithSession(gpuLayers: Int32) async throws {
            let name = "exitWithSession\(gpuLayers)"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .success) { [name = name as String, gpuLayers = gpuLayers as Int32] in
                ExitTestTimer.childStarted(name)
                let session = try await LlamaDecisionModelExitTests.usedSession(gpuLayers: gpuLayers)
                ExitTestTimer.stepStarted(name)
                withExtendedLifetime(session) { exit(0) }
            }
            await ExitTestTimer.finish(name, watchdog, measuring: false)
        }

        @Test("A process exits normally with a model only in the shared cache", arguments: [Int32(-1), 0])
        func exitWithCachedModel(gpuLayers: Int32) async throws {
            let name = "exitWithCachedModel\(gpuLayers)"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .success) { [name = name as String, gpuLayers = gpuLayers as Int32] in
                ExitTestTimer.childStarted(name)
                do { _ = try await LlamaDecisionModelExitTests.usedSession(gpuLayers: gpuLayers) }
                let model = LlamaDecisionModel(modelPath: LlamaDecisionModelExitTests.modelPath, gpuLayers: gpuLayers)
                // A distinct status shows that the cache doesn't hold the model.
                guard model.isLoaded else { exit(4) }
                ExitTestTimer.stepStarted(name)
                exit(0)
            }
            await ExitTestTimer.finish(name, watchdog, measuring: false)
        }

        @Test("A process exits normally with a session in a global variable", arguments: [Int32(-1), 0])
        func exitWithGlobalSession(gpuLayers: Int32) async throws {
            let name = "exitWithGlobalSession\(gpuLayers)"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .success) { [name = name as String, gpuLayers = gpuLayers as Int32] in
                ExitTestTimer.childStarted(name)
                globalExitTestSession = try await LlamaDecisionModelExitTests.usedSession(gpuLayers: gpuLayers)
                ExitTestTimer.stepStarted(name)
                exit(0)
            }
            await ExitTestTimer.finish(name, watchdog, measuring: false)
        }

        @Test(
            "A process exits normally after the model is removed from the cache and released",
            arguments: [Int32(-1), 0]
        )
        func exitAfterRelease(gpuLayers: Int32) async {
            let name = "exitAfterRelease\(gpuLayers)"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .success) { [name = name as String, gpuLayers = gpuLayers as Int32] in
                ExitTestTimer.childStarted(name)
                weak var released: LlamaModel?
                do {
                    let session = try await LlamaDecisionModelExitTests.usedSession(gpuLayers: gpuLayers)
                    let model = LlamaDecisionModel(
                        modelPath: LlamaDecisionModelExitTests.modelPath,
                        gpuLayers: gpuLayers
                    )
                    released = try await model.loadModel()
                    model.removeFromCache()
                    _ = session
                }
                // A distinct status shows that something still holds the model.
                guard released == nil else { exit(4) }
                ExitTestTimer.stepStarted(name)
                exit(0)
            }
            await ExitTestTimer.finish(name, watchdog, measuring: false)
        }

        @Test("After the exit handler runs, a decision throws, and nothing is freed twice")
        func decisionAfterShutdown() async {
            let name = "decisionAfterShutdown"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .success) {
                ExitTestTimer.childStarted("decisionAfterShutdown")
                weak var released: LlamaModel?
                do {
                    let session = try await LlamaDecisionModelExitTests.usedSession(gpuLayers: -1)
                    let model = LlamaDecisionModel(modelPath: LlamaDecisionModelExitTests.modelPath, gpuLayers: -1)
                    released = try await model.loadModel()
                    guard LlamaLifecycle.shared.shutDown() == .freed else { exit(5) }
                    do {
                        _ = try await session.probability(of: "Is this about a payment?")
                        exit(6)
                    } catch DecisionError.modelUnavailable {}
                    do {
                        try await DecisionSession(model: model, state: "ticket").prewarm()
                        exit(7)
                    } catch DecisionError.modelUnavailable {}
                    model.removeFromCache()
                    _ = session
                }
                // The session's and the model's deinits ran after the handler freed them,
                // and freed nothing again.
                guard released == nil else { exit(4) }
                exit(0)
            }
            await ExitTestTimer.finish(name, watchdog, measuring: false)
        }

        @Test("After the exit handler runs, a model that isn't loaded fails to load")
        func loadAfterShutdown() async {
            let name = "loadAfterShutdown"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .success) {
                ExitTestTimer.childStarted("loadAfterShutdown")
                _ = try await LlamaDecisionModelExitTests.usedSession(gpuLayers: -1)
                LlamaLifecycle.shared.shutDown()
                let other = LlamaDecisionModel(modelPath: LlamaDecisionModelExitTests.modelPath, gpuLayers: 0)
                do {
                    _ = try await DecisionSession(model: other, state: "ticket").probability(of: "Is this urgent?")
                    exit(5)
                } catch DecisionError.modelUnavailable {}
                exit(0)
            }
            await ExitTestTimer.finish(name, watchdog, measuring: false)
        }
    }

    #if DEBUG
        /// Exit tests that use ``LlamaTestHooks``, which only debug builds compile.
        extension LlamaDecisionModelExitTests {
            /// Holds work until the exit handler has started,
            /// so that the handler must wait for it.
            static func holdUntilShutdownStarts() {
                let limit = ExitTestTimer.now() + 10_000_000_000
                while !LlamaLifecycle.shared.isStopping {
                    // A distinct status shows that the exit handler never started.
                    if ExitTestTimer.now() > limit { _exit(9) }
                    usleep(1_000)
                }
            }

            /// Returns a hook that runs `body` the first time it is called.
            static func once(_ body: @escaping @Sendable () -> Void) -> @Sendable () -> Void {
                let done = Locked(false)
                return {
                    let isFirst = done.withLock { done in
                        defer { done = true }
                        return !done
                    }
                    if isFirst { body() }
                }
            }

            /// Waits for work to start in another task.
            static func waitForStart(_ started: DispatchSemaphore) {
                // A distinct status shows that the work never started.
                if started.wait(timeout: .now() + 30) == .timedOut { _exit(3) }
            }

            /// Starts a decision in another task,
            /// and returns when its first batch starts.
            ///
            /// - Parameter whileActive: What the decision does at the start of that batch,
            ///   while it is known to be running.
            static func startDecision(
                _ session: DecisionSession,
                whileActive: @escaping @Sendable () -> Void
            ) {
                let started = DispatchSemaphore(value: 0)
                LlamaTestHooks.beforeBatch.withLock {
                    $0 = once {
                        started.signal()
                        whileActive()
                    }
                }
                Task.detached {
                    _ = try? await session.probability(of: "Is this about a payment?")
                }
                waitForStart(started)
            }

            /// Starts a second load of the model in another task,
            /// and returns when the load reports its first progress.
            ///
            /// The load reads the model through a symbolic link,
            /// which has another cache key, so the file loads again.
            static func startSecondLoad(gpuLayers: Int32, whileActive: @escaping @Sendable () -> Void) throws {
                let link = NSTemporaryDirectory() + "llama-exit-test-\(getpid()).gguf"
                try? FileManager.default.removeItem(atPath: link)
                try FileManager.default.createSymbolicLink(atPath: link, withDestinationPath: modelPath)
                let started = DispatchSemaphore(value: 0)
                LlamaTestHooks.onLoadProgress.withLock {
                    $0 = once {
                        started.signal()
                        whileActive()
                    }
                }
                Task.detached {
                    _ = try? await LlamaDecisionModel(modelPath: link, gpuLayers: gpuLayers).loadModel()
                }
                waitForStart(started)
            }

            @Test(
                "The exit handler waits for a decision that is running, and the process exits normally",
                arguments: [Int32(-1), 0]
            )
            func exitDuringDecision(gpuLayers: Int32) async throws {
                let name = "exitDuringDecision\(gpuLayers)"
                let watchdog = ExitTestTimer.watch(name)
                await #expect(processExitsWith: .success) { [name = name as String, gpuLayers = gpuLayers as Int32] in
                    ExitTestTimer.childStarted(name)
                    let session = try await LlamaDecisionModelExitTests.usedSession(gpuLayers: gpuLayers)
                    LlamaDecisionModelExitTests.startDecision(session) {
                        LlamaDecisionModelExitTests.holdUntilShutdownStarts()
                    }
                    ExitTestTimer.stepStarted(name)
                    exit(0)
                }
                await ExitTestTimer.finish(name, watchdog, measuring: false)
            }

            @Test(
                "The exit handler stops a second load that is running, and the process exits normally",
                arguments: [Int32(-1), 0]
            )
            func exitDuringSecondLoad(gpuLayers: Int32) async throws {
                let name = "exitDuringSecondLoad\(gpuLayers)"
                let watchdog = ExitTestTimer.watch(name)
                await #expect(processExitsWith: .success) { [name = name as String, gpuLayers = gpuLayers as Int32] in
                    ExitTestTimer.childStarted(name)
                    let session = try await LlamaDecisionModelExitTests.usedSession(gpuLayers: gpuLayers)
                    try LlamaDecisionModelExitTests.startSecondLoad(gpuLayers: gpuLayers) {
                        LlamaDecisionModelExitTests.holdUntilShutdownStarts()
                    }
                    ExitTestTimer.stepStarted(name)
                    withExtendedLifetime(session) { exit(0) }
                }
                await ExitTestTimer.finish(name, watchdog, measuring: false)
            }

            @Test("A decision that runs past the deadline doesn't keep the process alive")
            func exitPastDeadline() async throws {
                let name = "exitPastDeadline"
                let watchdog = ExitTestTimer.watch(name)
                // The handler frees nothing, so ggml's assertion ends the process,
                // as it would without the handler.
                await #expect(processExitsWith: .signal(SIGABRT)) {
                    ExitTestTimer.childStarted("exitPastDeadline")
                    let session = try await LlamaDecisionModelExitTests.usedSession(gpuLayers: -1)
                    LlamaDecisionModelExitTests.startDecision(session) {
                        while true { sleep(60) }
                    }
                    ExitTestTimer.stepStarted("exitPastDeadline")
                    exit(0)
                }
                let elapsed = try #require(await ExitTestTimer.finish(name, watchdog))
                #expect(elapsed >= .seconds(LlamaLifecycle.exitDeadline))
            }

            @Test("An exit from inside a decision doesn't wait for that decision")
            func exitInsideOperation() async throws {
                let name = "exitInsideOperation"
                let watchdog = ExitTestTimer.watch(name)
                // The handler returns at once and frees nothing,
                // so ggml's assertion ends the process before the deadline.
                // The exit status can't show how long the handler took;
                // `shutDownInsideOperation()` checks that it returns at once.
                await #expect(processExitsWith: .signal(SIGABRT)) {
                    ExitTestTimer.childStarted("exitInsideOperation")
                    let session = try await LlamaDecisionModelExitTests.usedSession(gpuLayers: -1)
                    LlamaDecisionModelExitTests.startDecision(session) {
                        ExitTestTimer.stepStarted("exitInsideOperation")
                        exit(0)
                    }
                    // The decision's thread exits the process.
                    while true { sleep(60) }
                }
                await ExitTestTimer.finish(name, watchdog, measuring: false)
            }

            @Test("A log callback that exits while a model is freed doesn't hang the process")
            func exitFromLogDuringCleanup() async throws {
                // The handler sees that the cleanup's thread is exiting, and returns at once,
                // which `shutDownInsideOperation()` checks without a model.
                // The model is only partly freed, so ggml's assertion ends the process,
                // as it would without the handler.
                let name = "exitFromLogDuringCleanup"
                let watchdog = ExitTestTimer.watch(name)
                await #expect(processExitsWith: .signal(SIGABRT)) {
                    ExitTestTimer.childStarted("exitFromLogDuringCleanup")
                    do {
                        let session = try await LlamaDecisionModelExitTests.usedSession(gpuLayers: -1)
                        LlamaDecisionModel(modelPath: LlamaDecisionModelExitTests.modelPath, gpuLayers: -1)
                            .removeFromCache()
                        LlamaTestHooks.onLog.withLock { $0 = LlamaDecisionModelExitTests.once { exit(0) } }
                        ExitTestTimer.stepStarted("exitFromLogDuringCleanup")
                        // Releasing the last reference frees the context and the model,
                        // and llama.cpp logs while it frees them.
                        _ = session
                    }
                    // A distinct status shows that llama.cpp logged nothing.
                    exit(4)
                }
                await ExitTestTimer.finish(name, watchdog, measuring: false)
            }
        }
    #endif

    /// Lets the parent of an exit test kill a child that hangs,
    /// and measure how long the child takes to end after the step under test.
    ///
    /// Swift Testing's time limit doesn't end a child process,
    /// so a child that hangs would keep the test run waiting.
    /// The child writes its process ID when it starts,
    /// and the monotonic time when the step under test starts,
    /// to files that both processes name from the parent's process ID.
    /// The measurement starts after the child's setup,
    /// so a slow load doesn't change it.
    ///
    /// The measurement ends when the parent sees that the child has ended,
    /// which a busy machine can delay by seconds.
    /// So a test checks only that a time is at least a bound,
    /// which a delay can't break,
    /// and never that a time is less than a bound.
    enum ExitTestTimer {
        /// How long the parent waits for a child before it kills the child.
        static let limit: Duration = .seconds(120)

        /// The current time of a monotonic clock that every process on the system shares.
        static func now() -> UInt64 {
            clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        }

        private static func file(_ name: String, _ suffix: String, parent: pid_t) -> URL {
            URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("llama-exit-test-\(parent)-\(name).\(suffix)")
        }

        /// The exit status of a child that couldn't write a timer file.
        ///
        /// The child also writes the error to standard error.
        static let writeFailureStatus: CInt = 70

        /// In the child: writes `text` to a timer file,
        /// or ends the child with ``writeFailureStatus`` if it can't.
        private static func write(_ text: String, to url: URL) {
            do {
                try text.write(to: url, atomically: true, encoding: .utf8)
            } catch {
                fputs("ExitTestTimer couldn't write \(url.path): \(error)\n", stderr)
                _exit(writeFailureStatus)
            }
        }

        /// In the child: records its process ID.
        static func childStarted(_ name: String) {
            write(String(getpid()), to: file(name, "pid", parent: getppid()))
        }

        /// In the child: records when the step under test starts.
        static func stepStarted(_ name: String) {
            write(String(now()), to: file(name, "start", parent: getppid()))
        }

        private static func read(_ name: String, _ suffix: String) -> String? {
            try? String(contentsOf: file(name, suffix, parent: getpid()), encoding: .utf8)
        }

        /// In the parent: starts a task that kills the child if it hasn't ended after ``limit``,
        /// and returns whether it did.
        static func watch(_ name: String) -> Task<Bool, Never> {
            for suffix in ["pid", "start"] {
                try? FileManager.default.removeItem(at: file(name, suffix, parent: getpid()))
            }
            return Task.detached {
                guard (try? await Task.sleep(for: limit)) != nil else { return false }
                guard let pid = read(name, "pid").flatMap({ pid_t($0) }) else { return false }
                kill(pid, SIGKILL)
                return true
            }
        }

        /// In the parent, after the child ends: stops the watchdog,
        /// records an issue if it killed the child,
        /// and, if `measuring`, returns the time from the step's start to now.
        ///
        /// If the start time can't be read,
        /// this records an issue with the error and returns `nil`.
        /// A child that couldn't write the file
        /// exits with ``writeFailureStatus`` instead.
        @discardableResult
        static func finish(
            _ name: String,
            _ watchdog: Task<Bool, Never>,
            measuring: Bool = true
        ) async -> Duration? {
            watchdog.cancel()
            let killed = await watchdog.value
            #expect(!killed, "The child process didn't end within \(limit).")
            guard measuring else { return nil }
            let url = file(name, "start", parent: getpid())
            let text: String
            do {
                text = try String(contentsOf: url, encoding: .utf8)
            } catch {
                Issue.record("Couldn't read the step's start time from \(url.path): \(error)")
                return nil
            }
            guard let start = UInt64(text) else {
                Issue.record("The step's start time in \(url.path) isn't a number: \(text.debugDescription)")
                return nil
            }
            return .nanoseconds(Int64(now() - start))
        }
    }

    /// Tests of the exit handler's bookkeeping, without a model.
    @Suite("LlamaLifecycle")
    struct LlamaLifecycleTests {
        @Test("Shutting down with no work frees, and a second shutdown does nothing")
        func shutDownIdle() throws {
            let lifecycle = LlamaLifecycle()
            #expect(lifecycle.shutDown(deadline: 0) == .freed)
            #expect(lifecycle.shutDown(deadline: 0) == .alreadyShutDown)
            #expect(lifecycle.isStopping)
            #expect(throws: DecisionError.self) { try lifecycle.perform {} }
            #expect(throws: DecisionError.self) { try lifecycle.stopIfExiting() }
        }

        /// Starts work on another thread that runs until `release` is signaled,
        /// and returns when the work has started.
        static func startWork(_ lifecycle: LlamaLifecycle, until release: DispatchSemaphore) {
            let started = DispatchSemaphore(value: 0)
            Thread.detachNewThread {
                try? lifecycle.perform {
                    started.signal()
                    release.wait()
                }
            }
            started.wait()
        }

        @Test("Shutting down waits for running work, and frees nothing at the deadline")
        func shutDownTimesOut() async throws {
            let lifecycle = LlamaLifecycle()
            let release = DispatchSemaphore(value: 0)
            Self.startWork(lifecycle, until: release)
            let start = ContinuousClock.now
            #expect(lifecycle.shutDown(deadline: 0.2) == .timedOut)
            #expect(ContinuousClock.now - start >= .milliseconds(200))
            release.signal()
            while lifecycle.activeOperationCount > 0 { try await Task.sleep(for: .milliseconds(10)) }
        }

        @Test("Shutting down returns when running work ends")
        func shutDownWaitsForWork() async throws {
            let lifecycle = LlamaLifecycle()
            let release = DispatchSemaphore(value: 0)
            Self.startWork(lifecycle, until: release)
            let result = await withCheckedContinuation { continuation in
                Thread.detachNewThread {
                    continuation.resume(returning: lifecycle.shutDown(deadline: 10))
                }
                usleep(100_000)
                release.signal()
            }
            #expect(result == .freed)
            #expect(lifecycle.activeOperationCount == 0)
        }

        @Test("Shutting down inside work on the current thread returns at once")
        func shutDownInsideOperation() throws {
            let lifecycle = LlamaLifecycle()
            let result = try lifecycle.perform { lifecycle.shutDown(deadline: 10) }
            #expect(result == .insideOperation)
            #expect(lifecycle.activeOperationCount == 0)
        }

        @Test("After the exit handler runs, a missing model file throws the exit error")
        func missingFileAfterShutdown() async {
            await #expect(processExitsWith: .success) {
                LlamaLifecycle.shared.shutDown()
                let model = LlamaDecisionModel(modelPath: "/nonexistent/model.gguf")
                do {
                    _ = try await DecisionSession(model: model, state: "ticket").probability(of: "Is this urgent?")
                    exit(5)
                } catch let error as DecisionError {
                    // A distinct status shows that the file check ran first.
                    guard error == LlamaLifecycle.exiting else { exit(6) }
                }
                exit(0)
            }
        }

        @Test("A failed registration of the exit handler makes the handler free nothing")
        func registrationFailure() {
            let lifecycle = LlamaLifecycle()
            lifecycle.exitHandlerRegistrationFailed()
            #expect(lifecycle.shutDown(deadline: 0) == .registrationFailed)
        }
    }
#endif  // Llama
