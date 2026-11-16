// Adapted in part from AnyLanguageModel (https://github.com/huggingface/AnyLanguageModel).
// Copyright AnyLanguageModel contributors. Licensed under the Apache License, Version 2.0.

import Foundation

#if Llama
    import LlamaSwift

    extension LlamaRuntime {
        /// Registers ``LlamaLifecycle/shutDown(deadline:)`` to run when the process exits.
        ///
        /// ggml creates static objects as it needs them,
        /// such as its Metal device and buffer types during a model's first load,
        /// and the C runtime destroys static objects and runs exit handlers
        /// in the reverse order of their creation.
        /// A handler that was registered before the first load
        /// could run after ggml's objects are destroyed,
        /// and freeing a model then would crash.
        /// ggml doesn't report when it creates such an object,
        /// so every load registers the handler again when it ends.
        /// The last registration runs first,
        /// and the handler does nothing after its first run.
        ///
        /// If a registration fails,
        /// an earlier one could run after objects that the models use are destroyed,
        /// so the handler then frees nothing.
        static func registerExitHandler() {
            if atexit({ _ = LlamaLifecycle.shared.shutDown() }) != 0 {
                LlamaLifecycle.shared.exitHandlerRegistrationFailed()
            }
        }
    }

    /// Tracks the llama.cpp work in this process,
    /// and tries to free every loaded model when the process exits.
    ///
    /// ggml's Metal backend frees its device in a C++ static destructor at exit,
    /// and it stops the process with an assertion
    /// if a model or a context still has buffers on the device then.
    /// A decision that is still running at exit can also crash,
    /// on the GPU or the CPU,
    /// because the static objects that it uses are destroyed under it.
    ///
    /// The exit handler is best effort.
    /// It must never make an exit worse than an exit without it,
    /// so it never waits without a limit,
    /// and it never frees a model or a context that running work could use.
    /// It stops new loads and decisions,
    /// and it waits up to ``exitDeadline`` for the ones that are running:
    /// a load stops at its next progress report,
    /// and a decision stops before its next batch of prompt tokens.
    /// If no work is running then,
    /// it frees every session context and every model
    /// of every loaded ``LlamaDecisionModel``,
    /// including models that the shared cache or sessions still hold.
    /// Otherwise it frees nothing.
    /// Freeing only the idle models wouldn't help,
    /// because the busy model's buffers would still cause the assertion.
    ///
    /// The handler can't help, and the assertion can still happen,
    /// in these cases:
    ///
    /// - An exit during the first load,
    ///   because the handler is registered when a load ends.
    ///   An earlier registration could run after ggml's objects are destroyed.
    /// - An exit from inside a load, a decision, or a cleanup,
    ///   such as from a llama.cpp callback.
    ///   On a thread that this package marked as running that work,
    ///   the handler sees that the work can't end while it waits,
    ///   and returns at once without freeing anything.
    ///   On a thread that llama.cpp or another library owns,
    ///   such as one of ggml's worker threads,
    ///   the handler can't tell,
    ///   so it waits until the deadline and then frees nothing.
    /// - Work that is still running at the deadline.
    /// - An exit that runs no exit handlers, such as `_exit` or a crash.
    ///
    /// After the handler starts, every load and decision throws
    /// ``DecisionError/modelUnavailable(_:)``.
    ///
    /// All llama.cpp work in this package is synchronous,
    /// so an operation belongs to the thread that runs it.
    ///
    /// Nothing here calls llama.cpp with a lock held.
    /// llama.cpp logs while it frees a model,
    /// and runs the log callback on the same thread,
    /// so a callback that calls `exit` would run the exit handler on that thread,
    /// and a lock that isn't recursive would then never be released.
    final class LlamaLifecycle: @unchecked Sendable {
        static let shared = LlamaLifecycle()

        /// How long the exit handler waits for running loads and decisions.
        ///
        /// A running decision stops before its next batch of prompt tokens,
        /// so the wait is about one call to `llama_decode`
        /// with at most ``LlamaDecisionModel/batchSize`` tokens,
        /// which takes milliseconds to a few hundred milliseconds for a 4B model on the GPU.
        /// On the CPU, a batch of 512 tokens can take several seconds,
        /// and the handler then frees nothing.
        /// Three seconds leaves room for a much larger model or a slow CPU,
        /// and still ends a process that can't finish its work
        /// soon enough that the delay looks like a slow quit, not a hang.
        static let exitDeadline: TimeInterval = 3

        /// How long the exit handler's cleanup may take before the process aborts.
        ///
        /// The cleanup frees every model after running work has ended.
        /// llama.cpp calls the installed log callback while it frees,
        /// and another library can replace that callback with one that blocks,
        /// for example on a lock that the exiting thread holds,
        /// and a GPU driver can stall while it releases buffers.
        /// Without a limit the process would then hang,
        /// where without the handler it would have stopped with ggml's assertion.
        /// So the cleanup runs under a watchdog,
        /// which calls `abort()` if the cleanup is still running after this time.
        /// It tries to write one line to standard error first,
        /// but it aborts whether or not the line can be written.
        ///
        /// Freeing Qwen3-4B (2.5 GB) and one session context
        /// took about 20 milliseconds on the GPU and 40 to 70 milliseconds on the CPU
        /// on an Apple silicon Mac.
        /// The work grows with the number and size of the buffers,
        /// so a 70B model would take about a second by that measure.
        /// Ten seconds leaves a margin of about ten times that
        /// for a slow but working free.
        static let cleanupDeadline: TimeInterval = 10

        /// What ``shutDown(deadline:)`` did.
        enum ShutdownResult: Equatable {
            /// It freed every model.
            case freed
            /// It had already run, so it did nothing.
            case alreadyShutDown
            /// It ran on a thread inside an operation, so it freed nothing without waiting.
            case insideOperation
            /// Work was still running at the deadline, so it freed nothing.
            case timedOut
            /// A registration of the exit handler failed, so it freed nothing.
            case registrationFailed
        }

        /// The key in a thread's dictionary for the number of operations that the thread runs.
        private static let depthKey = "AnyDecisionModel.LlamaLifecycle.operationDepth"

        /// Guards the properties below, and signals when no operation is running.
        private let condition = NSCondition()
        private var isShuttingDown = false
        private var registrationFailed = false
        private var activeOperations = 0
        private var resources: [ObjectIdentifier: LlamaModelResources] = [:]

        /// Whether the process is exiting.
        ///
        /// Decisions read this before each batch of prompt tokens,
        /// which costs one uncontended lock and unlock.
        var isStopping: Bool {
            condition.withLock { isShuttingDown }
        }

        /// The number of loads, decisions, and cleanups that are running.
        var activeOperationCount: Int {
            condition.withLock { activeOperations }
        }

        /// The error that work throws after the exit handler starts.
        static var exiting: DecisionError {
            .modelUnavailable("The process is exiting, so llama.cpp work has stopped.")
        }

        /// Throws ``exiting`` if the exit handler has started.
        func stopIfExiting() throws(DecisionError) {
            if isStopping { throw Self.exiting }
        }

        /// Runs a load, a decision, or other synchronous llama.cpp work,
        /// unless the process is exiting.
        ///
        /// - Throws: ``exiting`` if the exit handler has started,
        ///   or the error that `body` throws.
        func perform<Result>(_ body: () throws -> Result) throws -> Result {
            try condition.withLock {
                guard !isShuttingDown else { throw Self.exiting }
                activeOperations += 1
            }
            return try runCounted(body)
        }

        /// Runs `body` as work that the caller has already counted,
        /// marks the current thread as running it,
        /// and stops counting it when it returns.
        private func runCounted<Result>(_ body: () throws -> Result) rethrows -> Result {
            // The body is synchronous, so it stays on this thread.
            let thread = Thread.current.threadDictionary
            let depth = thread[Self.depthKey] as? Int ?? 0
            thread[Self.depthKey] = depth + 1
            defer {
                thread[Self.depthKey] = depth
                condition.withLock {
                    activeOperations -= 1
                    if activeOperations == 0 { condition.broadcast() }
                }
            }
            return try body()
        }

        /// Whether the current thread is running an operation.
        private var isInsideOperation: Bool {
            (Thread.current.threadDictionary[Self.depthKey] as? Int ?? 0) > 0
        }

        /// Records that a registration of the exit handler failed.
        func exitHandlerRegistrationFailed() {
            condition.withLock { registrationFailed = true }
        }

        // MARK: Resources

        /// Records the resources of a model, so that ``shutDown(deadline:)`` can free them.
        func track(_ model: LlamaModelResources) {
            condition.withLock { resources[ObjectIdentifier(model)] = model }
        }

        /// Frees the resources of a model, unless ``shutDown(deadline:)`` freed them already.
        ///
        /// The cleanup runs even while the process is exiting.
        /// It counts as running work,
        /// so that the exit handler's thread check and deadline apply to it,
        /// and it calls llama.cpp without the lock.
        func free(_ model: LlamaModelResources) {
            condition.withLock {
                resources[ObjectIdentifier(model)] = nil
                activeOperations += 1
            }
            runCounted { model.free() }
        }

        /// Frees a session context of a model,
        /// unless ``shutDown(deadline:)`` freed it already.
        ///
        /// The cleanup runs even while the process is exiting, as in ``free(_:)``.
        func free(_ context: OpaquePointer, of model: LlamaModelResources) {
            condition.withLock { activeOperations += 1 }
            runCounted { model.free(context) }
        }

        /// Stops new work, waits up to `deadline` for the work that is running,
        /// and frees the resources of every model if no work is running then.
        @discardableResult
        func shutDown(deadline: TimeInterval = exitDeadline) -> ShutdownResult {
            let insideOperation = isInsideOperation
            var models: [LlamaModelResources.Taken] = []
            let result: ShutdownResult = condition.withLock {
                guard !isShuttingDown else { return .alreadyShutDown }
                isShuttingDown = true
                // The operation on this thread can't end while its thread waits here.
                guard !insideOperation else { return .insideOperation }
                guard !registrationFailed else { return .registrationFailed }
                // Measure the deadline with a monotonic clock,
                // and wait in short steps that start from the current wall-clock time,
                // so that a change to the wall clock can't extend the wait.
                let clock = ContinuousClock()
                let end = clock.now + .seconds(deadline)
                while activeOperations > 0 {
                    let remaining = clock.now.duration(to: end)
                    guard remaining > .zero else { break }
                    let step = min(remaining, .milliseconds(100))
                    _ = condition.wait(until: Date(timeIntervalSinceNow: step / .seconds(1)))
                }
                guard activeOperations == 0 else { return .timedOut }
                // Take the pointers now, under the lock,
                // so that a cleanup that starts after this point frees nothing.
                models = resources.values.compactMap { $0.take() }
                resources = [:]
                return .freed
            }
            // Free without the lock, as in `free(_:)`.
            // If a log callback calls `exit` here, this handler runs again and does nothing.
            guard !models.isEmpty else { return result }
            Self.withAbortDeadline(Self.cleanupDeadline) {
                for model in models {
                    model.free()
                }
            }
            return result
        }

        /// The state of a cleanup that runs under ``withAbortDeadline(_:_:)``.
        private enum CleanupState: Sendable {
            case running
            case completed
            case aborting
        }

        /// Runs `body`, and aborts the process if `body` is still running after `deadline`.
        ///
        /// A separate thread measures the time on a monotonic clock,
        /// so the exiting thread can't stop it.
        /// The thread ends as soon as `body` returns,
        /// and doesn't keep the process alive.
        ///
        /// Completion and the timeout are decided in one step under a lock.
        /// Once `body` has returned and completion is recorded,
        /// the watchdog never aborts.
        /// Once the watchdog has recorded the timeout,
        /// this function aborts instead of returning,
        /// even if `body` returns after that.
        static func withAbortDeadline<Result>(_ deadline: TimeInterval, _ body: () -> Result) -> Result {
            let state = Locked(CleanupState.running)
            let finished = DispatchSemaphore(value: 0)
            let watchdog = Thread {
                guard finished.wait(timeout: .now() + deadline) == .timedOut else { return }
                #if DEBUG
                    LlamaTestHooks.cleanupWatchdog.withLock { $0 }?(.beforeDecision)
                #endif
                let aborts = state.withLock { state in
                    guard state == .running else { return false }
                    state = .aborting
                    return true
                }
                #if DEBUG
                    LlamaTestHooks.cleanupWatchdog.withLock { $0 }?(.decided(aborts: aborts))
                #endif
                if aborts {
                    abortAfterExitCleanupTimeout()
                }
            }
            watchdog.start()
            let result = body()
            let completed = state.withLock { state in
                guard state == .running else { return false }
                state = .completed
                return true
            }
            // The watchdog has decided to abort, and is about to.
            guard completed else { abortAfterLateExitCleanup() }
            finished.signal()
            return result
        }

        /// Aborts the process because the exit handler's cleanup returned
        /// after the watchdog had decided to abort.
        ///
        /// A crash report for this abort shows this function
        /// in the backtrace of the thread that called `abort()`.
        /// It writes no message and doesn't wait for the watchdog.
        @inline(never)
        private static func abortAfterLateExitCleanup() -> Never {
            abort()
        }

        /// Aborts the process because the exit handler's cleanup took too long.
        ///
        /// A crash report for this abort shows this function
        /// in the backtrace of the thread that called `abort()`,
        /// which identifies it.
        /// The function also tries to write one line to standard error,
        /// from another thread and with `write(2)`,
        /// but waits at most 100 milliseconds for it:
        /// standard error can be a full pipe that nobody reads,
        /// and its lock can be held by the blocked thread,
        /// so the abort must not depend on the message.
        @inline(never)
        private static func abortAfterExitCleanupTimeout() -> Never {
            let written = DispatchSemaphore(value: 0)
            let writer = Thread {
                let message = "AnyDecisionModel: freeing llama.cpp models at exit didn't finish in time; aborting.\n"
                _ = message.withCString { write(STDERR_FILENO, $0, strlen($0)) }
                written.signal()
            }
            writer.start()
            _ = written.wait(timeout: .now() + .milliseconds(100))
            abort()
        }
    }

    /// A loaded llama.cpp model and the session contexts created from it.
    ///
    /// ``LlamaLifecycle`` holds these resources until it frees them,
    /// separately from the `LlamaModel` and the sessions that use them,
    /// so that the exit handler can free a model that a Swift object still holds,
    /// and so that a later deinit frees nothing twice.
    final class LlamaModelResources: @unchecked Sendable {
        let model: OpaquePointer
        let vocab: OpaquePointer

        /// Guards `contexts` and `isFreed`.
        private let lock = NSLock()
        private var contexts: [OpaquePointer] = []
        private var isFreed = false

        init(model: OpaquePointer, vocab: OpaquePointer) {
            self.model = model
            self.vocab = vocab
        }

        /// Records a session context, which ``free()`` frees before the model.
        ///
        /// Call it in the same operation that creates the context,
        /// so that the exit handler can't free the model in between.
        func add(_ context: OpaquePointer) {
            lock.withLock { contexts.append(context) }
        }

        /// The pointers that one call must free, in order.
        struct Taken {
            let model: OpaquePointer
            let contexts: [OpaquePointer]

            /// Frees the contexts, then the model.
            func free() {
                for context in contexts {
                    llama_free(context)
                }
                llama_model_free(model)
            }
        }

        /// Returns the pointers to free and marks them freed,
        /// or returns `nil` if they were taken already.
        func take() -> Taken? {
            lock.withLock {
                guard !isFreed else { return nil }
                isFreed = true
                defer { contexts = [] }
                return Taken(model: model, contexts: contexts)
            }
        }

        /// Frees the contexts and the model, once.
        ///
        /// Call it only through ``LlamaLifecycle``,
        /// when no load or decision uses the model.
        func free() {
            // Take the pointers under the lock, and call llama.cpp without it.
            take()?.free()
        }

        /// Frees one session context, unless the model's resources were taken already.
        ///
        /// Call it only through ``LlamaLifecycle``,
        /// when no decision uses the context.
        func free(_ context: OpaquePointer) {
            let owned: Bool = lock.withLock {
                guard !isFreed, let index = contexts.firstIndex(of: context) else { return false }
                contexts.remove(at: index)
                return true
            }
            if owned {
                llama_free(context)
            }
        }
    }

    #if DEBUG
        /// Functions that tests run at points inside llama.cpp work.
        ///
        /// Only debug builds, which `swift test` uses, compile these hooks,
        /// so they cost nothing in a release build.
        enum LlamaTestHooks {
            /// Runs at the start of each batch of a decision's prompt,
            /// before its check for exit, on the decision's thread.
            ///
            /// After the hook returns,
            /// the decision does no llama.cpp work before that check,
            /// so a decision that the hook holds until the exit handler starts
            /// stops at once.
            static let beforeBatch = Locked<(@Sendable () -> Void)?>(nil)

            /// Runs for each progress report of a load, on the load's thread.
            static let onLoadProgress = Locked<(@Sendable () -> Void)?>(nil)

            /// Runs for each llama.cpp log message, before the level filter,
            /// on the thread that logs it.
            static let onLog = Locked<(@Sendable () -> Void)?>(nil)

            /// A point in the cleanup watchdog after its deadline passes.
            enum CleanupWatchdogEvent: Sendable, Equatable {
                /// The watchdog is about to decide between completion and the timeout.
                case beforeDecision
                /// The watchdog has decided, and aborts if `aborts` is true.
                case decided(aborts: Bool)
            }

            /// Runs on the cleanup watchdog's thread after its deadline passes.
            static let cleanupWatchdog = Locked<(@Sendable (CleanupWatchdogEvent) -> Void)?>(nil)
        }
    #endif
#endif  // Llama
