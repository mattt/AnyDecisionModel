import Foundation

#if MLX
    /// Whether tests that run MLX on the GPU are enabled.
    ///
    /// Set `ENABLE_MLX_TESTS=1` to run them.
    let shouldRunMLXTests = isMLXEnabled(by: "ENABLE_MLX_TESTS")

    /// Whether tests that download and run a model with a decision head are enabled.
    ///
    /// Set `ENABLE_MLX_HEAD_TESTS=1` to run them. They download Clef-flash 4-bit (6.2 GB).
    let shouldRunMLXHeadTests = isMLXEnabled(by: "ENABLE_MLX_HEAD_TESTS")

    private func isMLXEnabled(by variable: String) -> Bool {
        #if arch(arm64)
            guard ProcessInfo.processInfo.environment[variable] != nil else { return false }
            // SwiftPM puts MLX's Metal library in a resource bundle inside the test bundle.
            // MLX searches registered bundles only, so register the test bundle.
            _ = Bundle(for: TestBundleMarker.self).resourceURL
            return true
        #else
            return false
        #endif
    }

    private final class TestBundleMarker: NSObject {}
#endif
