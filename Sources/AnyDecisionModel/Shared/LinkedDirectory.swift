import Foundation

/// Creates a temporary directory with a symbolic link to each file in a directory.
///
/// A model loader can read the linked directory in place of the original,
/// without the excluded files or with replacements for them.
/// The caller removes the directory when it no longer needs it.
///
/// - Parameters:
///   - directory: The directory whose files to link.
///   - excluded: The names of files not to link.
func makeLinkedDirectory(to directory: URL, excluding excluded: Set<String>) throws -> URL {
    let fileManager = FileManager.default
    let linked = fileManager.temporaryDirectory
        .appendingPathComponent("AnyDecisionModel-\(UUID().uuidString)", isDirectory: true)
    try fileManager.createDirectory(at: linked, withIntermediateDirectories: true)
    do {
        for name in try fileManager.contentsOfDirectory(atPath: directory.path) where !excluded.contains(name) {
            try fileManager.createSymbolicLink(
                at: linked.appendingPathComponent(name),
                withDestinationURL: directory.appendingPathComponent(name)
            )
        }
    } catch {
        try? fileManager.removeItem(at: linked)
        throw error
    }
    return linked
}
