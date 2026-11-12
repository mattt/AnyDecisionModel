import Foundation

#if canImport(CoreGraphics) && canImport(ImageIO)
    import CoreGraphics
    import ImageIO
#endif

/// The input that a decision session asks questions about.
///
/// A state is immutable for the lifetime of a session.
/// Local models can reuse the computation for the state across questions.
public enum DecisionState: Hashable, Sendable {
    /// A text state.
    case text(String)

    /// A structured state, such as a JSON object or array.
    case json(JSONValue)

    /// A state of text and attachments, such as images, in order.
    ///
    /// Only models whose ``DecisionCapabilities/supportedAttachmentKinds``
    /// include every attachment kind in the state can answer questions about it.
    /// A state with text segments only works with every model,
    /// which reads the text segments joined by newlines.
    ///
    /// ```swift
    /// let state = DecisionState.segments([
    ///     .text("A photo from a customer's damage claim:"),
    ///     .attachment(Attachment(imageData: jpeg, mimeType: "image/jpeg")),
    /// ])
    /// ```
    case segments([Segment])

    /// The attachments in the state, in order.
    public var attachments: [Attachment] {
        guard case .segments(let segments) = self else { return [] }
        return segments.compactMap { segment in
            guard case .attachment(let attachment) = segment else { return nil }
            return attachment
        }
    }

    /// The state as a JSON value, for prompts and System One requests.
    ///
    /// A state of text segments only is the text of the segments, joined by newlines.
    /// A state with attachments is its tagged `Codable` form;
    /// a session does not pass such a state to a model that does not support attachments.
    var jsonValue: JSONValue {
        switch self {
        case .text(let text):
            return .string(text)
        case .json(let value):
            return value
        case .segments(let segments):
            if attachments.isEmpty {
                return .string(Segment.joinedText(segments))
            }
            return taggedJSONValue(segments)
        }
    }

    /// The tagged form of a segments state.
    private func taggedJSONValue(_ segments: [Segment]) -> JSONValue {
        .object([Self.segmentsKey: .array(segments.map(\.jsonValue))])
    }
}

// MARK: - Segments and attachments

extension DecisionState {
    /// A part of a ``DecisionState/segments(_:)`` state.
    public enum Segment: Hashable, Sendable {
        /// Text.
        case text(String)

        /// An attachment, such as an image.
        case attachment(Attachment)

        /// Returns the text of text segments, joined by newlines.
        static func joinedText(_ segments: [Segment]) -> String {
            segments.compactMap { segment in
                guard case .text(let text) = segment else { return nil }
                return text
            }
            .joined(separator: "\n")
        }

        fileprivate var jsonValue: JSONValue {
            switch self {
            case .text(let text):
                return .object(["kind": "text", "text": .string(text)])
            case .attachment(let attachment):
                return .object(["kind": "attachment", "attachment": attachment.jsonValue])
            }
        }
    }

    /// Media in a decision state, such as an image, a recording, a video, or a PDF document.
    ///
    /// An attachment holds its bytes or a URL, with a MIME type.
    /// Creating an attachment from a URL does not read the URL.
    public struct Attachment: Hashable, Sendable {
        /// The kind of media in an attachment.
        public enum Kind: String, Hashable, Sendable, Codable, CaseIterable {
            /// An image, such as a PNG or JPEG file.
            case image

            /// An audio recording, such as a WAV or MP3 file.
            case audio

            /// A video, such as an MP4 file.
            case video

            /// A document, such as a PDF file.
            case document
        }

        /// The origin of an attachment's content.
        public enum Source: Hashable, Sendable {
            /// The bytes of the media and their MIME type (for example, `image/jpeg`).
            case data(Data, mimeType: String)

            /// A URL that references the media, and its MIME type.
            case url(URL, mimeType: String)

            /// The MIME type of the media.
            public var mimeType: String {
                switch self {
                case .data(_, let mimeType), .url(_, let mimeType):
                    return mimeType
                }
            }
        }

        /// The kind of media.
        public let kind: Kind

        /// The origin of the content.
        public let source: Source

        /// The MIME type of the media.
        public var mimeType: String { source.mimeType }

        /// Creates an attachment.
        ///
        /// - Parameters:
        ///   - kind: The kind of media.
        ///   - source: The origin of the content.
        public init(kind: Kind, source: Source) {
            self.kind = kind
            self.source = source
        }

        /// Creates an image attachment from encoded image bytes.
        public init(imageData data: Data, mimeType: String) {
            self.init(kind: .image, source: .data(data, mimeType: mimeType))
        }

        /// Creates an image attachment from a URL.
        public init(imageURL url: URL, mimeType: String) {
            self.init(kind: .image, source: .url(url, mimeType: mimeType))
        }

        /// Creates an audio attachment from encoded audio bytes.
        public init(audioData data: Data, mimeType: String) {
            self.init(kind: .audio, source: .data(data, mimeType: mimeType))
        }

        /// Creates an audio attachment from a URL.
        public init(audioURL url: URL, mimeType: String) {
            self.init(kind: .audio, source: .url(url, mimeType: mimeType))
        }

        /// Creates a video attachment from encoded video bytes.
        public init(videoData data: Data, mimeType: String) {
            self.init(kind: .video, source: .data(data, mimeType: mimeType))
        }

        /// Creates a video attachment from a URL.
        public init(videoURL url: URL, mimeType: String) {
            self.init(kind: .video, source: .url(url, mimeType: mimeType))
        }

        /// Creates a document attachment from document bytes.
        ///
        /// The MIME type defaults to `application/pdf`.
        public init(documentData data: Data, mimeType: String = "application/pdf") {
            self.init(kind: .document, source: .data(data, mimeType: mimeType))
        }

        /// Creates a document attachment from a URL.
        ///
        /// The MIME type defaults to `application/pdf`.
        public init(documentURL url: URL, mimeType: String = "application/pdf") {
            self.init(kind: .document, source: .url(url, mimeType: mimeType))
        }

        fileprivate var jsonValue: JSONValue {
            let source: JSONValue
            switch self.source {
            case .data(let data, let mimeType):
                source = .object([
                    "kind": "data",
                    "data": .string(data.base64EncodedString()),
                    "mimeType": .string(mimeType),
                ])
            case .url(let url, let mimeType):
                source = .object([
                    "kind": "url",
                    "url": .string(url.absoluteString),
                    "mimeType": .string(mimeType),
                ])
            }
            return .object(["kind": .string(kind.rawValue), "source": source])
        }
    }
}

#if canImport(CoreGraphics) && canImport(ImageIO)
    extension DecisionState.Attachment {
        /// An error from converting a platform image to encoded data.
        public enum ImageEncodingError: Error, Hashable, Sendable {
            /// The image couldn't be converted to PNG data.
            case imageConversionFailed
        }

        /// Creates an image attachment by encoding an image as PNG data.
        ///
        /// - Parameter image: The image.
        /// - Throws: ``ImageEncodingError/imageConversionFailed`` if the image can't be encoded.
        public init(_ image: CGImage) throws {
            let bytes = NSMutableData()
            guard
                let destination = CGImageDestinationCreateWithData(bytes, "public.png" as CFString, 1, nil)
            else {
                throw ImageEncodingError.imageConversionFailed
            }
            CGImageDestinationAddImage(destination, image, nil)
            guard CGImageDestinationFinalize(destination) else {
                throw ImageEncodingError.imageConversionFailed
            }
            self.init(imageData: bytes as Data, mimeType: "image/png")
        }
    }
#endif

// MARK: - Codable

/// A text or JSON state encodes as a bare JSON value.
///
/// A ``DecisionState/segments(_:)`` state encodes as an object
/// whose only key is `$segments`, with an array of tagged segments:
///
/// ```json
/// {"$segments": [
///   {"kind": "text", "text": "A photo:"},
///   {"kind": "attachment", "attachment": {
///     "kind": "image",
///     "source": {"kind": "data", "data": "iVBORw0...", "mimeType": "image/png"}
///   }}
/// ]}
/// ```
///
/// A JSON state whose value is an object with the only key `$segments` or `$json`
/// encodes inside an object with the only key `$json`,
/// so that it can't be confused with a segments state.
/// Every other JSON state encodes as before.
extension DecisionState: Codable {
    fileprivate static let segmentsKey = "$segments"
    fileprivate static let escapeKey = "$json"

    public init(from decoder: any Decoder) throws {
        let value = try JSONValue(from: decoder)
        switch value {
        case .string(let text):
            self = .text(text)
        case .object(let object) where object.count == 1 && object[Self.escapeKey] != nil:
            self = .json(object[Self.escapeKey]!)
        case .object(let object) where object.count == 1 && object[Self.segmentsKey] != nil:
            let container = try decoder.container(keyedBy: TaggedKeys.self)
            self = .segments(try container.decode([Segment].self, forKey: .segments))
        default:
            self = .json(value)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        switch self {
        case .text(let text):
            try JSONValue.string(text).encode(to: encoder)
        case .segments(let segments):
            try taggedJSONValue(segments).encode(to: encoder)
        case .json(let value):
            if case .object(let object) = value, object.count == 1,
                object[Self.segmentsKey] != nil || object[Self.escapeKey] != nil
            {
                try JSONValue.object([Self.escapeKey: value]).encode(to: encoder)
            } else {
                try value.encode(to: encoder)
            }
        }
    }

    private enum TaggedKeys: String, CodingKey {
        case segments = "$segments"
    }
}

extension DecisionState.Segment: Codable {
    private enum CodingKeys: String, CodingKey {
        case kind, text, attachment
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try container.decode(String.self, forKey: .kind)
        switch kind {
        case "text":
            self = .text(try container.decode(String.self, forKey: .text))
        case "attachment":
            self = .attachment(try container.decode(DecisionState.Attachment.self, forKey: .attachment))
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .kind,
                in: container,
                debugDescription: "Unknown segment kind: \(kind)"
            )
        }
    }

    public func encode(to encoder: any Encoder) throws {
        try jsonValue.encode(to: encoder)
    }
}

extension DecisionState.Attachment: Codable {
    private enum CodingKeys: String, CodingKey {
        case kind, source
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            kind: try container.decode(Kind.self, forKey: .kind),
            source: try container.decode(Source.self, forKey: .source)
        )
    }

    public func encode(to encoder: any Encoder) throws {
        try jsonValue.encode(to: encoder)
    }
}

extension DecisionState.Attachment.Source: Codable {
    private enum CodingKeys: String, CodingKey {
        case kind, data, url, mimeType
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try container.decode(String.self, forKey: .kind)
        let mimeType = try container.decode(String.self, forKey: .mimeType)
        switch kind {
        case "data":
            let encoded = try container.decode(String.self, forKey: .data)
            guard let data = Data(base64Encoded: encoded) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .data,
                    in: container,
                    debugDescription: "The attachment data is not valid base64."
                )
            }
            self = .data(data, mimeType: mimeType)
        case "url":
            self = .url(try container.decode(URL.self, forKey: .url), mimeType: mimeType)
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .kind,
                in: container,
                debugDescription: "Unknown attachment source kind: \(kind)"
            )
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .data(let data, let mimeType):
            try container.encode("data", forKey: .kind)
            try container.encode(data.base64EncodedString(), forKey: .data)
            try container.encode(mimeType, forKey: .mimeType)
        case .url(let url, let mimeType):
            try container.encode("url", forKey: .kind)
            try container.encode(url, forKey: .url)
            try container.encode(mimeType, forKey: .mimeType)
        }
    }
}

// MARK: - ExpressibleByStringLiteral

extension DecisionState: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) {
        self = .text(value)
    }
}
