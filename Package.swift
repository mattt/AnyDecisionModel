// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "AnyDecisionModel",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "AnyDecisionModel",
            targets: ["AnyDecisionModel"]
        )
    ],
    traits: [
        .trait(name: "MLX"),
        .default(enabledTraits: []),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.4"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "AnyDecisionModel",
            dependencies: [
                .product(
                    name: "MLXLLM",
                    package: "mlx-swift-lm",
                    condition: .when(traits: ["MLX"])
                ),
                .product(
                    name: "MLXLMCommon",
                    package: "mlx-swift-lm",
                    condition: .when(traits: ["MLX"])
                ),
                .product(
                    name: "MLXHuggingFace",
                    package: "mlx-swift-lm",
                    condition: .when(traits: ["MLX"])
                ),
                .product(
                    name: "HuggingFace",
                    package: "swift-huggingface",
                    condition: .when(traits: ["MLX"])
                ),
                .product(
                    name: "Tokenizers",
                    package: "swift-transformers",
                    condition: .when(traits: ["MLX"])
                ),
            ]
        ),
        .testTarget(
            name: "AnyDecisionModelTests",
            dependencies: ["AnyDecisionModel"]
        ),
    ]
)
