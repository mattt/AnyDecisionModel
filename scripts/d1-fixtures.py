# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub>=0.30",
#     "mlx==0.32.4",
#     "mlx-vlm==0.8.0",
#     "numpy>=2",
# ]
# ///
"""Writes the d1 fixtures for the AnyDecisionModel tests.

The answers come from mlx-vlm's d1 port, which follows the model repository's `prompt.py`,
at a pinned revision of LiquidAI/d1-3B-MLX-8bit.

    uv run scripts/d1-fixtures.py   # downloads d1-3B 8-bit (3.72 GB)

Each case has one question, which mlx-vlm evaluates in one pass.
The fixture records the rendered prompt, its token IDs, the token IDs read for each answer,
and two sets of probabilities:
`probabilities` with mlx-vlm's model as it loads (float16 activations),
and `float32_probabilities` with every float16 and bfloat16 parameter converted to float32,
as `MLXDecisionModel` does when it loads a model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY = "LiquidAI/d1-3B-MLX-8bit"
REVISION = "7f54fff5721e610db14e8d0e0d7f4579f345ed46"
FIXTURE = Path(__file__).resolve().parent.parent / "Tests/AnyDecisionModelTests/Fixtures/D1/d1-3b-mlx-8bit.json"

TICKET = (
    "I was charged twice for my order, "
    "and now the app logs me out when I try to ask for a refund."
)

# JSON states use sorted keys, because `JSONValue` objects have no key order
# and `MLXDecisionModel` renders them with sorted keys.
ORDER = {
    "customer": {"name": "Zoë Martín", "plan": "pro", "tenure_months": 27},
    "items": [
        {"quantity": 2, "sku": "KB-104", "unit_price": 89.5},
        {"quantity": 1, "sku": "MS-220", "unit_price": 34},
    ],
    "message": "Where is my package? It was due Monday.\nTracking says \"label created\".",
    "paid": True,
    "refund_requested": False,
    "shipping": {"carrier": "UPS", "delivered_at": None, "status": "label_created"},
}

CASES: list[dict[str, Any]] = [
    {
        "name": "ticket urgency",
        "state": {"text": TICKET},
        "question": {"type": "binary", "instructions": "Does this convey urgency?"},
    },
    {
        "name": "ticket team",
        "state": {"text": TICKET},
        "question": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "options": [
                {"name": "billing", "description": "payments, charges, refunds"},
                {"name": "technical", "description": "bugs, outages, login problems"},
                {"name": "sales"},
            ],
        },
    },
    {
        "name": "ticket frustration",
        "state": {"text": TICKET},
        "question": {
            "type": "score",
            "instructions": "How frustrated?",
            "levels": ["Calm", "Frustrated", "Very angry"],
        },
    },
    {
        "name": "order delayed",
        "state": {"json": ORDER},
        "question": {
            "type": "binary",
            "instructions": "Has the shipment been delayed?",
            "criteria": {
                "whenTrue": "the package is late or has not shipped by its due date",
                "whenFalse": "the package is on schedule",
            },
        },
    },
    {
        "name": "order intent",
        "state": {"json": ORDER},
        "question": {
            "type": "choice",
            "instructions": "What does the customer want?",
            "options": [
                {"name": "track_order", "description": "find out where an order is"},
                {"name": "cancel_order"},
                {"name": "return_item", "description": "send an item back"},
                {"name": "change_address", "description": "ship to a different address"},
                {"name": "billing_question"},
                {"name": "other", "description": "anything else"},
            ],
        },
    },
    {
        "name": "review sentiment",
        "state": {
            "text": (
                "The keyboard feels great and the battery lasts for weeks, "
                "but the Bluetooth drops every few hours and support never replied."
            )
        },
        "question": {
            "type": "score",
            "instructions": "What is the overall sentiment of this review?",
            "levels": ["Very negative", "Negative", "Mixed", "Positive", "Very positive"],
        },
    },
    {
        "name": "review stars",
        "state": {"text": "Five stars. Arrived early, works perfectly, and the price was fair."},
        "question": {
            "type": "score",
            "instructions": "How many stars would this reviewer give?",
            "levels": ["0 stars", "1 star", "2 stars", "3 stars", "4 stars", "5 stars", "6 stars"],
        },
    },
    {
        "name": "exam letters",
        "state": {"text": "Water boils at 100 degrees Celsius at sea level."},
        "question": {
            "type": "choice",
            "instructions": "At what temperature does water boil at sea level?",
            "options": [
                {"name": "a", "description": "0 °C"},
                {"name": "b", "description": "50 °C"},
                {"name": "c", "description": "100 °C"},
                {"name": "d", "description": "212 °C"},
            ],
        },
    },
    {
        "name": "json string",
        "state": {"json": "Can I upgrade to the annual plan halfway through the month?"},
        "question": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "options": [
                {"name": "billing"},
                {"name": "technical"},
                {"name": "sales"},
            ],
        },
    },
]


def d1_question(question: dict[str, Any]) -> dict[str, Any]:
    """The question in the Decision Index schema that mlx-vlm reads."""
    kind = question["type"]
    if kind == "binary":
        out: dict[str, Any] = {"type": "noul", "instructions": question["instructions"]}
        if criteria := question.get("criteria"):
            out["criteria"] = {"true": criteria["whenTrue"], "false": criteria["whenFalse"]}
        return out
    if kind == "choice":
        return {
            "type": "choice",
            "instructions": question["instructions"],
            "criteria": {o["name"]: o.get("description") for o in question["options"]},
        }
    return {"type": "score", "instructions": question["instructions"], "criteria": question["levels"]}


def d1_state(state: dict[str, Any]) -> Any:
    return state["text"] if "text" in state else state["json"]


def convert_to_float32(model: Any) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    parameters = [
        (key, value.astype(mx.float32))
        for key, value in tree_flatten(model.parameters())
        if value.dtype in (mx.float16, mx.bfloat16)
    ]
    model.update(tree_unflatten(parameters))
    mx.eval(model.parameters())


def main() -> None:
    from importlib.metadata import version

    from huggingface_hub import snapshot_download
    from mlx_vlm import load
    from mlx_vlm.models.d1.prompt import as_question, prefix_text, readout_ids, suffix_text

    path = snapshot_download(REPOSITORY, revision=REVISION)
    model, processor = load(path)
    tokenizer = getattr(processor, "tokenizer", processor)
    bos = tokenizer.bos_token if isinstance(tokenizer.bos_token, str) else ""

    cases = []
    for case in CASES:
        state = d1_state(case["state"])
        question = as_question(d1_question(case["question"]))
        prompt = prefix_text(state, bos) + suffix_text(tokenizer, question)
        input_ids, branches, _ = model.decision_inputs(processor, state, [question])
        assert not branches and input_ids == tokenizer.encode(prompt, add_special_tokens=False)
        probabilities, _ = model.probabilities(processor, state, [question])
        cases.append(
            {
                **case,
                "prompt": prompt,
                "input_ids": input_ids,
                "readout_ids": readout_ids(tokenizer, question),
                "probabilities": probabilities[0],
            }
        )

    convert_to_float32(model)
    for case in cases:
        state = d1_state(case["state"])
        question = as_question(d1_question(case["question"]))
        probabilities, _ = model.probabilities(processor, state, [question])
        case["float32_probabilities"] = probabilities[0]

    fixture = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "versions": {"mlx": version("mlx"), "mlx-vlm": version("mlx-vlm")},
        "cases": cases,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for case in cases:
        print(case["name"], [round(p, 4) for p in case["probabilities"]], file=sys.stderr)


if __name__ == "__main__":
    main()
