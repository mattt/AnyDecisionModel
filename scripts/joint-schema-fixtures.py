# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub>=0.30",
#     "mlx==0.32.3",
#     "mlx-lm==0.32.0",
#     "numpy>=2",
# ]
# ///
"""Writes the joint schema head fixtures for the AnyDecisionModel tests.

The fixtures come from `clef_mlx.py`, the reference MLX implementation of Cloudflare's Clef,
at a pinned revision of mlx-community/clef-flash-4bit.

    uv run scripts/joint-schema-fixtures.py encoding   # prompt tokens and spans (no model download)
    uv run scripts/joint-schema-fixtures.py head       # a small random head (no model download)
    uv run scripts/joint-schema-fixtures.py model      # Clef-flash 4-bit answers (downloads 6.2 GB)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPOSITORY = "mlx-community/clef-flash-4bit"
REVISION = "3fb6bec9d8017565c767a9f3796d49f7d2bdba6c"
QUESTION_ID = "question"
FIXTURES = Path(__file__).resolve().parent.parent / "Tests/AnyDecisionModelTests/Fixtures/JointSchema"

TICKET = "I was charged twice this month and nobody has answered my emails in a week. Fix this NOW."
DEPARTMENTS = [
    {"name": "technical", "description": "bugs, outages, errors, login problems"},
    {"name": "billing", "description": "payments, charges, refunds, invoices"},
    {"name": "sales", "description": "pricing questions, upgrades, new purchases"},
]
INVOICE = {
    "invoice": {
        "vendor": "Acme Café",
        "total": 1250.5,
        "currency": "USD",
        "status": "overdue",
        "lines": [{"sku": "A-1", "quantity": 2}, {"sku": "B/2", "quantity": 1}],
        "notes": None,
        "paid": False,
    }
}


def load_reference(full: bool = False) -> Any:
    """Imports clef_mlx.py from the pinned revision and returns the module and the snapshot path."""
    from huggingface_hub import hf_hub_download, snapshot_download

    if full:
        path = Path(snapshot_download(REPOSITORY, revision=REVISION, ignore_patterns=["__pycache__/*"]))
    else:
        path = Path(hf_hub_download(REPOSITORY, "clef_mlx.py", revision=REVISION)).parent
    sys.path.insert(0, str(path))
    import clef_mlx

    return clef_mlx, path


def clef_question(question: dict[str, Any]) -> dict[str, Any]:
    """Converts a question in AnyDecisionModel terms to a Clef question."""
    kind = question["type"]
    if kind == "binary":
        result: dict[str, Any] = {"type": "noul", "instructions": question["instructions"]}
        if criteria := question.get("criteria"):
            result["criteria"] = {"true": criteria["whenTrue"], "false": criteria["whenFalse"]}
        return result
    if kind == "choice":
        return {
            "type": "choice",
            "instructions": question["instructions"],
            "criteria": {option["name"]: option.get("description") for option in question["options"]},
        }
    return {"type": "score", "instructions": question["instructions"], "criteria": question["levels"]}


def clef_state(state: dict[str, Any]) -> Any:
    return state["text"] if "text" in state else state["json"]


def answer_names(question: dict[str, Any]) -> list[str]:
    """The answer names of a question, in AnyDecisionModel answer order."""
    if question["type"] == "binary":
        return ["true", "false"]
    if question["type"] == "choice":
        return [option["name"] for option in question["options"]]
    return [str(index) for index in range(len(question["levels"]))]


def write(name: str, value: Any) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / name
    text = json.dumps(value, ensure_ascii=False, indent=1)
    # Keep arrays of numbers on one line.
    text = re.sub(
        r"\[\s+(-?[\d.e+-]+(?:,\s+-?[\d.e+-]+)*)\s+\]",
        lambda match: "[" + ", ".join(item.strip() for item in match.group(1).split(",")) + "]",
        text,
    )
    path.write_text(text + "\n")
    print(f"wrote {path.relative_to(FIXTURES.parents[2])}")


# --------------------------------------------------------------------------- encoding


class ScalarTokenizer:
    """A tokenizer that maps each Unicode code point to its value."""

    class Encoding:
        def __init__(self, ids: list[int]):
            self.input_ids = ids

    def __call__(self, text: str, add_special_tokens: bool = False) -> ScalarTokenizer.Encoding:
        return ScalarTokenizer.Encoding([ord(character) for character in text])


ENCODING_CASES = [
    {
        "name": "binary about text",
        "state": {"text": "A 6 sided die rolled a 3."},
        "question": {"type": "binary", "instructions": "Is the number rolled odd?"},
    },
    {
        "name": "binary with criteria",
        "state": {"text": TICKET},
        "question": {
            "type": "binary",
            "instructions": "Is the customer asking for a refund?",
            "criteria": {"whenTrue": "The customer wants money back.", "whenFalse": "No refund is requested."},
        },
    },
    {
        "name": "choice with unsorted options",
        "state": {"text": TICKET},
        "question": {
            "type": "choice",
            "instructions": "Route this support ticket.",
            "options": DEPARTMENTS
            + [{"name": "Émigré"}, {"name": "Zebra"}, {"name": "émigré", "description": "lowercase \"quoted\""}],
        },
    },
    {
        "name": "score",
        "state": {"text": TICKET},
        "question": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "levels": ["Calm", "Frustrated", "Very angry"],
        },
    },
    {
        "name": "binary about JSON",
        "state": {"json": INVOICE},
        "question": {"type": "binary", "instructions": "Is the total above 1000 USD?"},
    },
    {
        "name": "empty text and one option",
        "state": {"text": ""},
        "question": {"type": "choice", "instructions": "Pick one.", "options": [{"name": "only"}]},
    },
]

RENDER_CASES = [
    "plain text is not quoted",
    {"b": 1, "a": [True, None, False], "É": 2, "Z": 3, "😀": "emoji key", "é": "after É"},
    [0, -7, 1000000000000000, 9007199254740991, -9007199254740991],
    [-0.0, 0.1, -2.5, 1250.5, 123.456, 1e-05, 0.0001, 1e-07, 1e16, 1e21, 1.5e300, 5e-324, 1.2345678901234568e16],
    ["x/y", 'quote " backslash \\', "newline \n tab \t return \r", "\b\f", "\x01\x1f\x7f", "  ", "é😀"],
    {"nested": {"z": [], "a": {}}, "empty": ""},
    INVOICE,
]


def encoding(_: argparse.Namespace) -> None:
    clef_mlx, _ = load_reference()
    tokenizer = ScalarTokenizer()
    cases = []
    for case in ENCODING_CASES:
        question = case["question"]
        record = {"state": clef_state(case["state"]), "questions": {QUESTION_ID: clef_question(question)}}
        encoded = clef_mlx.encode_record(tokenizer, record, truncate=False)
        [field] = encoded.questions
        prefix = (
            tokenizer(f"<|im_start|>system\n{clef_mlx.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n").input_ids
            + tokenizer(clef_mlx.render(record["state"])).input_ids
            + tokenizer("\n\nSCHEMA FIELDS:\n").input_ids
        )
        assert list(encoded.input_ids[: len(prefix)]) == prefix
        names = answer_names(question)
        cases.append(
            {
                **case,
                "input_ids": list(encoded.input_ids),
                "prefix_length": len(prefix),
                "type_id": field.question_type,
                "question_span": list(field.question_span),
                "option_spans": [list(span) for span in field.option_spans],
                "answer_indices": [names.index(option_id) for option_id in field.option_ids],
            }
        )
    renders = [{"value": value, "text": clef_mlx.render(value)} for value in RENDER_CASES]
    write("encoding.json", {"revision": REVISION, "cases": cases, "renders": renders})


# --------------------------------------------------------------------------- head

HEAD_CONFIG = {"hidden_size": 32, "width": 16, "routing_layers": 2, "layers": 2, "heads": 4, "feedforward": 32}
HEAD_FIELDS = [
    {"type_id": 0, "question_span": [2, 5], "option_spans": [[6, 9], [10, 12]]},
    {"type_id": 1, "question_span": [13, 16], "option_spans": [[17, 19], [20, 23], [24, 26]]},
    {"type_id": 2, "question_span": [27, 29], "option_spans": [[30, 32], [33, 35], [36, 38], [38, 39]]},
]


def original_key(key: str) -> str:
    """Returns the PyTorch name of a head parameter, the inverse of `JointSchemaHead.sanitize`."""
    key = key.replace(".feedforward.fc1.", ".feedforward.0.").replace(".feedforward.fc2.", ".feedforward.3.")
    return key.replace("scorer1.", "residual_scorer.0.").replace("scorer2.", "residual_scorer.3.")


def head(_: argparse.Namespace) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    clef_mlx, _ = load_reference()
    mx.random.seed(0)
    model = clef_mlx.JointSchemaHead(**HEAD_CONFIG)
    weights = {
        key: mx.random.normal(value.shape, scale=0.5 if value.ndim == 0 else 0.2)
        for key, value in tree_flatten(model.parameters())
    }
    model.load_weights(list(weights.items()), strict=True)

    length, vocabulary = 40, 64
    hidden = mx.random.normal((length, HEAD_CONFIG["hidden_size"]))
    input_ids = mx.random.randint(0, vocabulary, (length,))
    lexical = mx.random.normal((vocabulary, HEAD_CONFIG["hidden_size"]))
    record = clef_mlx.EncodedRecord(
        input_ids=tuple(input_ids.tolist()),
        questions=tuple(
            clef_mlx.EncodedQuestion(
                question_id=str(index),
                question_type=field["type_id"],
                question_span=tuple(field["question_span"]),
                option_spans=tuple(tuple(span) for span in field["option_spans"]),
                option_ids=tuple(str(option) for option in range(len(field["option_spans"]))),
            )
            for index, field in enumerate(HEAD_FIELDS)
        ),
    )
    logits = model(hidden, input_ids, record, lambda ids: lexical[ids])

    arrays = {f"head.{original_key(key)}": value for key, value in weights.items()}
    arrays |= {"hidden": hidden, "input_ids": input_ids, "lexical": lexical}
    arrays |= {f"logits.{index}": value for index, value in enumerate(logits)}
    FIXTURES.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(FIXTURES / "head.safetensors"), arrays)
    print(f"wrote {(FIXTURES / 'head.safetensors').relative_to(FIXTURES.parents[2])}")
    write("head.json", {"revision": REVISION, "config": HEAD_CONFIG, "fields": HEAD_FIELDS})


# --------------------------------------------------------------------------- model

MODEL_CASES = [
    {
        "name": "odd roll",
        "state": {"text": "A 6 sided die rolled a 3."},
        "question": {"type": "binary", "instructions": "Is the number rolled odd?"},
    },
    {
        "name": "even roll",
        "state": {"text": "A 6 sided die rolled a 4."},
        "question": {"type": "binary", "instructions": "Is the number rolled odd?"},
    },
    {
        "name": "routing",
        "state": {"text": TICKET},
        "question": {"type": "choice", "instructions": "Route this support ticket.", "options": DEPARTMENTS},
    },
    {
        "name": "frustration",
        "state": {"text": TICKET},
        "question": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "levels": ["Calm", "Frustrated", "Very angry"],
        },
    },
    {
        "name": "refund",
        "state": {"text": TICKET},
        "question": {
            "type": "binary",
            "instructions": "Is the customer asking for a refund?",
            "criteria": {"whenTrue": "The customer wants money back.", "whenFalse": "No refund is requested."},
        },
    },
    {
        "name": "invoice total",
        "state": {"json": INVOICE},
        "question": {"type": "binary", "instructions": "Is the total above 1000 USD?"},
    },
    {
        "name": "invoice status",
        "state": {"json": INVOICE},
        "question": {
            "type": "choice",
            "instructions": "What is the invoice status?",
            "options": [
                {"name": "paid", "description": "The invoice is paid."},
                {"name": "overdue", "description": "The invoice is past due."},
                {"name": "draft", "description": "The invoice was not sent."},
            ],
        },
    },
]

MODEL_CASES += [
    {
        "name": "special characters in JSON",
        "state": {
            "json": {
                "título": "Café “quoted” — dash",
                "emoji": "🚀 launch 👍🏽",
                "control": "tab\tnewline\nbell\u0007",
                "separator": "line\u2028paragraph\u2029",
                "日本語": "テキスト",
            }
        },
        "question": {"type": "binary", "instructions": "Does the state mention a rocket launch?"},
    },
    {
        "name": "special characters in text",
        "state": {"text": "Show HN: I built a “tiny” compiler — it’s fast… 🚀\nComments: naïve façade, ½ price, ™"},
        "question": {
            "type": "choice",
            "instructions": "What kind of post is this?",
            "options": [
                {"name": "Show HN", "description": "Someone shows something they made."},
                {"name": "Ask HN", "description": "Someone asks the community a question."},
                {"name": "Job", "description": "A company is hiring."},
            ],
        },
    },
]


def model(_: argparse.Namespace) -> None:
    import importlib.metadata

    clef_mlx, path = load_reference(full=True)
    reference = clef_mlx.load(path, backend="lm")
    cases = []
    for case in MODEL_CASES:
        question = case["question"]
        record = {"state": clef_state(case["state"]), "questions": {QUESTION_ID: clef_question(question)}}
        encoded = clef_mlx.encode_record(reference.tokenizer, record, truncate=False)
        probabilities = reference.predict(record, truncate=False)[QUESTION_ID]
        cases.append(
            {
                **case,
                "input_ids": list(encoded.input_ids),
                "probabilities": [probabilities[name] for name in answer_names(question)],
            }
        )
    versions = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm")}
    write("clef-flash-4bit.json", {"repository": REPOSITORY, "revision": REVISION, "versions": versions, "cases": cases})


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("encoding", help="prompt tokens and spans").set_defaults(run=encoding)
    commands.add_parser("head", help="a small random head").set_defaults(run=head)
    commands.add_parser("model", help="Clef-flash 4-bit answers").set_defaults(run=model)
    arguments = parser.parse_args()
    arguments.run(arguments)


if __name__ == "__main__":
    main()
