from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from datasets import Dataset


def extract_instruction_pair(example: dict[str, Any]) -> tuple[str, str]:
    if "instruction" in example and "output" in example:
        return str(example["instruction"]), str(example["output"])
    if "inputs" in example and "targets" in example:
        return str(example["inputs"]), str(example["targets"])
    if "source" in example and "target" in example:
        return str(example["source"]), str(example["target"])
    if "data" in example and "text" in example:
        return str(example["data"]), str(example["text"])
    if "triple" in example and "verbalisation" in example:
        return str(example["triple"]), str(example["verbalisation"])
    if "prompt" in example and "response" in example:
        return str(example["prompt"]), str(example["response"])
    if "query" in example and "response" in example:
        return str(example["query"]), str(example["response"])
    if "text" in example and "title" in example:
        return str(example["text"]), str(example["title"])
    if "article" in example and "headline" in example:
        return str(example["article"]), str(example["headline"])
    candidates_user = [
        "instruction",
        "inputs",
        "prompt",
        "query",
        "input",
        "source",
        "data",
        "triple",
    ]
    candidates_assistant = [
        "output",
        "targets",
        "response",
        "target",
        "answer",
        "text",
        "verbalisation",
    ]
    user_val = next(
        (example[k] for k in candidates_user if k in example),
        None,
    )
    assistant_val = next(
        (example[k] for k in candidates_assistant if k in example),
        None,
    )
    if user_val is None or assistant_val is None:
        raise KeyError("Unable to locate instruction and response fields in example.")
    return str(user_val), str(assistant_val)


def reconstruct_entities_from_iob(
    tokens: list[str], tag_ids: Iterable[int], tag_map: list[str]
) -> list[str]:
    entities: list[str] = []
    current_tokens: list[str] = []
    current_label: str | None = None
    for token, tag_id in zip(tokens, tag_ids, strict=False):
        tag_name = tag_map[tag_id]
        if tag_name.startswith("B-"):
            if current_tokens:
                entities.append(f"{current_label}: {' '.join(current_tokens)}")
            current_tokens = [token]
            current_label = tag_name[2:]
        elif tag_name.startswith("I-"):
            if current_label == tag_name[2:]:
                current_tokens.append(token)
            else:
                if current_tokens:
                    entities.append(f"{current_label}: {' '.join(current_tokens)}")
                current_tokens = []
                current_label = None
        else:
            if current_tokens:
                entities.append(f"{current_label}: {' '.join(current_tokens)}")
            current_tokens = []
            current_label = None
    if current_tokens:
        entities.append(f"{current_label}: {' '.join(current_tokens)}")
    return entities


def resolve_language_column(dataset: Dataset) -> list[Any] | None:
    if "lang" in dataset.column_names:
        return list(dataset["lang"])
    if "language_code" in dataset.column_names:
        return list(dataset["language_code"])
    if "language" in dataset.column_names:
        return list(dataset["language"])
    return None
