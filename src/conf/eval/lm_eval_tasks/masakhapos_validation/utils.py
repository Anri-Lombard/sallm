from __future__ import annotations

from datasets import Dataset


def process_pos_docs(split: Dataset) -> Dataset:
    docs: list[dict[str, object]] = []
    tokens: list[str] = []
    upos: list[str] = []
    row_id = 0

    for raw in split["text"]:
        line = str(raw).strip()

        if not line:
            if tokens:
                docs.append({"id": str(row_id), "tokens": tokens, "upos": upos})
                row_id += 1
                tokens = []
                upos = []
            continue

        if line.startswith("-DOCSTART-"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        tokens.append(parts[0])
        upos.append(parts[-1])

    if tokens:
        docs.append({"id": str(row_id), "tokens": tokens, "upos": upos})

    return Dataset.from_list(docs)


def doc_to_target(doc: dict[str, object]) -> list[str]:
    return list(doc["upos"])


def token_accuracy(references, predictions, **kwargs) -> float:
    def unwrap_singleton_lists(value):
        while (
            isinstance(value, list) and len(value) == 1 and isinstance(value[0], list)
        ):
            value = value[0]
        return value

    if not references or not predictions:
        return 0.0

    ref = unwrap_singleton_lists(references)
    pred = unwrap_singleton_lists(predictions)

    if not isinstance(ref, list):
        return 0.0
    if not isinstance(pred, list):
        return 0.0
    if not ref:
        return 0.0

    max_len = len(ref)
    aligned = min(len(ref), len(pred))
    correct = sum(1 for i in range(aligned) if str(ref[i]) == str(pred[i]))
    return correct / max_len
