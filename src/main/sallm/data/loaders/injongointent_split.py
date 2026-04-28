from __future__ import annotations

from collections import defaultdict
from hashlib import md5
from typing import Any

INJONGOINTENT_VALIDATION_RATIO = 0.1


def split_injongointent_rows(
    rows: list[dict[str, Any]],
    validation_ratio: float = INJONGOINTENT_VALIDATION_RATIO,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped_keys: dict[str, list[str]] = defaultdict(list)
    row_keys: list[str] = []

    for index, row in enumerate(rows):
        key = _row_key(row, index)
        row_keys.append(key)
        grouped_keys[str(row["intent"])].append(key)

    validation_keys: set[str] = set()
    for keys in grouped_keys.values():
        ordered_keys = sorted(keys, key=_stable_hash)
        group_size = len(ordered_keys)
        if group_size <= 1:
            continue

        validation_count = min(
            max(int(round(group_size * validation_ratio)), 1),
            group_size - 1,
        )
        validation_keys.update(ordered_keys[:validation_count])

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for key, row in zip(row_keys, rows, strict=True):
        target = validation_rows if key in validation_keys else train_rows
        target.append(dict(row))

    return train_rows, validation_rows


def _row_key(row: dict[str, Any], fallback_index: int) -> str:
    for field in ("example_id", "raw", "text"):
        value = row.get(field)
        if value:
            return str(value)
    return str(fallback_index)


def _stable_hash(value: str) -> str:
    return md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()
