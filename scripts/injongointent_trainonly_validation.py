#!/usr/bin/env python3
"""Freeze and reload an InjongoIntent validation split from training files only."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sallm.data.loaders.injongointent_split import split_injongointent_rows

DATASET_NAME = "masakhane/InjongoIntent"
DATASET_REVISION = "fe4be3882a1614161dfe231ec793197bb74f4b44"
MANIFEST_SCHEMA = "sallm.injongointent_train_only_validation/v1"
VALIDATION_RATIO = 0.1


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in row.items()
        if not str(key).startswith("__validation_")
    }


def canonical_row_bytes(row: dict[str, Any]) -> bytes:
    return json.dumps(
        canonical_row(row),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def row_sha256(row: dict[str, Any]) -> str:
    return sha256_bytes(canonical_row_bytes(row))


def rows_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_row_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_train_jsonl(content: bytes, language: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{language} train line {line_number} is not an object")
        if "lang" in row and str(row["lang"]) != language:
            raise ValueError(
                f"{language} train line {line_number} has another language"
            )
        rows.append({**row, "lang": language})
    if not rows:
        raise ValueError(f"Pinned {language} train file is empty")
    return rows


def _normalized_text(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def deduplicate_train_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Keep the first within-train occurrence of each normalized label-text pair."""
    intents_by_text: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()
    unique_rows: list[dict[str, Any]] = []
    duplicate_indices: list[int] = []
    for source_index, source_row in enumerate(rows):
        row = canonical_row(source_row)
        intent = str(row.get("intent", "")).strip()
        text = _normalized_text(row.get("text"))
        if not intent or not text:
            raise ValueError(f"Missing intent/text at source index {source_index}")
        intents_by_text[text].add(intent)
        key = (intent, text)
        if key in seen:
            duplicate_indices.append(source_index)
            continue
        seen.add(key)
        row["__validation_source_index"] = source_index
        unique_rows.append(row)
    contradictions = {
        text: sorted(intents)
        for text, intents in intents_by_text.items()
        if len(intents) > 1
    }
    if contradictions:
        preview = list(contradictions.items())[:5]
        raise ValueError(
            f"Contradictory within-train labels for normalized text: {preview}"
        )
    return unique_rows, duplicate_indices


def derive_language_validation(
    language: str,
    content: bytes,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = parse_train_jsonl(content, language)
    unique_rows, duplicate_indices = deduplicate_train_rows(raw_rows)
    train_rows, validation_rows = split_injongointent_rows(
        unique_rows,
        validation_ratio=VALIDATION_RATIO,
    )
    if not validation_rows:
        raise ValueError(f"Derived {language} validation split is empty")
    records = []
    selected = []
    for row in validation_rows:
        source_index = int(row["__validation_source_index"])
        digest = row_sha256(row)
        clean = canonical_row(row)
        clean["__validation_source_index"] = source_index
        clean["__validation_row_sha256"] = digest
        selected.append(clean)
        records.append({"source_index": source_index, "row_sha256": digest})
    evidence = {
        "source_file_sha256": sha256_bytes(content),
        "raw_row_count": len(raw_rows),
        "raw_canonical_sha256": rows_sha256(raw_rows),
        "deduplicated_row_count": len(unique_rows),
        "deduplicated_canonical_sha256": rows_sha256(unique_rows),
        "within_train_duplicate_count": len(duplicate_indices),
        "within_train_duplicate_source_indices": duplicate_indices,
        "derived_train_row_count": len(train_rows),
        "validation_row_count": len(validation_rows),
        "validation_source_indices": [row["source_index"] for row in records],
        "validation_content_sha256": rows_sha256(validation_rows),
        "validation_records": records,
    }
    return selected, evidence


def load_frozen_validation_rows(
    manifest_path: Path,
    languages: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Unexpected InjongoIntent validation manifest schema")
    if manifest.get("dataset") != DATASET_NAME:
        raise ValueError("Unexpected InjongoIntent dataset binding")
    if manifest.get("revision") != DATASET_REVISION:
        raise ValueError("Unexpected InjongoIntent revision binding")
    if (
        manifest.get("source_split") != "train"
        or manifest.get("held_out_data_accessed") is not False
    ):
        raise ValueError("Validation manifest is not train-only")
    if manifest.get("architecture_blind") is not True:
        raise ValueError("Validation split is not declared architecture-blind")
    if set(manifest.get("languages", {})) != set(languages):
        raise ValueError("Validation manifest languages do not match evaluation")

    selected_rows: list[dict[str, Any]] = []
    verified_languages: dict[str, Any] = {}
    for language in languages:
        frozen = manifest["languages"][language]
        relative = frozen.get("source_path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError(f"Invalid relative train source path for {language}")
        source = (manifest_path.parent / relative).resolve()
        if manifest_path.parent.resolve() not in source.parents:
            raise ValueError(f"Train source escapes manifest directory: {source}")
        content = source.read_bytes()
        rows, observed = derive_language_validation(language, content)
        expected = {key: value for key, value in frozen.items() if key != "source_path"}
        if observed != expected:
            raise ValueError(
                f"Frozen train-only validation evidence changed for {language}"
            )
        selected_rows.extend(rows)
        verified_languages[language] = {
            "source_path": relative,
            "source_file_sha256": observed["source_file_sha256"],
            "raw_row_count": observed["raw_row_count"],
            "deduplicated_row_count": observed["deduplicated_row_count"],
            "within_train_duplicate_count": observed["within_train_duplicate_count"],
            "validation_row_count": observed["validation_row_count"],
            "validation_content_sha256": observed["validation_content_sha256"],
        }
    return selected_rows, {
        "manifest_sha256": sha256_file(manifest_path),
        "dataset": DATASET_NAME,
        "revision": DATASET_REVISION,
        "source_split": "train",
        "held_out_data_accessed": False,
        "architecture_blind": True,
        "languages": verified_languages,
    }
