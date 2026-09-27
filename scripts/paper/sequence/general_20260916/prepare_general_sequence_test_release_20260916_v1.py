#!/usr/bin/env python3
"""Verify the frozen selector output and seal one-time General test releases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_manifest(directory: Path) -> None:
    manifest = directory / "ARTIFACTS.sha256"
    entries = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name in entries or "/" in name or name.startswith("."):
            raise ValueError(f"Invalid manifest entry: {name}")
        entries[name] = digest
    expected = {"VALIDATION_VERIFIED.json", "SELECTION.json"}
    if set(entries) != expected:
        raise ValueError(f"Unexpected selector manifest entries: {set(entries)}")
    for name, expected_hash in entries.items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"Selector artifact hash mismatch: {name}")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    verify_manifest(args.selection_dir)
    protocol_sha = sha256(args.protocol)
    verified_path = args.selection_dir / "VALIDATION_VERIFIED.json"
    selection_path = args.selection_dir / "SELECTION.json"
    verified = json.loads(verified_path.read_text())
    selection = json.loads(selection_path.read_text())
    expected_verified = {
        "schema": "sallm.general_sequence_validation_verified/v1",
        "status": "VERIFIED",
        "data_boundary": "validation_only",
        "heldout_accessed": False,
        "protocol_sha256": protocol_sha,
        "rows": {"ner": 4 * 10760, "pos": 4 * 1800},
        "ner_metric_authority": "canonical row-derived task-pack aggregation",
    }
    if any(verified.get(key) != value for key, value in expected_verified.items()):
        raise ValueError("Validation verification contract mismatch")
    if set(verified.get("input_sha256", {})) != {"ner", "pos"}:
        raise ValueError("Validation input hash coverage mismatch")
    if any(
        set(verified["input_sha256"][task])
        != {"mzansilm", "mamba2", "xlstm", "gdn"}
        for task in ("ner", "pos")
    ):
        raise ValueError("Validation architecture hash coverage mismatch")
    selection_sha = sha256(selection_path)
    expected_selection = {
        "schema": "sallm.general_sequence_selection/v1",
        "selection_data": "validation_only",
        "architectures": ["mzansilm", "mamba2", "xlstm", "gdn"],
        "maximum_input_tokens": 1024,
        "validation_verified_sha256": sha256(verified_path),
        "protocol_sha256": protocol_sha,
        "ner_metric_amendment_sha256": verified["ner_metric_amendment_sha256"],
    }
    if any(selection.get(key) != value for key, value in expected_selection.items()):
        raise ValueError("Frozen selection contract mismatch")
    selected = selection.get("selected_prompts", {})
    if set(selected) != {"ner", "pos"}:
        raise ValueError("Selected task coverage mismatch")
    for task, maximum in (("ner", 5), ("pos", 4)):
        if set(selected[task]) != {"tsn", "xho", "zul"}:
            raise ValueError(f"Selected language coverage mismatch for {task}")
        if any(not 1 <= int(prompt) <= maximum for prompt in selected[task].values()):
            raise ValueError(f"Invalid selected prompt for {task}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    ready = {
        "schema": "sallm.general_sequence_test_ready/v1",
        "status": "VALIDATION_SELECTION_FROZEN_HELDOUT_LOCKED",
        "authorization": "explicit_user_request_2026-09-16",
        "selection_sha256": selection_sha,
        "validation_verified_sha256": sha256(verified_path),
        "protocol_sha256": protocol_sha,
        "required_release_files": {
            "ner": "NER_TEST_ACCESS_RELEASED_V1.json",
            "pos": "POS_TEST_ACCESS_RELEASED_V1.json",
        },
        "one_time_test": True,
        "no_score_based_retry": True,
    }
    paths = [args.output_dir / "READY_FOR_TEST.json"]
    atomic_json(paths[0], ready)
    for task, filename in ready["required_release_files"].items():
        path = args.output_dir / filename
        atomic_json(
            path,
            {
                "schema": "sallm.general_sequence_test_release/v1",
                "task": task,
                "selection_sha256": selection_sha,
                "authorized": True,
                "one_time_test": True,
                "no_score_based_retry": True,
            },
        )
        paths.append(path)
    manifest = args.output_dir / "ARTIFACTS.sha256"
    manifest.write_text("".join(f"{sha256(path)}  {path.name}\n" for path in paths))
    print(
        "GENERAL_TEST_RELEASE_SEALED "
        f"selection_sha256={selection_sha} tasks=ner,pos"
    )


if __name__ == "__main__":
    main()
