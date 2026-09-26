#!/usr/bin/env python3
"""Create one immutable official attempt and task releases after validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--sequence-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    validation_marker = args.selection_root / "VALIDATION_VERIFIED.json"
    selection_path = args.selection_root / "SELECTION.json"
    validation = json.loads(validation_marker.read_text())
    if validation.get("status") != "VALIDATION_VERIFIED" or validation.get("test_accessed"):
        raise ValueError("Validation gate is not clean")
    selection = json.loads(selection_path.read_text())
    units_path = args.inventory / "official_units.json"
    units = json.loads(units_path.read_text())
    if len(units) != 97 or sum(int(unit["cells"]) for unit in units) != 293:
        raise ValueError("Official scope differs")
    args.output.mkdir(parents=True, exist_ok=False)
    attempt_id = f"full-matrix-v1-{os.environ.get('SLURM_JOB_ID', 'manual')}"
    common = {
        "schema": "sallm.full_matrix_official_release/v1",
        "attempt_id": attempt_id,
        "authorized": True,
        "one_time_test": True,
        "no_score_based_retry": True,
        "operational_resume_missing_only": True,
        "inventory_sha256": sha256(units_path),
        "bindings_sha256": sha256(args.bindings),
        "selection_sha256": sha256(selection_path),
        "validation_verified_sha256": sha256(validation_marker),
        "selected_bindings": selection["selected_bindings"],
        "official_units": 97,
        "official_cells": 293,
    }
    write_once(args.output / "OFFICIAL_ATTEMPT.json", common)
    for task in ("ner", "pos"):
        release = {
            "schema": "sallm.general_sequence_test_release/v1",
            "task": task,
            "selection_sha256": sha256(args.sequence_selection),
            "authorized": True,
            "one_time_test": True,
            "no_score_based_retry": True,
        }
        write_once(args.output / f"{task.upper()}_TEST_ACCESS_RELEASED.json", release)
    write_once(args.output / "READY_FOR_OFFICIAL.json", common)
    print(json.dumps(common, sort_keys=True))


if __name__ == "__main__":
    main()
