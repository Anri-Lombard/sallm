#!/usr/bin/env python3
"""Verify exact coverage of the 97-unit, 293-cell official attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    release = json.loads(args.release.read_text())
    if not release.get("authorized") or not release.get("no_score_based_retry"):
        raise ValueError("Release gate differs")
    units = json.loads((args.inventory / "official_units.json").read_text())
    verified = []
    for unit in units:
        index = int(unit["array_index"])
        group = unit["task_group"]
        if group in {"ner", "pos"}:
            path = args.official_root / "sequence" / f"{index}.json"
            payload = json.loads(path.read_text())
            if payload.get("phase") != "test" or payload.get("task") != group:
                raise ValueError(f"Sequence scope mismatch: {path}")
            if set(payload.get("reported_metrics", {})) == set() or not payload.get("rows"):
                raise ValueError(f"Sequence output is empty: {path}")
        elif group == "generation":
            path = args.official_root / "generation" / str(index) / "UNIT_VERIFIED.json"
            payload = json.loads(path.read_text())
            if len(payload.get("tasks", [])) != len(unit["cell_ids"]):
                raise ValueError(f"Generation coverage mismatch: {path}")
        else:
            path = args.official_root / "prompt" / f"{index}.json"
            payload = json.loads(path.read_text())
            if payload.get("mode") != "official" or payload.get("item") != unit:
                raise ValueError(f"Prompt scope mismatch: {path}")
            observed = {
                task
                for call in payload.get("calls", {}).values()
                for task in call.get("results", {})
            }
            if observed != set(unit["task_names"]):
                raise ValueError(f"Prompt coverage mismatch: {path}")
        verified.append(
            {
                "array_index": index,
                "unit_id": unit["unit_id"],
                "cells": unit["cells"],
                "artifact": str(path),
                "sha256": sha256(path),
            }
        )
    if len(verified) != 97 or sum(item["cells"] for item in verified) != 293:
        raise ValueError("Verified coverage differs")
    payload = {
        "schema": "sallm.full_matrix_official_verified/v1",
        "status": "OFFICIAL_VERIFIED",
        "attempt_id": release["attempt_id"],
        "unit_count": 97,
        "cell_count": 293,
        "no_score_based_retry": True,
        "units": verified,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("status", "unit_count", "cell_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
