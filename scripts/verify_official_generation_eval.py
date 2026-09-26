#!/usr/bin/env python3
"""Verify official generation-evaluation coverage without selecting on metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_generation_eval(
    root: Path,
    *,
    expected_task: str,
    expected_language: str,
    expected_rows: int,
    required_metric: str,
) -> dict[str, Any]:
    root = root.resolve()
    task_root = root / expected_task
    summary_path = root / "evaluation_summary.json"
    task_summary_path = task_root / "summary.json"
    metrics_path = task_root / "metrics.json"
    examples_path = task_root / "examples.jsonl"

    evaluation_summary = _read_json(summary_path)
    task_summary = _read_json(task_summary_path)
    metrics = _read_json(metrics_path)
    if evaluation_summary != [task_summary] or task_summary != {
        "task": expected_task,
        "metrics": metrics,
    }:
        raise ValueError("metric artifacts disagree")
    if required_metric not in metrics:
        raise ValueError(f"missing required metric: {required_metric}")
    if not metrics or any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise ValueError("metrics must be finite numbers")

    if not examples_path.is_file():
        raise ValueError(f"missing artifact: {examples_path}")
    with examples_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} examples, found {len(rows)}")
    for index, row in enumerate(rows):
        if row.get("task") != expected_task:
            raise ValueError(f"row {index} has unexpected task")
        if row.get("language") != expected_language:
            raise ValueError(f"row {index} has unexpected language")
        if not isinstance(row.get("prediction"), str) or not isinstance(
            row.get("reference"), str
        ):
            raise ValueError(f"row {index} lacks string prediction/reference")

    artifacts = (summary_path, task_summary_path, metrics_path, examples_path)
    return {
        "schema": "sallm_official_generation_eval_verification/v1",
        "verified": True,
        "task": expected_task,
        "language": expected_language,
        "rows": len(rows),
        "required_metric_present": required_metric,
        "metric_values_included": False,
        "artifact_sha256": {
            str(path.relative_to(root)): _sha256(path) for path in artifacts
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-task", required=True)
    parser.add_argument("--expected-language", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--required-metric", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_generation_eval(
        args.root,
        expected_task=args.expected_task,
        expected_language=args.expected_language,
        expected_rows=args.expected_rows,
        required_metric=args.required_metric,
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Verified {result['rows']} official rows for {result['task']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
