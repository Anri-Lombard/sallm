#!/usr/bin/env python3
"""Verify an official lm-eval task-pack artifact without exposing metrics."""

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


def verify_task_pack_eval(
    root: Path,
    *,
    expected_pack: str,
    expected_task_prefix: str,
    expected_rows: int,
    expected_prompt_count: int = 5,
    expected_task_suffix: str = "_test",
    required_metric: str = "f1,flexible-extract",
) -> dict[str, Any]:
    root = root.resolve()
    summary_path = root / "evaluation_summary.json"
    result_path = root / expected_pack / "results.json"
    summary = _read_json(summary_path)
    result = _read_json(result_path)
    expected_tasks = {
        f"{expected_task_prefix}_prompt_{index}{expected_task_suffix}"
        for index in range(1, expected_prompt_count + 1)
    }

    if not isinstance(summary, list) or len(summary) != 1:
        raise ValueError("evaluation summary must contain exactly one task pack")
    pack_summary = summary[0]
    if (
        pack_summary.get("type") != "lm_eval"
        or pack_summary.get("task_pack") != expected_pack
        or set(pack_summary.get("tasks", [])) != expected_tasks
        or Path(pack_summary.get("result_path", "")).resolve() != result_path
    ):
        raise ValueError("unexpected task-pack summary")
    if pack_summary.get("results") != result.get("results") or pack_summary.get(
        "metrics"
    ) != result.get("metrics", {}):
        raise ValueError("task-pack result artifacts disagree")

    results = result.get("results")
    counts = result.get("n-samples")
    samples = result.get("samples")
    if not all(isinstance(value, dict) for value in (results, counts, samples)):
        raise ValueError("missing task-pack result, coverage, or sample mappings")
    if set(results) != expected_tasks or set(counts) != expected_tasks:
        raise ValueError("unexpected task set")
    if set(samples) != expected_tasks:
        raise ValueError("unexpected sample set")

    rows = 0
    for task in expected_tasks:
        metric = results[task].get(required_metric)
        if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
            raise ValueError(f"missing finite {required_metric} for {task}")
        task_samples = samples[task]
        if not isinstance(task_samples, list):
            raise ValueError(f"samples for {task} must be a list")
        declared = counts[task]
        if declared != {
            "original": len(task_samples),
            "effective": len(task_samples),
        }:
            raise ValueError(f"coverage artifacts disagree for {task}")
        rows += len(task_samples)
    if rows != expected_rows:
        raise ValueError(f"expected {expected_rows} samples, found {rows}")

    return {
        "schema": "sallm_official_task_pack_eval_verification/v1",
        "verified": True,
        "task_pack": expected_pack,
        "tasks": sorted(expected_tasks),
        "rows": rows,
        "required_metric_present": required_metric,
        "metric_values_included": False,
        "artifact_sha256": {
            str(path.relative_to(root)): _sha256(path)
            for path in (summary_path, result_path)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-pack", required=True)
    parser.add_argument("--expected-task-prefix", required=True)
    parser.add_argument("--expected-prompt-count", type=int, default=5)
    parser.add_argument("--expected-task-suffix", default="_test")
    parser.add_argument("--required-metric", default="f1,flexible-extract")
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_task_pack_eval(
        args.root,
        expected_pack=args.expected_pack,
        expected_task_prefix=args.expected_task_prefix,
        expected_prompt_count=args.expected_prompt_count,
        expected_task_suffix=args.expected_task_suffix,
        required_metric=args.required_metric,
        expected_rows=args.expected_rows,
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Verified {result['rows']} official rows for {result['task_pack']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
