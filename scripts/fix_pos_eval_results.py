#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import stdev


def unwrap_singleton_lists(value):
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return value


def token_accuracy_from_sample(sample: dict) -> float:
    ref = unwrap_singleton_lists(sample.get("target"))
    pred = unwrap_singleton_lists(sample.get("filtered_resps"))

    if not isinstance(ref, list):
        return 0.0
    if not isinstance(pred, list):
        return 0.0
    if not ref:
        return 0.0

    aligned = min(len(ref), len(pred))
    correct = sum(1 for i in range(aligned) if str(ref[i]) == str(pred[i]))
    return correct / len(ref)


def iter_results_files(paths: list[Path]):
    for path in paths:
        if path.is_file():
            yield path
            continue
        if path.is_dir():
            yield from sorted(path.rglob("results.json"))


def fix_results_file(path: Path, *, dry_run: bool) -> dict[str, float]:
    data = json.loads(path.read_text())
    updated: dict[str, float] = {}

    for task_name, samples in data.get("samples", {}).items():
        result_row = data.get("results", {}).get(task_name)
        if not isinstance(result_row, dict):
            continue
        if "token_accuracy,flexible-extract" not in result_row:
            continue

        sample_scores = [token_accuracy_from_sample(sample) for sample in samples]
        if not sample_scores:
            continue

        mean_score = sum(sample_scores) / len(sample_scores)
        stderr = (
            stdev(sample_scores) / math.sqrt(len(sample_scores))
            if len(sample_scores) > 1
            else 0.0
        )

        result_row["token_accuracy,flexible-extract"] = mean_score
        result_row["token_accuracy_stderr,flexible-extract"] = stderr

        for sample, score in zip(samples, sample_scores):
            sample["token_accuracy"] = score

        updated[task_name] = mean_score

    if updated and not dry_run:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    return updated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair POS lm-eval results.json files from sample-level outputs."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="results.json files or directories to scan recursively",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print corrected scores without rewriting files",
    )
    args = parser.parse_args()

    changed = 0
    for path in iter_results_files(args.paths):
        updated = fix_results_file(path, dry_run=args.dry_run)
        if not updated:
            continue
        changed += 1
        print(path)
        for task_name, score in updated.items():
            print(f"  {task_name}: {score:.6f}")

    print(f"updated_files={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
