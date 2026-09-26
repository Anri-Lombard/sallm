#!/usr/bin/env python3
"""Verify the preregistered serial versus row-batched POS gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def verify(payload: dict[str, Any]) -> dict[str, Any]:
    serial = payload["results"]["serial"]
    batched = payload["results"]["batched"]
    serial_rows = serial["rows"]
    batched_rows = batched["rows"]
    score_diffs = [
        abs(left["selected_sequence_score"] - right["selected_sequence_score"])
        for left, right in zip(serial_rows, batched_rows, strict=True)
    ]
    predictions_match = all(
        left["predictions"] == right["predictions"]
        and left["gold"] == right["gold"]
        and left["tokens"] == right["tokens"]
        and left["correct"] == right["correct"]
        and left["cell"] == right["cell"]
        for left, right in zip(serial_rows, batched_rows, strict=True)
    )
    checks = {
        "schema_match": payload["schema"] == "sallm_pos_row_batch_equivalence/v1",
        "implementations_match": (
            serial["implementation"] == "full_prefix_v1"
            and serial["row_batch_size"] == 1
            and batched["implementation"] == "row_batch_full_prefix_v1"
            and batched["row_batch_size"] == 8
        ),
        "row_count_match": len(serial_rows) == len(batched_rows) == 12,
        "predictions_match": predictions_match,
        "cell_counts_match": serial["cell_counts"] == batched["cell_counts"],
        "cell_metrics_match": serial["cell_metrics"] == batched["cell_metrics"],
        "aggregate_accuracy_match": (
            serial["all_token_accuracy"] == batched["all_token_accuracy"]
        ),
        "max_score_difference_at_most_0_05": max(score_diffs) <= 0.05,
        "mean_score_difference_at_most_0_01": (
            sum(score_diffs) / len(score_diffs) <= 0.01
        ),
        "batched_time_at_most_one_third": (
            batched["elapsed_seconds"] <= serial["elapsed_seconds"] / 3
        ),
    }
    return {
        "schema": "sallm_pos_row_batch_equivalence_verification/v1",
        "source_artifact": payload,
        "score_differences": {
            "maximum": max(score_diffs),
            "mean": sum(score_diffs) / len(score_diffs),
        },
        "speedup": serial["elapsed_seconds"] / batched["elapsed_seconds"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(json.loads(args.input.read_text(encoding="utf-8")))
    encoded = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
    args.output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    args.output.with_suffix(".json.sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(f"Wrote {args.output} ({digest}); passed={result['passed']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
