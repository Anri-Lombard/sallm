#!/usr/bin/env python3
"""Verify A100-40GB versus A100-80GB validation-only equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def verify(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    reference_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
) -> dict[str, Any]:
    left = reference["results"]["full"]
    right = candidate["results"]["full"]
    left_rows = left["rows"]
    right_rows = right["rows"]
    score_diffs = [
        abs(a["selected_sequence_score"] - b["selected_sequence_score"])
        for a, b in zip(left_rows, right_rows, strict=True)
    ]
    rows_match = all(
        a["predictions"] == b["predictions"]
        and a["gold"] == b["gold"]
        and a["tokens"] == b["tokens"]
        and a["correct"] == b["correct"]
        and a["cell"] == b["cell"]
        for a, b in zip(left_rows, right_rows, strict=True)
    )
    checks = {
        "execution_manifest_schema_match": (
            reference_manifest["schema"]
            == candidate_manifest["schema"]
            == "sallm_execution_manifest/v1"
        ),
        "immutable_snapshot_match": (
            bool(reference_manifest["repo_root"])
            and reference_manifest["repo_root"] == candidate_manifest["repo_root"]
            and reference_manifest["entrypoint"]
            == candidate_manifest["entrypoint"]
            == "scripts/run_pos_runtime_equivalence_gate.sh"
            and reference_manifest["git_head"] == candidate_manifest["git_head"]
            and reference_manifest["git_status_porcelain"]
            == candidate_manifest["git_status_porcelain"]
        ),
        "source_manifest_match": (
            bool(reference_manifest["source_hashes"])
            and reference_manifest["source_hashes"]
            == candidate_manifest["source_hashes"]
        ),
        "model_manifest_match": (
            bool(reference_manifest["artifact_hashes"])
            and reference_manifest["artifact_hashes"]
            == candidate_manifest["artifact_hashes"]
        ),
        "runtime_environment_match": all(
            reference_manifest["environment"][key]
            == candidate_manifest["environment"][key]
            for key in ("python", "executable", "platform", "packages")
        ),
        "stable_environment_match": all(
            reference_manifest["environment"]["variables"].get(key)
            == candidate_manifest["environment"]["variables"].get(key)
            == "1"
            for key in (
                "FLA_DISABLE_BACKEND_DISPATCH",
                "SALLM_SKIP_MAMBA_KERNEL_CHECK",
            )
        ),
        "schema_match": (
            reference["schema"]
            == candidate["schema"]
            == "sallm_pos_runtime_equivalence/v1"
        ),
        "validation_only": (
            reference["data_boundary"]
            == candidate["data_boundary"]
            == "validation-only; no held-out split loaded or scored"
        ),
        "hardware_pair_match": (
            reference["slurm_job_gres"] == "gpu:ampere:1"
            and candidate["slurm_job_gres"] == "gpu:ampere80:1"
        ),
        "artifact_identity_match": all(
            reference[key] == candidate[key]
            for key in (
                "checkpoint",
                "adapter",
                "adapter_hashes",
                "validation_rows",
                "subset_indices",
            )
        ),
        "implementation_match": (
            left["implementation"] == right["implementation"] == "full_prefix_v1"
        ),
        "row_count_match": len(left_rows) == len(right_rows) == 12,
        "predictions_match": rows_match,
        "cell_counts_match": left["cell_counts"] == right["cell_counts"],
        "cell_metrics_match": left["cell_metrics"] == right["cell_metrics"],
        "aggregate_accuracy_match": (
            left["all_token_accuracy"] == right["all_token_accuracy"]
        ),
        "max_score_difference_at_most_0_05": max(score_diffs) <= 0.05,
        "mean_score_difference_at_most_0_01": (
            sum(score_diffs) / len(score_diffs) <= 0.01
        ),
    }
    return {
        "schema": "sallm_a100_hardware_equivalence_verification/v1",
        "data_boundary": "validation-only; no held-out split loaded or scored",
        "reference_artifact": reference,
        "candidate_artifact": candidate,
        "reference_execution_manifest": reference_manifest,
        "candidate_execution_manifest": candidate_manifest,
        "score_differences": {
            "maximum": max(score_diffs),
            "mean": sum(score_diffs) / len(score_diffs),
        },
        "runtime_ratio_reference_over_candidate": (
            left["elapsed_seconds"] / right["elapsed_seconds"]
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        json.loads(args.reference.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
        json.loads(args.reference_manifest.read_text(encoding="utf-8")),
        json.loads(args.candidate_manifest.read_text(encoding="utf-8")),
    )
    encoded = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    args.output.with_suffix(".json.sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(f"Wrote {args.output} ({digest}); passed={result['passed']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
