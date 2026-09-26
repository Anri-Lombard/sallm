#!/usr/bin/env python3
"""Verify a Kombuys RTX 5090 result against the ratified A100 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hashes_by_name(values: dict[str, str]) -> dict[str, str]:
    return {Path(path).name: digest for path, digest in values.items()}


def runtime_recorded(manifest: dict[str, Any]) -> bool:
    environment = manifest.get("environment", {})
    return all(environment.get(key) for key in ("python", "executable", "platform", "packages"))


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
        all(a[key] == b[key] for key in ("predictions", "gold", "tokens", "correct", "cell"))
        for a, b in zip(left_rows, right_rows, strict=True)
    )
    candidate_variables = candidate_manifest["environment"]["variables"]
    checks = {
        "execution_manifest_schema_match": (
            reference_manifest["schema"]
            == candidate_manifest["schema"]
            == "sallm_execution_manifest/v1"
        ),
        "entrypoint_match": (
            reference_manifest["entrypoint"]
            == candidate_manifest["entrypoint"]
            == "scripts/run_pos_runtime_equivalence_gate.sh"
        ),
        "source_manifest_match": (
            bool(reference_manifest["source_hashes"])
            and reference_manifest["source_hashes"] == candidate_manifest["source_hashes"]
        ),
        "model_manifest_match": (
            bool(reference_manifest["artifact_hashes"])
            and hashes_by_name(reference_manifest["artifact_hashes"])
            == hashes_by_name(candidate_manifest["artifact_hashes"])
        ),
        "runtime_recorded": runtime_recorded(reference_manifest) and runtime_recorded(candidate_manifest),
        "stable_environment_set": all(
            candidate_variables.get(key) == "1"
            for key in ("FLA_DISABLE_BACKEND_DISPATCH", "SALLM_SKIP_MAMBA_KERNEL_CHECK")
        ),
        "one_visible_cuda_device": candidate_variables.get("CUDA_VISIBLE_DEVICES") == "0",
        "schema_match": (
            reference["schema"] == candidate["schema"] == "sallm_pos_runtime_equivalence/v1"
        ),
        "validation_only": (
            reference["data_boundary"]
            == candidate["data_boundary"]
            == "validation-only; no held-out split loaded or scored"
        ),
        "hardware_pair_match": (
            reference["slurm_job_gres"] == "gpu:ampere80:1"
            and "A100" in reference["gpu"]
            and candidate["slurm_job_gres"] == "gpu:rtx5090:1"
            and candidate["host"] == "kombuys"
            and candidate["gpu"] == "NVIDIA GeForce RTX 5090"
        ),
        "artifact_identity_match": (
            reference["adapter_hashes"] == candidate["adapter_hashes"]
            and reference["validation_rows"] == candidate["validation_rows"]
            and reference["subset_indices"] == candidate["subset_indices"]
        ),
        "implementation_match": left["implementation"] == right["implementation"] == "full_prefix_v1",
        "row_count_match": len(left_rows) == len(right_rows) == 12,
        "predictions_match": rows_match,
        "cell_counts_match": left["cell_counts"] == right["cell_counts"],
        "cell_metrics_match": left["cell_metrics"] == right["cell_metrics"],
        "aggregate_accuracy_match": left["all_token_accuracy"] == right["all_token_accuracy"],
        "max_score_difference_at_most_0_05": max(score_diffs) <= 0.05,
        "mean_score_difference_at_most_0_01": sum(score_diffs) / len(score_diffs) <= 0.01,
    }
    return {
        "schema": "sallm_cross_host_hardware_equivalence_verification/v1",
        "data_boundary": "validation-only; no held-out split loaded or scored",
        "reference_artifact": reference,
        "candidate_artifact": candidate,
        "reference_execution_manifest": reference_manifest,
        "candidate_execution_manifest": candidate_manifest,
        "score_differences": {
            "maximum": max(score_diffs),
            "mean": sum(score_diffs) / len(score_diffs),
        },
        "runtime_ratio_reference_over_candidate": left["elapsed_seconds"] / right["elapsed_seconds"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-verification", type=Path, required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actual_reference_sha256 = sha256_file(args.reference_verification)
    if actual_reference_sha256 != args.expected_reference_sha256:
        raise SystemExit(
            f"Reference verification hash mismatch: {actual_reference_sha256}"
        )
    prior = json.loads(args.reference_verification.read_text(encoding="utf-8"))
    if prior.get("passed") is not True:
        raise SystemExit("Reference A100 hardware verification did not pass")
    result = verify(
        prior["candidate_artifact"],
        json.loads(args.candidate.read_text(encoding="utf-8")),
        prior["candidate_execution_manifest"],
        json.loads(args.candidate_manifest.read_text(encoding="utf-8")),
    )
    result["reference_verification"] = {
        "path": str(args.reference_verification),
        "sha256": actual_reference_sha256,
        "passed": True,
    }
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
