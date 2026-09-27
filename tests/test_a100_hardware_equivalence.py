import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_a100_hardware_equivalence import verify  # noqa: E402


def _artifact(gres: str, score: float, elapsed: float) -> dict:
    row = {
        "cell": "xho/p1",
        "tokens": 2,
        "predictions": ["NOUN", "VERB"],
        "gold": ["NOUN", "VERB"],
        "correct": 2,
        "selected_sequence_score": score,
    }
    result = {
        "implementation": "full_prefix_v1",
        "elapsed_seconds": elapsed,
        "rows": [dict(row) for _ in range(12)],
        "cell_counts": {"xho/p1": {"correct": 24, "total": 24}},
        "cell_metrics": {"xho_p1_token_accuracy": 1.0},
        "all_token_accuracy": 1.0,
    }
    return {
        "schema": "sallm_pos_runtime_equivalence/v1",
        "data_boundary": "validation-only; no held-out split loaded or scored",
        "gpu": "NVIDIA A100",
        "slurm_job_gres": gres,
        "checkpoint": "/model",
        "adapter": "/adapter",
        "adapter_hashes": {"adapter_model.safetensors": "abc"},
        "validation_rows": 1800,
        "subset_indices": list(range(12)),
        "results": {"full": result},
    }


def _manifest(job_id: str) -> dict:
    return {
        "schema": "sallm_execution_manifest/v1",
        "created_at_utc": f"2026-08-22T20:00:{job_id}+00:00",
        "repo_root": "/immutable/snapshot",
        "git_head": None,
        "git_status_porcelain": "",
        "entrypoint": "scripts/run_pos_runtime_equivalence_gate.sh",
        "command": f"mode=full gres=job-{job_id}",
        "source_hashes": {"scripts/gate.py": "source-hash"},
        "artifact_hashes": {"/model/config.json": "model-hash"},
        "environment": {
            "python": "3.12",
            "executable": "/venv/python",
            "platform": "linux",
            "packages": {"torch": "2.7"},
            "variables": {
                "FLA_DISABLE_BACKEND_DISPATCH": "1",
                "SALLM_SKIP_MAMBA_KERNEL_CHECK": "1",
                "SLURM_JOB_ID": job_id,
            },
        },
    }


def test_a100_gate_accepts_matching_outputs_without_speed_threshold() -> None:
    reference_manifest = _manifest("40")
    candidate_manifest = _manifest("80")
    verified = verify(
        _artifact("gpu:ampere:1", -1.0, 12.0),
        _artifact("gpu:ampere80:1", -0.995, 30.0),
        reference_manifest,
        candidate_manifest,
    )

    assert verified["passed"] is True
    assert abs(verified["score_differences"]["maximum"] - 0.005) < 1e-12
    assert "runtime_threshold" not in verified["checks"]

    candidate_manifest["source_hashes"]["scripts/gate.py"] = "different"
    rejected = verify(
        _artifact("gpu:ampere:1", -1.0, 12.0),
        _artifact("gpu:ampere80:1", -0.995, 30.0),
        reference_manifest,
        candidate_manifest,
    )
    assert rejected["passed"] is False
    assert rejected["checks"]["source_manifest_match"] is False
