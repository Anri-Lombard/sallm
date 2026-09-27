import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_cross_host_hardware_equivalence import verify  # noqa: E402
from tests.test_a100_hardware_equivalence import _artifact, _manifest  # noqa: E402


def test_cross_host_gate_accepts_exact_outputs_with_declared_runtime_difference() -> (
    None
):
    reference = _artifact("gpu:ampere80:1", -1.0, 30.0)
    reference["host"] = "srvrocgpu011"
    candidate = copy.deepcopy(reference)
    candidate.update(
        host="kombuys",
        gpu="NVIDIA GeForce RTX 5090",
        slurm_job_gres="gpu:rtx5090:1",
    )
    reference_manifest = _manifest("80")
    candidate_manifest = copy.deepcopy(reference_manifest)
    candidate_manifest["repo_root"] = "/scratch/alombard/immutable-snapshot"
    candidate_manifest["environment"]["packages"]["torch"] = "2.9.1"
    candidate_manifest["environment"]["variables"].update(
        CUDA_VISIBLE_DEVICES="0",
        FLA_DISABLE_BACKEND_DISPATCH="1",
        SALLM_SKIP_MAMBA_KERNEL_CHECK="1",
    )

    result = verify(reference, candidate, reference_manifest, candidate_manifest)

    assert result["passed"] is True
    assert result["checks"]["runtime_recorded"] is True
    assert "runtime_environment_match" not in result["checks"]

    candidate["results"]["full"]["rows"][0]["predictions"] = ["VERB", "VERB"]
    rejected = verify(reference, candidate, reference_manifest, candidate_manifest)
    assert rejected["passed"] is False
    assert rejected["checks"]["predictions_match"] is False
