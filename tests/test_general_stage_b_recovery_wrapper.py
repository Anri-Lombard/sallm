from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/run_general_stage_b_walltime_recovery.sh"


def _runtime_environment() -> dict[str, object]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": dict(
            sorted(
                (
                    distribution.metadata.get("Name", "unknown"),
                    distribution.version,
                )
                for distribution in importlib.metadata.distributions()
            )
        ),
    }


def test_recovery_wrapper_binds_candidate_checkpoint_and_frozen_launcher(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    scripts = snapshot / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "run_pure_gdn_validation_trial.sh").write_text(
        "frozen launcher\n", encoding="utf-8"
    )
    nested = scripts / "run_validation_hpo_trial.sh"
    nested.write_text(
        "#!/bin/bash\n"
        "printf 'args=%s\\n' \"$*\"\n"
        "printf 'resume=%s\\n' \"$SALLM_HPO_RESUME_FROM_CHECKPOINT\"\n"
        "printf 'repo=%s\\n' \"$SALLM_REPO_DIR\"\n",
        encoding="utf-8",
    )
    (scripts / "create_execution_manifest.py").write_text(
        "print('verified-full-manifest')\n",
        encoding="utf-8",
    )
    (snapshot / "deployment_manifest.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "deployment_manifest.json.sha256").write_text(
        "manifest sidecar\n", encoding="utf-8"
    )

    scratch = tmp_path / "scratch"
    output = (
        scratch
        / "masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/general/stage_b"
        / "b2/seed_42"
    )
    checkpoint = output / "checkpoint-10912"
    checkpoint.mkdir(parents=True)
    for name in (
        "adapter_model.bin",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    ):
        (checkpoint / name).write_text(name, encoding="utf-8")
    (output / "execution_manifest.json").write_text(
        json.dumps({"environment": _runtime_environment()}), encoding="utf-8"
    )
    archive = (
        scratch
        / "masters/sallm/recovery_archives/general-stage-b"
        / "b2-seed42-pre-recovery.tar"
    )
    archive.parent.mkdir(parents=True)
    archive.write_text("preserved interrupted root", encoding="utf-8")
    archive.chmod(0o444)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sha256sum = fake_bin / "sha256sum"
    hashes = {
        "run_pure_gdn_validation_trial.sh": (
            "45fec06bf2ea71d4f69a43a686c88b9b8d895750d85b797ed104ac59271cceab"
        ),
        "create_execution_manifest.py": (
            "55095f225dba0710b23831fb306ca3e471cbb1bbaaeec48529dd080fec235024"
        ),
        "deployment_manifest.json": (
            "8b6a744bc43d402aedccb44e6d62ef5f34144b95c7e871bfd7e2be823d0d9bb6"
        ),
        "deployment_manifest.json.sha256": (
            "4f5d47a49dabd4deaebe19a36aa8fe8c9a1e1f169e27ae2622e98cdc28c9a021"
        ),
        "verify_execution_runtime.py": (
            "cfe2eb873f53d83de2c2c296c6a172f74a8059ee4bd11f296785c444e9525b23"
        ),
        "adapter_model.bin": (
            "cc2a37683387dfc2ebb4969ab8337ce24f42bdf372d568be628545b3f34aedcb"
        ),
        "optimizer.pt": (
            "5a42ee1ef504c3a4e43990f04ad983142debf06691bde0f7ca1d2a6808bcaac8"
        ),
        "scheduler.pt": (
            "8ad971f1950fc2d8cbce77d99f29c9806ee058be8ba844f45d74d9a63fed6707"
        ),
        "rng_state.pth": (
            "3180a03567b7f40c2f694e184313fd4ca7212fda830074f2e9bc3350b97907e5"
        ),
        "trainer_state.json": (
            "fe862cfbfc8430edd220ec888a7b4ed3dab4151f26e26f0aa090d9341314c804"
        ),
        "b2-seed42-pre-recovery.tar": (
            "c82dc55f3ed836759015f68a9851318fc1bdd2e1584b74081cf0b45362705e96"
        ),
        "execution_manifest.json": (
            "32371bdd0c4f9a7b571a70b369cc6f1dccb9ac2665be4489b2d7be05dcec274e"
        ),
    }
    cases = "".join(
        f"  */{name}) hash={digest} ;;\n" for name, digest in hashes.items()
    )
    sha256sum.write_text(
        '#!/bin/bash\ncase "$1" in\n' + cases + "esac\n"
        'printf \'%s  %s\\n\' "$hash" "$1"\n',
        encoding="utf-8",
    )
    sha256sum.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SALLM_RECOVERY_SNAPSHOT": str(snapshot),
        "SALLM_RUNTIME_REPO": str(tmp_path / "runtime"),
        "SALLM_RUNTIME_PYTHON": sys.executable,
        "SALLM_RECOVERY_BUNDLE": str(ROOT / "scripts"),
        "SCRATCH": str(scratch),
    }
    preflight = subprocess.run(
        ["bash", str(SCRIPT), "b2", "--preflight-only"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "b2"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Recovery preflight verified" in preflight.stdout
    assert "args=" not in preflight.stdout
    assert "verified-full-manifest" in result.stdout
    assert "Verified exact execution runtime" in result.stdout
    assert "args=pure_gdn general stage_b b2 42" in result.stdout
    assert f"resume={checkpoint}" in result.stdout
    assert f"repo={snapshot}" in result.stdout

    sha256sum.write_text(
        sha256sum.read_text(encoding="utf-8").replace(
            hashes["execution_manifest.json"], "0" * 64
        ),
        encoding="utf-8",
    )
    mismatch = subprocess.run(
        ["bash", str(SCRIPT), "b2", "--preflight-only"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert mismatch.returncode != 0
    assert "execution_manifest.json" in mismatch.stderr


def test_recovery_wrapper_rejects_non_interrupted_candidate() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "b0"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "candidate must be b1, b2, or b3" in result.stderr
