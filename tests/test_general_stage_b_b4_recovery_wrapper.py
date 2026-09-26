# ruff: noqa: E501 -- frozen SHA-256 bindings are intentionally literal

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("candidate", "bindings"),
    (
        (
            "b4",
            {
                "archive": "ab2c93160222ba8bfe0a48e086a67a1548cbcbfcc67ce0977e60aa583a4ac399",
                "manifest": "5791f6fb4dcb03c3cf41cc3f2d1c6ab91627f1cea71d05454c8bc645a0e7ae9e",
                "adapter": "edc7cdac66426ac5824903492a65da3987503309b5d4cbdfb06af0406c29be66",
                "optimizer": "f5cbd3ac2e9d0ab27464fde538bf01e3a2eb75f1652ea2bb260d668fdba04e5d",
                "scheduler": "ce4fcc97ebd7b7c36ccf7a5faa5df7b8a4794b4841d3f329517b029a0eff92a5",
                "rng": "d31146d7c73d1c62b0541ebcd0c731dcb2a9aa1652731b25333180d24e9d170f",
                "trainer": "13f8c6df16c4ac250b17642868499db350be5a0bf98b0160c8f7c4a52189abac",
            },
        ),
        (
            "b5",
            {
                "archive": "3f52518bf028a1a921ea7c4718654b5bd21e3f27115aa761c8a689dd92051595",
                "manifest": "662ed16cf8705aa230f94a721fa981eae8cf13d3965d212429cd837f551d6c19",
                "adapter": "252af47bbbb411a87ed3bc59bec8596a2845d8afc9c0cdfb093759be71434d5f",
                "optimizer": "7853c01504a5d823055b33eb31bd10400736a1750217f0074b29d2cf04018294",
                "scheduler": "b403be7999c394122e0db5d991abdd751bfe923ce60b4cb961d4b1de55a58413",
                "rng": "3a5fa191be89653519d53a698bb731547927ec65d27015072e0de1b7e71c27d3",
                "trainer": "51a64b5a7c3f09296dd8cb7b61bd8f4ef54bb0202bae9e695b66fc8f9d695f70",
            },
        ),
    ),
)
def test_recovery_preflight_binds_preserved_trial(
    tmp_path: Path, candidate: str, bindings: dict[str, str]
) -> None:
    snapshot = tmp_path / "snapshot"
    scripts = snapshot / "scripts"
    scripts.mkdir(parents=True)
    for name in (
        "run_pure_gdn_validation_trial.sh",
        "create_execution_manifest.py",
        "run_validation_hpo_trial.sh",
    ):
        (scripts / name).write_text("#!/bin/bash\n", encoding="utf-8")

    (snapshot / "deployment_manifest.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "deployment_manifest.json.sha256").write_text(
        "manifest sidecar\n", encoding="utf-8"
    )

    scratch = tmp_path / "scratch"
    output = (
        scratch
        / "masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/general/stage_b"
        / f"{candidate}/seed_42"
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
    (output / "execution_manifest.json").write_text("{}\n", encoding="utf-8")

    archive = (
        scratch
        / "masters/sallm/recovery_archives/general-stage-b"
        / f"{candidate}-seed42-pre-recovery.tar"
    )
    archive.parent.mkdir(parents=True)
    archive.write_text("preserved interrupted root", encoding="utf-8")
    archive.chmod(0o444)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sha256sum = fake_bin / "sha256sum"
    hashes = {
        "run_pure_gdn_validation_trial.sh": "45fec06bf2ea71d4f69a43a686c88b9b8d895750d85b797ed104ac59271cceab",
        "create_execution_manifest.py": "55095f225dba0710b23831fb306ca3e471cbb1bbaaeec48529dd080fec235024",
        "deployment_manifest.json": "8b6a744bc43d402aedccb44e6d62ef5f34144b95c7e871bfd7e2be823d0d9bb6",
        "deployment_manifest.json.sha256": "4f5d47a49dabd4deaebe19a36aa8fe8c9a1e1f169e27ae2622e98cdc28c9a021",
        "verify_execution_runtime.py": "cfe2eb873f53d83de2c2c296c6a172f74a8059ee4bd11f296785c444e9525b23",
        f"{candidate}-seed42-pre-recovery.tar": bindings["archive"],
        "execution_manifest.json": bindings["manifest"],
        "adapter_model.bin": bindings["adapter"],
        "optimizer.pt": bindings["optimizer"],
        "scheduler.pt": bindings["scheduler"],
        "rng_state.pth": bindings["rng"],
        "trainer_state.json": bindings["trainer"],
    }
    cases = "".join(
        f"  */{name}) hash={digest} ;;\n" for name, digest in hashes.items()
    )
    sha256sum.write_text(
        '#!/bin/bash\ncase "$1" in\n'
        + cases
        + "esac\n"
        + 'printf \'%s  %s\\n\' "$hash" "$1"\n',
        encoding="utf-8",
    )
    sha256sum.chmod(0o755)

    runtime_python = tmp_path / "runtime-python"
    runtime_python.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    runtime_python.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SALLM_RECOVERY_SNAPSHOT": str(snapshot),
        "SALLM_RUNTIME_REPO": str(tmp_path / "runtime"),
        "SALLM_RUNTIME_PYTHON": str(runtime_python),
        "SALLM_RECOVERY_BUNDLE": str(ROOT / "scripts"),
        "SCRATCH": str(scratch),
    }
    result = subprocess.run(
        [
            "bash",
            str(ROOT / f"scripts/run_general_stage_b_{candidate}_walltime_recovery.sh"),
            "--preflight-only",
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert f"Recovery preflight verified: candidate={candidate}" in result.stdout
    assert "checkpoint-10912" in result.stdout
