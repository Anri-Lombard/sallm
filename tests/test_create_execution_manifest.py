import importlib.util
import json
from pathlib import Path

import pytest


def _load_manifest_module():
    path = Path(__file__).parents[1] / "scripts" / "create_execution_manifest.py"
    spec = importlib.util.spec_from_file_location("create_execution_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_manifest_can_require_the_recorded_runtime(
    monkeypatch, tmp_path
) -> None:
    manifest = _load_manifest_module()
    monkeypatch.setattr(manifest.sys, "version", "python-version")
    monkeypatch.setattr(manifest.sys, "executable", "/runtime/python")
    monkeypatch.setattr(manifest.platform, "platform", lambda: "platform")
    monkeypatch.setattr(manifest, "package_versions", lambda: {"torch": "version"})
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "repo_root": str(tmp_path),
                "source_hashes": {},
                "artifact_hashes": {},
                "environment": {
                    "python": "python-version",
                    "executable": "/runtime/python",
                    "platform": "platform",
                    "packages": {"torch": "version"},
                    "variables": {"SLURM_JOB_ID": "creation-context"},
                },
            }
        )
    )

    assert manifest.verify_manifest(path, verify_runtime=True) == 0

    with pytest.raises(SystemExit, match="repo_root"):
        manifest.verify_manifest(path, expected_repo_root=tmp_path / "other")
    with pytest.raises(SystemExit, match="artifact_root"):
        manifest.verify_manifest(path, expected_artifact_root=tmp_path / "checkpoint")

    monkeypatch.setattr(manifest, "package_versions", lambda: {"torch": "changed"})
    with pytest.raises(SystemExit, match="environment.packages"):
        manifest.verify_manifest(path, verify_runtime=True)


def test_verify_manifest_allows_linux_kernel_patch_drift(monkeypatch, tmp_path) -> None:
    manifest = _load_manifest_module()
    monkeypatch.setattr(manifest.sys, "version", "python-version")
    monkeypatch.setattr(manifest.sys, "executable", "/runtime/python")
    monkeypatch.setattr(
        manifest.platform,
        "platform",
        lambda: "Linux-5.14.0-687.29.1.el9_8.x86_64-x86_64-with-glibc2.34",
    )
    monkeypatch.setattr(manifest, "package_versions", lambda: {"torch": "version"})
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "repo_root": str(tmp_path),
                "source_hashes": {},
                "artifact_hashes": {},
                "environment": {
                    "python": "python-version",
                    "executable": "/runtime/python",
                    "platform": (
                        "Linux-5.14.0-687.30.1.el9_8.x86_64-x86_64-with-glibc2.34"
                    ),
                    "packages": {"torch": "version"},
                },
            }
        )
    )

    assert manifest.verify_manifest(path, verify_runtime=True) == 0

    monkeypatch.setattr(
        manifest.platform,
        "platform",
        lambda: "Linux-6.12.0-1.1.1.el9_8.x86_64-x86_64-with-glibc2.34",
    )
    with pytest.raises(SystemExit, match="environment.platform"):
        manifest.verify_manifest(path, verify_runtime=True)

    monkeypatch.setattr(
        manifest.platform,
        "platform",
        lambda: "Linux-5.14.0-687.29.1.el8_9.x86_64-x86_64-with-glibc2.34",
    )
    with pytest.raises(SystemExit, match="environment.platform"):
        manifest.verify_manifest(path, verify_runtime=True)
