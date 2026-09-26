from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/verify_execution_runtime.py"


def _environment() -> dict[str, object]:
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


def test_runtime_verifier_fails_closed_on_package_drift(tmp_path: Path) -> None:
    manifest = tmp_path / "execution_manifest.json"
    environment = _environment()
    manifest.write_text(json.dumps({"environment": environment}), encoding="utf-8")

    subprocess.run([sys.executable, SCRIPT, manifest], check=True)

    environment["packages"]["transformers"] = "0.invalid"  # type: ignore[index]
    manifest.write_text(json.dumps({"environment": environment}), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, SCRIPT, manifest],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "packages" in result.stderr
