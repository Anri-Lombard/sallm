#!/usr/bin/env python3
"""Create or verify a complete source/config/environment execution manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HASHED_SUFFIXES = {".json", ".lock", ".py", ".sh", ".toml", ".yaml", ".yml"}
HASHED_ROOTS = ("src/main/sallm", "src/conf", "scripts", "ops")
SAFE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "FLA_DISABLE_BACKEND_DISPATCH",
    "SALLM_DISABLE_TASK_METRICS",
    "SALLM_GENERAL_SELECTION_PROTOCOL",
    "SALLM_HPO_CANDIDATE",
    "SALLM_HPO_CONFIG",
    "SALLM_HPO_DATA_SEED",
    "SALLM_HPO_FAMILY",
    "SALLM_HPO_LEARNING_RATE",
    "SALLM_HPO_LORA_ALPHA",
    "SALLM_HPO_LORA_DROPOUT",
    "SALLM_HPO_LORA_RANK",
    "SALLM_HPO_MODEL_ARCHITECTURE",
    "SALLM_HPO_MODEL_ID",
    "SALLM_HPO_MODEL",
    "SALLM_HPO_LOGGING_DIR_OVERRIDE",
    "SALLM_HPO_OUTPUT_DIR_OVERRIDE",
    "SALLM_HPO_REGISTRY",
    "SALLM_HPO_RUN_ID_OVERRIDE",
    "SALLM_HPO_SEED",
    "SALLM_HPO_STAGE",
    "SALLM_HPO_TARGET_MODULES",
    "SALLM_HPO_TOKENIZER",
    "SALLM_HPO_WARMUP_RATIO",
    "SALLM_POS_INCREMENTAL_CACHE",
    "SALLM_POS_ROW_BATCH_SIZE",
    "SALLM_SKIP_MAMBA_KERNEL_CHECK",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_QOS",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(repo_root: Path) -> list[Path]:
    paths: set[Path] = set()
    for relative_root in HASHED_ROOTS:
        root = repo_root / relative_root
        if not root.exists():
            raise FileNotFoundError(f"Manifest source root does not exist: {root}")
        paths.update(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in HASHED_SUFFIXES
        )
    for name in ("pyproject.toml", "uv.lock"):
        path = repo_root / name
        if path.exists():
            paths.add(path)
    return sorted(paths)


def git_value(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def package_versions() -> dict[str, str]:
    return dict(
        sorted(
            (
                distribution.metadata.get("Name", "unknown"),
                distribution.version,
            )
            for distribution in importlib.metadata.distributions()
        )
    )


def artifact_hashes(root: Path | None) -> dict[str, str]:
    if root is None:
        return {}
    if not root.is_dir():
        raise FileNotFoundError(f"Artifact root does not exist: {root}")
    return {
        str(path): sha256_file(path)
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def create_manifest(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    files = source_files(repo_root)
    payload: dict[str, Any] = {
        "schema": "sallm_execution_manifest/v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repo_root": str(repo_root),
        "git_head": git_value(repo_root, "rev-parse", "HEAD"),
        "git_status_porcelain": git_value(repo_root, "status", "--porcelain=v1") or "",
        "entrypoint": args.entrypoint,
        "command": args.command,
        "source_hashes": {
            str(path.relative_to(repo_root)): sha256_file(path) for path in files
        },
        "artifact_hashes": artifact_hashes(args.artifact_root),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": package_versions(),
            "variables": {
                key: os.environ[key] for key in SAFE_ENV_KEYS if key in os.environ
            },
        },
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} ({digest})")
    return 0


def verify_manifest(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "sallm_execution_manifest/v1":
        raise SystemExit("Unsupported execution manifest schema")
    repo_root = Path(payload["repo_root"])
    expected_sources = payload["source_hashes"]
    current_sources = {
        str(source.relative_to(repo_root)): sha256_file(source)
        for source in source_files(repo_root)
    }
    mismatches: list[str] = []
    for relative in sorted(expected_sources.keys() | current_sources.keys()):
        expected = expected_sources.get(relative)
        actual = current_sources.get(relative)
        if actual != expected:
            mismatches.append(f"{relative}: expected={expected}, actual={actual}")
    for absolute, expected in payload.get("artifact_hashes", {}).items():
        artifact = Path(absolute)
        actual = sha256_file(artifact) if artifact.is_file() else None
        if actual != expected:
            mismatches.append(f"{absolute}: expected={expected}, actual={actual}")
    if mismatches:
        raise SystemExit(
            "Execution manifest verification failed:\n" + "\n".join(mismatches)
        )
    print(f"Verified {len(expected_sources)} source/config files from {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--entrypoint")
    parser.add_argument("--command", default="")
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        return verify_manifest(args.verify)
    if args.output is None or args.entrypoint is None:
        parser.error("--output and --entrypoint are required when creating a manifest")
    return create_manifest(args)


if __name__ == "__main__":
    raise SystemExit(main())
