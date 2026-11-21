"""Write canonical JSON validation artifacts alongside their SHA-256 digests."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import wandb

logger = logging.getLogger(__name__)


def write_validation_artifact(
    *,
    output_dir: str | Path,
    subdir: str,
    step: int,
    payload: dict[str, Any],
    label: str,
    save_to_wandb: bool = False,
) -> Path:
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    artifact_dir = Path(str(output_dir)) / "validation_artifacts" / subdir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"step-{int(step):08d}.json"
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    path.with_suffix(".json.sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )
    logger.info("Saved %s validation artifact %s (%s)", label, path, digest)
    if save_to_wandb and getattr(wandb, "run", None) is not None:
        wandb.save(str(path), policy="now")
    return path
