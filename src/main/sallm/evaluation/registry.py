from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml
from pydantic import ValidationError

from sallm.evaluation.config import TaskPack

TASK_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "conf" / "eval" / "tasks"
)

_CACHE: Dict[str, TaskPack] = {}


def load_task_pack(key: str) -> TaskPack:
    """Load a *TaskPack* by name and fail fast if YAML is incomplete or malformed."""
    if key in _CACHE:
        return _CACHE[key]

    yaml_path = TASK_DIR / f"{key}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Task‑pack YAML '{yaml_path}' not found.")

    with yaml_path.open("r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Task‑pack YAML '{yaml_path}' is empty.")

    cfg["name"] = key

    try:
        pack = TaskPack(**cfg)
    except ValidationError as e:
        raise ValueError(f"Invalid Task‑pack YAML '{yaml_path}': {e}") from e

    _CACHE[key] = pack
    return pack
