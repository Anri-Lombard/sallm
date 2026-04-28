from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from sallm.evaluation.config import TaskPack

CONF_DIR = Path(__file__).resolve().parent.parent.parent.parent / "conf"
TASK_DIR = CONF_DIR / "eval" / "tasks"
RERANK_TASK_DIR = CONF_DIR / "rerank" / "tasks"
RERANK_LM_EVAL_TASK_DIR = CONF_DIR / "rerank" / "lm_eval_tasks"

_CACHE: dict[tuple[str, str], TaskPack] = {}


def _load_task_pack_from_dir(key: str, task_dir: Path, scope: str) -> TaskPack:
    """Load a task pack by name from an explicit config namespace."""
    cache_key = (scope, key)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    yaml_path = task_dir / f"{key}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Task-pack YAML '{yaml_path}' not found.")

    with yaml_path.open("r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Task-pack YAML '{yaml_path}' is empty.")

    cfg["name"] = key

    try:
        pack = TaskPack(**cfg)
    except ValidationError as e:
        raise ValueError(f"Invalid Task-pack YAML '{yaml_path}': {e}") from e

    _CACHE[cache_key] = pack
    return pack


def load_task_pack(key: str) -> TaskPack:
    """Load a final/test evaluation task pack by name."""
    if key.endswith("_val"):
        raise ValueError(
            f"Task pack '{key}' is validation-scoped. Final evaluation may only "
            "load test/final packs from src/conf/eval/tasks; use "
            "load_rerank_task_pack() for validation reranking."
        )
    return _load_task_pack_from_dir(key, TASK_DIR, "eval")


def load_rerank_task_pack(key: str) -> TaskPack:
    """Load a validation/rerank task pack by name."""
    return _load_task_pack_from_dir(key, RERANK_TASK_DIR, "rerank")
