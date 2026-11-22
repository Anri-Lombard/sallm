from __future__ import annotations

import yaml
from pydantic import ValidationError

from sallm.evaluation.config import TaskPack
from sallm.paths import CONF_DIR

TASK_DIR = CONF_DIR / "eval" / "tasks"

_CACHE: dict[str, TaskPack] = {}


def load_task_pack(key: str, *, allow_validation: bool = False) -> TaskPack:
    """Load an evaluation task pack from `src/conf/eval/tasks`."""
    if key.endswith("_val") and not allow_validation:
        raise ValueError(
            f"Task pack '{key}' is validation-scoped. Set "
            "`evaluation.task_pack_scope=validation` (or pass "
            "`allow_validation=True`) to run validation packs."
        )
    if key in _CACHE:
        return _CACHE[key]

    yaml_path = TASK_DIR / f"{key}.yaml"
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

    _CACHE[key] = pack
    return pack
