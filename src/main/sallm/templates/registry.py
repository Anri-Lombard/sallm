from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import yaml
from pydantic import BaseModel

_TEMPLATE_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "configs"
    / "templates"
)
_CACHE: Dict[str, "TemplateSpec"] = {}
_TASK_INDEX: Dict[str, List[str]] = {}


class TemplateSpec(BaseModel):
    id: str
    prompt: str
    label_mapping: Dict[int | str, str]
    task: str


def _load_all() -> None:
    for path in _TEMPLATE_ROOT.rglob("*.yaml"):
        with path.open("r") as f:
            cfg = yaml.safe_load(f)
        spec = TemplateSpec(**cfg)
        _CACHE[spec.id] = spec
        _TASK_INDEX.setdefault(spec.task, []).append(spec.id)


if not _CACHE:
    _load_all()


def get(template_id: str) -> TemplateSpec:
    return _CACHE[template_id]


def list_by_task(task: str) -> List[str]:
    return _TASK_INDEX.get(task, [])
