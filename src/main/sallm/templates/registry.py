from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

_TEMPLATE_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent / "conf" / "templates"
)
_CACHE: dict[str, TemplateSpec] = {}
_TASK_INDEX: dict[str, list[str]] = {}


class TemplateSpec(BaseModel):
    id: str
    prompt: str
    label_mapping: dict[int | str, str] | None = None
    ner_tags: list[str] | None = None
    task: str


def _load_all() -> None:
    if not _TEMPLATE_ROOT.exists():
        print(f"ERROR: Template directory not found at {_TEMPLATE_ROOT}")
        return

    for path in _TEMPLATE_ROOT.rglob("*.yaml"):
        with path.open("r") as f:
            cfg = yaml.safe_load(f)

        relative_path = path.relative_to(_TEMPLATE_ROOT)
        template_id = str(relative_path.with_suffix(""))

        cfg["id"] = template_id

        spec = TemplateSpec(**cfg)
        _CACHE[spec.id] = spec
        _TASK_INDEX.setdefault(spec.task, []).append(spec.id)


if not _CACHE:
    _load_all()


def get(template_id: str) -> TemplateSpec:
    if not _CACHE:
        raise RuntimeError(
            "Template cache is empty. Check that the path '"
            + str(_TEMPLATE_ROOT)
            + "' is correct and contains template YAML files."
        )
    return _CACHE[template_id]


def list_by_task(task: str) -> list[str]:
    return _TASK_INDEX.get(task, [])
