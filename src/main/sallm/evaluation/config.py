from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["TaskPack"]

TASK_MANAGER_KWARG_KEYS = frozenset(
    {
        "include_path",
        "include_defaults",
    }
)


class TaskPack(BaseModel):
    name: str
    tasks: list[str]
    fewshot: int = 0
    batch_size: int | str = "auto:4"
    max_batch_size: int | None = 64
    apply_chat_template: bool = True
    lm_eval_kwargs: dict[str, Any] = Field(default_factory=dict)
    task_manager_kwargs: dict[str, Any] = Field(default_factory=dict)

    def to_evaluator_kwargs(self) -> dict[str, Any]:
        base = {
            "tasks": self.tasks,
            "batch_size": self.batch_size,
            "num_fewshot": self.fewshot,
        }
        if self.max_batch_size is not None:
            base["max_batch_size"] = self.max_batch_size
        base.update(
            {
                key: value
                for key, value in self.lm_eval_kwargs.items()
                if key not in TASK_MANAGER_KWARG_KEYS
            }
        )
        return base

    def to_task_manager_kwargs(self) -> dict[str, Any]:
        task_manager_kwargs = {
            key: value
            for key, value in self.lm_eval_kwargs.items()
            if key in TASK_MANAGER_KWARG_KEYS
        }
        task_manager_kwargs.update(self.task_manager_kwargs)
        return task_manager_kwargs

    def to_lm_eval_kwargs(self) -> dict[str, Any]:
        return self.to_evaluator_kwargs()
