from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["TaskPack"]


class TaskPack(BaseModel):
    name: str
    tasks: list[str]
    fewshot: int = 0
    batch_size: int | str = "auto:4"
    max_batch_size: int | None = 64
    apply_chat_template: bool = True
    lm_eval_kwargs: dict[str, Any] = Field(default_factory=dict)

    def to_lm_eval_kwargs(self) -> dict[str, Any]:
        base = {
            "tasks": self.tasks,
            "batch_size": self.batch_size,
            "num_fewshot": self.fewshot,
        }
        if self.max_batch_size is not None:
            base["max_batch_size"] = self.max_batch_size
        base.update(self.lm_eval_kwargs)
        return base
