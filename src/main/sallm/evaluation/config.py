from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

__all__ = ["TaskPack"]


class TaskPack(BaseModel):
    name: str
    tasks: List[str]
    fewshot: int = 0
    batch_size: int = 8
    apply_chat_template: bool = True
    lm_eval_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def to_lm_eval_kwargs(self) -> Dict[str, Any]:
        base = {
            "tasks": self.tasks,
            "batch_size": self.batch_size,
            "num_fewshot": self.fewshot,
        }
        base.update(self.lm_eval_kwargs)
        return base
