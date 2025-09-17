from __future__ import annotations

from typing import Any

from sallm.config import FinetuneDatasetConfig
from sallm.data.formatters.base import TaskFormatter
from sallm.data.utils import extract_instruction_pair
from sallm.templates.registry import TemplateSpec


class InstructionFormatter(TaskFormatter):
    def format(
        self,
        example: dict[str, Any],
        template: TemplateSpec | None,
        config: FinetuneDatasetConfig,
    ) -> list[dict[str, str]]:
        user_text, assistant_text = extract_instruction_pair(example)
        return [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
