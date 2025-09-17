from __future__ import annotations

from typing import Any

from datasets import Dataset

from sallm.config import FinetuneDatasetConfig
from sallm.templates.registry import TemplateSpec


class TaskFormatter:
    def validate_dataset(
        self, dataset: Dataset, config: FinetuneDatasetConfig
    ) -> None:
        return None

    def validate_template(
        self, template: TemplateSpec | None, config: FinetuneDatasetConfig
    ) -> None:
        return None

    def format(
        self,
        example: dict[str, Any],
        template: TemplateSpec | None,
        config: FinetuneDatasetConfig,
    ) -> list[dict[str, str]]:
        raise NotImplementedError
