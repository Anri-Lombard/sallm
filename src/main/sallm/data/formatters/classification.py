from __future__ import annotations

from typing import Any

from datasets import Dataset

from sallm.config import FinetuneDatasetConfig
from sallm.data.formatters.base import TaskFormatter
from sallm.templates.registry import TemplateSpec


class ClassificationFormatter(TaskFormatter):
    def validate_dataset(self, dataset: Dataset, config: FinetuneDatasetConfig) -> None:
        column = config.label_column or "label"
        if column not in dataset.column_names:
            raise ValueError(
                f"Classification label column '{column}' not found in dataset"
            )

    def validate_template(
        self, template: TemplateSpec | None, config: FinetuneDatasetConfig
    ) -> None:
        if template is None or not template.label_mapping:
            raise ValueError("Classification templates require a label_mapping")

    def format(
        self,
        example: dict[str, Any],
        template: TemplateSpec | None,
        config: FinetuneDatasetConfig,
    ) -> list[dict[str, str]]:
        if template is None or not template.label_mapping:
            raise ValueError("Classification formatting requires a template")
        column = config.label_column or "label"
        raw_label = example[column]
        label_mapping = template.label_mapping
        numeric_keys = isinstance(next(iter(label_mapping.keys())), int)
        if numeric_keys:
            numeric_mapping = {
                int(str_key): value
                for str_key, value in label_mapping.items()
                if isinstance(str_key, int) or str(str_key).isdigit()
            }
            numeric_key = _coerce_numeric_label(raw_label, numeric_mapping)
            assistant_response = numeric_mapping[numeric_key]
        else:
            str_key = str(raw_label)
            assistant_response = label_mapping[str_key]
        user_prompt = template.prompt.format(**example)
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]


def _coerce_numeric_label(raw_label: Any, label_mapping: dict[int, str]) -> int:
    if isinstance(raw_label, str):
        try:
            key = int(raw_label)
        except ValueError as err:
            lookup = {str(v).lower(): k for k, v in label_mapping.items()}
            mapped_key = lookup.get(raw_label.lower())
            if mapped_key is None:
                raise ValueError(
                    f"Cannot map string label '{raw_label}' to a numeric key"
                ) from err
            key = mapped_key
    else:
        key = int(raw_label)
    if key not in label_mapping and (key - 1) in label_mapping:
        key -= 1
    return key
