from __future__ import annotations

from typing import Any

from sallm.config import FinetuneDatasetConfig
from sallm.data.formatters.base import TaskFormatter
from sallm.data.utils import reconstruct_entities_from_iob
from sallm.templates.registry import TemplateSpec


class NamedEntityFormatter(TaskFormatter):
    def validate_template(
        self, template: TemplateSpec | None, config: FinetuneDatasetConfig
    ) -> None:
        if template is None or not template.ner_tags:
            raise ValueError("NER templates require a ner_tags list")

    def format(
        self,
        example: dict[str, Any],
        template: TemplateSpec | None,
        config: FinetuneDatasetConfig,
    ) -> list[dict[str, str]]:
        if template is None or not template.ner_tags:
            raise ValueError("NER formatting requires a template with ner_tags")
        text_input = " ".join(example["tokens"])
        user_prompt = template.prompt.format(text=text_input)
        entities = reconstruct_entities_from_iob(
            example["tokens"], example["ner_tags"], template.ner_tags
        )
        assistant_response = " $$ ".join(entities)
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
