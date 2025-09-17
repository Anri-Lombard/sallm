from __future__ import annotations

from typing import Any

from datasets import Dataset

from sallm.config import FinetuneDatasetConfig
from sallm.data.formatters.base import TaskFormatter
from sallm.templates.registry import TemplateSpec


class POSTaggingFormatter(TaskFormatter):
    def __init__(self, dataset: Dataset) -> None:
        upos_feature = dataset.features.get("upos") if dataset.features else None
        names = None
        if hasattr(upos_feature, "feature") and hasattr(upos_feature.feature, "names"):
            names = list(upos_feature.feature.names)
        self.upos_class_names = names

    def format(
        self,
        example: dict[str, Any],
        template: TemplateSpec | None,
        config: FinetuneDatasetConfig,
    ) -> list[dict[str, str]]:
        tokens = example["tokens"]
        raw_upos = example["upos"]
        if raw_upos and isinstance(raw_upos[0], int) and self.upos_class_names:
            tags = [self.upos_class_names[i] for i in raw_upos]
        else:
            tags = [str(tag) for tag in raw_upos]
        if len(tokens) != len(tags):
            raise ValueError(
                f"Mismatch between tokens ({len(tokens)}) and upos tags ({len(tags)})"
            )
        tuple_list = [
            f"({repr(token)}, {repr(tag)})"
            for token, tag in zip(tokens, tags, strict=False)
        ]
        tuple_repr = "[" + ", ".join(tuple_list) + "]"
        if template is not None:
            user_prompt = template.prompt.format(
                tokens="[" + ", ".join(repr(t) for t in tokens) + "]"
            )
        else:
            tokens_repr = "[" + ", ".join(repr(t) for t in tokens) + "]"
            user_prompt = (
                "Please provide UPOS tags for each token as a list of (token, TAG) "
                "tuples.\nSentence: "
                + tokens_repr
                + "\nOutput: "
            )
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": tuple_repr},
        ]
