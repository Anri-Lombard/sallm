from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from datasets import Dataset

from sallm.config import FinetuneDatasetConfig
from sallm.data.formatters.base import TaskFormatter
from sallm.templates.registry import TemplateSpec


class POSTaggingFormatter(TaskFormatter):
    def __init__(self, dataset: Dataset) -> None:
        self.upos_class_names: list[str] | None = None
        features = getattr(dataset, "features", None)
        upos_feature = (
            features.get("upos") if features and hasattr(features, "get") else None
        )
        feature_meta = getattr(upos_feature, "feature", None)
        names = getattr(feature_meta, "names", None)
        if isinstance(names, Sequence):
            self.upos_class_names = [str(name) for name in names]

    def format(
        self,
        example: dict[str, Any],
        template: TemplateSpec | None,
        config: FinetuneDatasetConfig,
    ) -> list[dict[str, str]]:
        tokens = example["tokens"]
        raw_upos = example["upos"]
        tags: list[str]
        if (
            isinstance(raw_upos, Sequence)
            and raw_upos
            and isinstance(raw_upos[0], int)
            and self.upos_class_names
        ):
            tags = [self.upos_class_names[i] for i in raw_upos]
        elif isinstance(raw_upos, Sequence):
            tags = [str(tag) for tag in raw_upos]
        else:
            raise TypeError("Expected 'upos' to be a sequence of tags")
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
                "tuples.\nSentence: " + tokens_repr + "\nOutput: "
            )
        return [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": tuple_repr},
        ]
