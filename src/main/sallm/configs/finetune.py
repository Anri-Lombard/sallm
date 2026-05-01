from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from omegaconf import MISSING


class TemplateChoice(str, Enum):
    CYCLE = "cycle"
    RANDOM = "random"
    ALL = "all"


class FewshotTemplateMode(str, Enum):
    SAME = "same"
    RANDOM = "random"
    CYCLE = "cycle"


class TaskType(str, Enum):
    CAUSAL_LM = "causal_lm"
    CLASSIFICATION = "classification"


class FinetuneTaskType(str, Enum):
    INSTRUCTION = "instruction"
    CLASSIFICATION = "classification"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    POS_TAGGING = "pos_tagging"


@dataclass
class TemplateRef:
    id: str = MISSING
    weight: float = 1.0


@dataclass
class FinetuneDatasetConfig:
    hf_name: str = MISSING
    subset: str | None = None
    languages: list[str] | None = None
    task: FinetuneTaskType | None = None
    splits: dict[str, str] = field(default_factory=dict)
    templates: list[TemplateRef] = field(default_factory=list)
    template_choice: TemplateChoice = TemplateChoice.CYCLE
    eval_template_choice: TemplateChoice | None = None
    label_column: str | None = "label"
    max_seq_length: int = MISSING
    packing: bool = MISSING
    assistant_only_loss: bool = MISSING
    mix_name: str | None = None
    mix_weights: dict[str, float] = field(default_factory=dict)
    mix_temperature: float = 0.0
    mix_epoch_size: int | str | None = None
    mix_min_prob: float | None = None
    mix_max_prob: float | None = None

    def __post_init__(self) -> None:
        if self.template_choice == TemplateChoice.ALL and (
            self.templates is None or len(self.templates) == 0
        ):
            raise ValueError(
                "dataset.template_choice is 'ALL' but no templates were provided. "
                "Add at least one entry under dataset.templates."
            )
        if self.eval_template_choice == TemplateChoice.ALL and (
            self.templates is None or len(self.templates) == 0
        ):
            raise ValueError(
                "dataset.eval_template_choice is 'ALL' but no templates were "
                "provided. Add at least one entry under dataset.templates."
            )

        if self.mix_epoch_size not in (None, "sum") and not isinstance(
            self.mix_epoch_size, int
        ):
            raise ValueError(
                "dataset.mix_epoch_size must be null, 'sum', or a positive integer"
            )

        if isinstance(self.mix_epoch_size, int) and self.mix_epoch_size <= 0:
            raise ValueError("dataset.mix_epoch_size must be a positive integer")

        if self.mix_min_prob is not None and not 0 <= self.mix_min_prob <= 1:
            raise ValueError("dataset.mix_min_prob must lie in [0, 1]")

        if self.mix_max_prob is not None and not 0 <= self.mix_max_prob <= 1:
            raise ValueError("dataset.mix_max_prob must lie in [0, 1]")

        if (
            self.mix_min_prob is not None
            and self.mix_max_prob is not None
            and self.mix_min_prob > self.mix_max_prob
        ):
            raise ValueError("dataset.mix_min_prob cannot exceed mix_max_prob")

        has_mix = bool(self.mix_name) or (
            isinstance(self.hf_name, str) and self.hf_name.startswith("mix:")
        )
        if has_mix and not self.mix_weights:
            raise ValueError(
                "Mixture datasets require dataset.mix_weights to be specified"
            )


@dataclass
class TemplateConfig:
    prompt: str = MISSING
    label_mapping: dict[int | str, str] = field(default_factory=dict)


@dataclass
class PeftConfig:
    method: str = "qlora"
    kwargs: dict[str, Any] = field(default_factory=dict)
