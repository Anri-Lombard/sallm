from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from sallm.configs.evaluation import (
    DecodingConfig,
    EvaluationConfig,
    ModelEvalConfig,
)
from sallm.configs.finetune import FinetuneDatasetConfig
from sallm.configs.hub import HubConfig, WandbConfig
from sallm.configs.model import ModelConfig, TokenizerConfig


class RunMode(str, Enum):
    FINETUNE = "finetune"
    EVALUATE = "evaluate"


@dataclass
class ExperimentConfig:
    mode: RunMode
    wandb: WandbConfig
    model: ModelConfig | None = None
    tokenizer: TokenizerConfig | None = None
    training: dict[str, Any] | None = None
    evaluation: EvaluationConfig | None = None
    eval_model: ModelEvalConfig | None = None
    dataset: FinetuneDatasetConfig | None = None
    generation_decoding: DecodingConfig | None = None
    hub: HubConfig | None = None
