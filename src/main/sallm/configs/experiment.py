from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sallm.configs.data import DataConfig
from sallm.configs.evaluation import (
    DecodingConfig,
    EvaluationConfig,
    ModelEvalConfig,
)
from sallm.configs.finetune import FinetuneDatasetConfig, PeftConfig, TemplateConfig
from sallm.configs.hub import HubConfig, WandbConfig
from sallm.configs.model import ModelConfig, TokenizerConfig
from sallm.utils import RunMode


@dataclass
class ExperimentConfig:
    mode: RunMode
    wandb: WandbConfig
    model: ModelConfig | None = None
    data: DataConfig | None = None
    tokenizer: TokenizerConfig | None = None
    training: dict[str, Any] | None = None
    evaluation: EvaluationConfig | None = None
    eval_model: ModelEvalConfig | None = None
    dataset: FinetuneDatasetConfig | None = None
    peft: PeftConfig | None = None
    template: TemplateConfig | None = None
    generation_decoding: DecodingConfig | None = None
    hub: HubConfig | None = None
