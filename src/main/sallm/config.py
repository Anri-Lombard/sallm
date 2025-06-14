# [File: src/main/sallm/config.py]
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field


class ParamRangeConfig(BaseModel):
    """Specifies an acceptable range for model parameters in millions."""

    min_params_m: float = Field(..., description="Minimum parameters in millions.")
    max_params_m: float = Field(..., description="Maximum parameters in millions.")


class WandbConfig(BaseModel):
    """Configuration for Weights & Biases."""

    project: str
    entity: Optional[str] = None
    group: Optional[str] = None
    name: Optional[str] = None


class ModelConfig(BaseModel):
    architecture: str
    config: Dict[str, Any]
    param_validation: Optional[ParamRangeConfig] = None


class DataConfig(BaseModel):
    """Configuration for loading datasets."""

    path: str
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: Optional[str] = "test"


class TokenizerConfig(BaseModel):
    path: str


class TrainingConfig(BaseModel):
    """Configuration for Hugging Face TrainingArguments."""

    output_dir: str
    learning_rate: float
    num_train_epochs: int
    weight_decay: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    warmup_ratio: float
    lr_scheduler_type: str
    bf16: bool
    logging_dir: str
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_strategy: str
    save_steps: int
    save_total_limit: int
    report_to: str

    class Config:
        extra = "allow"


class ExperimentConfig(BaseModel):
    wandb: WandbConfig
    model: ModelConfig
    data: DataConfig
    tokenizer: TokenizerConfig
    training: TrainingConfig


def load_experiment_config(path: str) -> ExperimentConfig:
    """Loads a YAML config file and parses it into an ExperimentConfig object."""
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return ExperimentConfig(**config_dict)
