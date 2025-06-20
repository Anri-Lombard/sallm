import os
from typing import Any, Dict, Optional, Union

import regex as re
import yaml
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass, field

# from transformers import TrainingArguments

from sallm.models.registry import MODEL_CONFIG_REGISTRY
from sallm.utils import RunMode


@dataclass
class ScriptArguments:
    config_path: str = field(
        metadata={"help": "Path to the main YAML experiment config file."}
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb run ID to resume a specific crashed trial."},
    )


class ParamRangeConfig(BaseModel):
    min_params_m: float = Field(..., description="Minimum parameters in millions.")
    max_params_m: float = Field(..., description="Maximum parameters in millions.")


class WandbConfig(BaseModel):
    project: str
    entity: Optional[str] = None
    group: Optional[str] = None
    name: Optional[str] = None
    id: Optional[str] = None


class ModelConfig(BaseModel):
    architecture: str
    config: Dict[str, Any]
    param_validation: Optional[ParamRangeConfig] = None

    @field_validator("architecture")
    def validate_architecture(cls, v: str) -> str:
        """Ensure the requested architecture is available in the registry."""
        if v not in MODEL_CONFIG_REGISTRY:
            raise ValueError(
                f"Unsupported architecture '{v}'. "
                f"Available options are: {list(MODEL_CONFIG_REGISTRY.keys())}"
            )
        return v


class DataConfig(BaseModel):
    path: str
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: Optional[str] = "test"


class TokenizerConfig(BaseModel):
    path: str


class ExperimentConfig(BaseModel):
    mode: RunMode
    wandb: WandbConfig
    model: ModelConfig
    data: DataConfig
    tokenizer: TokenizerConfig
    training: Dict[str, Any]


# TODO load automatically
def load_experiment_config(path: str) -> ExperimentConfig:
    """Loads a YAML config file, expanding any environment variables."""
    path_matcher = re.compile(r"\$\{([^}]+)\}")

    def path_constructor(loader, node):
        value = loader.construct_scalar(node)
        match = path_matcher.search(value)
        if match:
            env_var = match.group(1)
            return value.replace(f"${{{env_var}}}", os.environ.get(env_var, ""))
        return value

    class EnvVarLoader(yaml.SafeLoader):
        pass

    EnvVarLoader.add_constructor("tag:yaml.org,2002:str", path_constructor)

    with open(path, "r") as f:
        config_dict = yaml.load(f, Loader=EnvVarLoader)

    return ExperimentConfig(**config_dict)
