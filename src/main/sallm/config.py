from enum import Enum
import os
from typing import Any, Dict, List, Optional, Union

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


class EvaluationConfig(BaseModel):
    task_packs: list[str]
    output_dir: str
    overrides: dict[str, dict] = {}
    wandb: Optional[WandbConfig] = None


class ModelEvalConfig(BaseModel):
    checkpoint: str
    adapter: str = "hf"
    dtype: str = "bfloat16"
    device: str = "cuda:0"


class ModelConfig(BaseModel):
    architecture: str
    config: Dict[str, Any]
    param_validation: Optional[ParamRangeConfig] = None

    @field_validator("architecture")
    def validate_architecture(cls, v: str) -> str:
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


class TemplateChoice(str, Enum):
    CYCLE = "cycle"
    RANDOM = "random"
    ALL = "all"
    SINGLE = "single"  # TODO‑remove after benchmark sanity‑checks


class TemplateRef(BaseModel):
    id: str
    weight: float = 1.0


class FinetuneDatasetConfig(BaseModel):
    hf_name: str
    subset: Optional[str] = None
    text_columns: List[str]
    label_column: str
    splits: Dict[str, str]
    templates: List[TemplateRef]
    template_choice: TemplateChoice = TemplateChoice.CYCLE


class PipelineConfig(BaseModel):
    base_checkpoint: str
    languages: List[str]
    task_name: str
    finetune_base_cfg: str
    eval_stub_cfg: str
    slurm_array: bool = False


class TemplateConfig(BaseModel):
    prompt: str
    label_mapping: Dict[Union[int, str], str]


class PeftConfig(BaseModel):
    method: str = "qlora"
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    mode: RunMode
    wandb: "WandbConfig"
    model: Optional["ModelConfig"] = None
    data: Optional["DataConfig"] = None
    tokenizer: Optional["TokenizerConfig"] = None
    training: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    eval_model: Optional["ModelEvalConfig"] = None
    dataset: Optional[FinetuneDatasetConfig] = None
    peft: Optional["PeftConfig"] = None
    template: Optional["TemplateConfig"] = None
    pipeline: Optional[PipelineConfig] = None


# TODO load automatically
def load_experiment_config(path: str) -> ExperimentConfig:
    env_var_pattern = re.compile(r"\$\{([^}]+)\}")

    class EnvVarLoader(yaml.SafeLoader):
        pass

    def _expand(loader, node):
        value = loader.construct_scalar(node)
        match = env_var_pattern.search(value)
        if match:
            env = match.group(1)
            return value.replace(f"${{{env}}}", os.environ.get(env, ""))
        return value

    EnvVarLoader.add_constructor("tag:yaml.org,2002:str", _expand)

    with open(path, "r") as fp:
        cfg_dict = yaml.load(fp, Loader=EnvVarLoader) or {}

    return ExperimentConfig(**cfg_dict)
