from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from omegaconf import MISSING

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


@dataclass
class ParamRangeConfig:
    min_params_m: float = MISSING
    max_params_m: float = MISSING


@dataclass
class WandbConfig:
    project: str = MISSING
    entity: Optional[str] = None
    group: Optional[str] = None
    name: Optional[str] = MISSING
    id: Optional[str] = None


@dataclass
class EvaluationConfig:
    task_packs: List[str] = field(default_factory=list)
    output_dir: str = MISSING
    overrides: Dict[str, Any] = field(default_factory=dict)
    wandb: Optional[WandbConfig] = MISSING


@dataclass
class ModelEvalConfig:
    checkpoint: str = MISSING
    adapter: str = "hf"
    dtype: str = "bfloat16"
    device: str = "cuda:0"


@dataclass
class ModelConfig:
    architecture: str = MISSING
    config: Optional[Dict[str, Any]] = None
    init_checkpoint: Optional[str] = None
    param_validation: Optional[ParamRangeConfig] = None

    def __post_init__(self):
        if self.config is None and self.init_checkpoint is None:
            raise ValueError(
                "Either `config` or `init_checkpoint` must be provided inside `model`."
            )


@dataclass
class DataConfig:
    path: str = MISSING
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: Optional[str] = "test"


@dataclass
class TokenizerConfig:
    path: str = MISSING


class TemplateChoice(str, Enum):
    CYCLE = "cycle"
    RANDOM = "random"
    ALL = "all"


@dataclass
class TemplateRef:
    id: str = MISSING
    weight: float = 1.0


@dataclass
class FinetuneDatasetConfig:
    hf_name: str = MISSING
    subset: Optional[str] = None
    text_columns: List[str] = field(default_factory=list)
    label_column: str = MISSING
    splits: Dict[str, str] = field(default_factory=dict)
    templates: List[TemplateRef] = field(default_factory=list)
    template_choice: TemplateChoice = TemplateChoice.CYCLE
    max_seq_length: int = MISSING


@dataclass
class PipelineConfig:
    base_checkpoint: str = MISSING
    languages: List[str] = field(default_factory=list)
    task_name: str = MISSING
    finetune_base_cfg: str = MISSING
    eval_stub_cfg: str = MISSING
    slurm_array: bool = False


@dataclass
class TemplateConfig:
    prompt: str = MISSING
    label_mapping: Dict[Union[int, str], str] = field(default_factory=dict)


@dataclass
class PeftConfig:
    method: str = "qlora"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    mode: RunMode
    wandb: WandbConfig
    model: Optional[ModelConfig] = None
    data: Optional[DataConfig] = None
    tokenizer: Optional[TokenizerConfig] = None
    training: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    eval_model: Optional[ModelEvalConfig] = None
    dataset: Optional[FinetuneDatasetConfig] = None
    peft: Optional[PeftConfig] = None
    template: Optional[TemplateConfig] = None
    pipeline: Optional[PipelineConfig] = None
