from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from omegaconf import MISSING

from sallm.utils import RunMode


@dataclass
class ScriptArguments:
    config_path: str = field(
        metadata={"help": "Path to the main YAML experiment config file."}
    )
    wandb_run_id: str | None = field(
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
    entity: str | None = None
    group: str | None = None
    name: str | None = MISSING
    id: str | None = None


@dataclass
class EvaluationConfig:
    task_packs: list[str] = field(default_factory=list)
    output_dir: str = MISSING
    overrides: dict[str, Any] = field(default_factory=dict)
    wandb: WandbConfig | None = MISSING


@dataclass
class PeftLoadConfig:
    path: str = MISSING
    merge: bool = True


@dataclass
class ModelEvalConfig:
    checkpoint: str = MISSING
    adapter: str = "hf"
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    peft_adapter: str | None = None
    merge_lora: bool = False

    def __post_init__(self) -> None:
        from pathlib import Path

        if self.peft_adapter and not Path(self.peft_adapter).exists():
            raise ValueError(f"PEFT adapter path '{self.peft_adapter}' does not exist.")


@dataclass
class ModelConfig:
    architecture: str = MISSING
    config: dict[str, Any] | None = None
    init_checkpoint: str | None = None
    param_validation: ParamRangeConfig | None = None

    def __post_init__(self) -> None:
        if self.config is None and self.init_checkpoint is None:
            raise ValueError(
                "Either `config` or `init_checkpoint` must be provided inside `model`."
            )


@dataclass
class DataConfig:
    path: str = MISSING
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str | None = "test"


@dataclass
class TokenizerConfig:
    path: str = MISSING


class TemplateChoice(str, Enum):
    CYCLE = "cycle"
    RANDOM = "random"
    ALL = "all"


class TaskType(str, Enum):
    CAUSAL_LM = "causal_lm"
    CLASSIFICATION = "classification"


# TODO choose this or tasktype?
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
    label_column: str | None = "label"
    max_seq_length: int = MISSING
    packing: bool = MISSING
    assistant_only_loss: bool = MISSING

    def __post_init__(self) -> None:
        if self.template_choice == TemplateChoice.ALL and (
            self.templates is None or len(self.templates) == 0
        ):
            raise ValueError(
                "dataset.template_choice is 'ALL' but no templates were provided. "
                "Add at least one entry under dataset.templates."
            )


@dataclass
class PipelineConfig:
    base_checkpoint: str = MISSING
    languages: list[str] = field(default_factory=list)
    task_name: str = MISSING
    finetune_base_cfg: str = MISSING
    eval_stub_cfg: str = MISSING
    slurm_array: bool = False


@dataclass
class TemplateConfig:
    prompt: str = MISSING
    label_mapping: dict[int | str, str] = field(default_factory=dict)


@dataclass
class PeftConfig:
    method: str = "qlora"
    kwargs: dict[str, Any] = field(default_factory=dict)


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
    pipeline: PipelineConfig | None = None
