from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from omegaconf import MISSING
from peft import PeftConfig as HFPEFTConfig

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
    generation_tasks: list["GenerationEvalTaskConfig"] = field(default_factory=list)


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
    merge_lora: bool | None = None

    def __post_init__(self) -> None:
        adapter_path = None
        if self.checkpoint is None:
            raise ValueError("eval_model.checkpoint must be provided and non-empty")
        checkpoint_str = str(self.checkpoint).rstrip("/")
        if checkpoint_str == "":
            raise ValueError("eval_model.checkpoint must be provided and non-empty")
        self.checkpoint = checkpoint_str
        checkpoint_path = Path(self.checkpoint)

        if checkpoint_path.exists():
            try:
                checkpoint_path = checkpoint_path.resolve()
                self.checkpoint = str(checkpoint_path)
            except Exception:
                pass

        if not checkpoint_path.exists():
            resolved = self._resolve_missing_checkpoint(checkpoint_path)
            if resolved is not None:
                checkpoint_path = resolved
                self.checkpoint = str(checkpoint_path)
                if self.merge_lora is False:
                    self.merge_lora = True

        if (
            not self.peft_adapter
            and checkpoint_path.exists()
            and checkpoint_path.is_dir()
        ):
            adapter_config = checkpoint_path / "adapter_config.json"
            adapter_weights = [
                checkpoint_path / "adapter_model.bin",
                checkpoint_path / "adapter_model.safetensors",
            ]
            full_model_weights = [
                checkpoint_path / "pytorch_model.bin",
                checkpoint_path / "model.safetensors",
            ]
            has_adapter = adapter_config.exists() and any(
                path.exists() for path in adapter_weights
            )
            has_full_model = any(path.exists() for path in full_model_weights)
            if has_adapter and not has_full_model:
                peft_config = HFPEFTConfig.from_pretrained(checkpoint_path)
                base_model = peft_config.base_model_name_or_path
                if not base_model:
                    raise ValueError(
                        "adapter_config.json is missing the `base_model_name_or_path`."
                    )
                base_model_str = str(base_model)
                base_model_path = Path(base_model_str)
                if not base_model_path.is_absolute() and not base_model_path.exists():
                    candidate = (checkpoint_path / base_model_path).resolve()
                    if candidate.exists():
                        base_model_str = str(candidate)
                self.peft_adapter = str(checkpoint_path)
                self.checkpoint = base_model_str
                adapter_path = checkpoint_path

        if self.peft_adapter:
            adapter_path = Path(self.peft_adapter)
            if not adapter_path.exists():
                raise ValueError(
                    f"PEFT adapter path '{self.peft_adapter}' does not exist."
                )

        if self.merge_lora is None and self.peft_adapter:
            self.merge_lora = True

    def _resolve_missing_checkpoint(self, checkpoint_path: Path) -> Path | None:
        if checkpoint_path.name == "final_merged_model":
            candidate = checkpoint_path.with_name("final_adapter")
            if candidate.exists():
                adapter_config = candidate / "adapter_config.json"
                adapter_weights = [
                    candidate / "adapter_model.bin",
                    candidate / "adapter_model.safetensors",
                ]
                has_weights = any(path.exists() for path in adapter_weights)
                if adapter_config.exists() and has_weights:
                    return candidate
        return None


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
class GenerationEvalTaskConfig:
    id: str = MISSING
    dataset: FinetuneDatasetConfig = MISSING
    split: str = MISSING
    max_new_tokens: int = MISSING
    max_samples_per_lang: int | None = None
    sample_seed: int | None = None


@dataclass
class GeneratedExample:
    prompt_messages: list[dict[str, str]]
    prompt_text: str
    prediction: str
    reference: str


@dataclass
class LanguageEvalResult:
    key: str
    metrics: dict[str, float]
    examples: list[GeneratedExample] = field(default_factory=list)


@dataclass
class GenerationEvalResult:
    metrics: dict[str, float]
    per_language: dict[str, LanguageEvalResult] = field(default_factory=dict)


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
