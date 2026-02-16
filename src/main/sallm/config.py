from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from omegaconf import MISSING, DictConfig, OmegaConf
from peft import PeftConfig as HFPEFTConfig

from sallm.utils import RunMode


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
    checkpoint: Any = MISSING
    adapter: str = "hf"
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    peft_adapter: str | None = None
    merge_lora: bool | None = None
    tie_word_embeddings: bool | None = None

    def __post_init__(self) -> None:
        adapter_path = None
        hub_checkpoint: str | None = None
        if self.checkpoint is None:
            raise ValueError("eval_model.checkpoint must be provided and non-empty")

        if isinstance(self.checkpoint, list | tuple):
            candidates = [str(p).rstrip("/") for p in self.checkpoint if p is not None]
        else:
            candidates = [str(self.checkpoint).rstrip("/")]

        candidates = [c for c in candidates if c]
        if not candidates:
            raise ValueError("eval_model.checkpoint must be provided and non-empty")

        chosen_path: Path | None = None
        used_adapter_fallback = False
        last_checked: Path | None = None

        def _is_hf_hub_path(path_str: str) -> bool:
            """Check if path looks like a HuggingFace Hub identifier (org/model)."""
            if path_str.startswith(("/", ".", "~")):
                return False
            parts = path_str.split("/")
            return len(parts) == 2 and all(p and not p.startswith(".") for p in parts)

        for cp in candidates:
            if _is_hf_hub_path(cp):
                self.checkpoint = cp
                hub_checkpoint = cp
                break

            checkpoint_path = Path(cp)
            last_checked = checkpoint_path
            if checkpoint_path.exists():
                try:
                    checkpoint_path = checkpoint_path.resolve()
                except Exception:
                    pass
                chosen_path = checkpoint_path
                break
            resolved = self._resolve_missing_checkpoint(checkpoint_path)
            if resolved is not None:
                chosen_path = resolved
                used_adapter_fallback = True
                break

        if chosen_path is None and hub_checkpoint is None:
            alt_hint = None
            if last_checked is not None:
                parent = last_checked.parent
                if last_checked.name == "final_merged_model" and parent.exists():
                    alt_hint = str(parent / "final_adapter")
            attempted = ", ".join(candidates)
            base_msg = f"Checkpoint path not found. Attempted: {attempted}."
            if alt_hint:
                suggestion = (
                    " If this run saved only PEFT adapters, point "
                    "`eval_model.checkpoint` to the adapter directory, e.g. '"
                    f"{alt_hint}'"
                    ", or set `eval_model.peft_adapter` to the adapter path."
                )
            else:
                suggestion = (
                    " If this run saved only PEFT adapters, set "
                    "`eval_model.peft_adapter` to the adapter path and ensure "
                    "`eval_model.checkpoint` refers to the base model used during "
                    "fine-tuning."
                )
            raise ValueError(base_msg + suggestion)

        checkpoint_path: Path | None = None
        if chosen_path is not None:
            self.checkpoint = str(chosen_path)
            checkpoint_path = chosen_path
        elif hub_checkpoint is not None:
            self.checkpoint = hub_checkpoint

        if used_adapter_fallback and self.merge_lora is False:
            self.merge_lora = True

        if (
            not self.peft_adapter
            and checkpoint_path is not None
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
        elif not self.peft_adapter and hub_checkpoint is not None:
            try:
                peft_config = HFPEFTConfig.from_pretrained(hub_checkpoint)
                base_model = peft_config.base_model_name_or_path
            except Exception:
                base_model = None
            if base_model:
                self.peft_adapter = hub_checkpoint
                self.checkpoint = str(base_model)
                adapter_path = Path(hub_checkpoint)

        if self.peft_adapter:
            adapter_path = Path(self.peft_adapter)
            if not adapter_path.exists() and not _is_hf_hub_path(self.peft_adapter):
                raise ValueError(
                    f"PEFT adapter path '{self.peft_adapter}' does not exist."
                )

        if self.merge_lora is None and self.peft_adapter:
            self.merge_lora = True

    def _resolve_missing_checkpoint(self, checkpoint_path: Path) -> Path | None:
        name = checkpoint_path.name
        parent = checkpoint_path.parent

        candidates: list[Path] = []

        if name in {"final_merged_model", "final_adapter", "final_model"}:
            if name != "final_merged_model":
                candidates.append(parent / "final_merged_model")
            if name != "final_adapter":
                candidates.append(parent / "final_adapter")
            if name != "final_model":
                candidates.append(parent / "final_model")
        else:
            candidates.extend(
                [
                    checkpoint_path / "final_merged_model",
                    checkpoint_path / "final_adapter",
                    checkpoint_path / "final_model",
                ]
            )

        for cand in candidates:
            if not cand.exists():
                continue
            if cand.name == "final_adapter":
                adapter_config = cand / "adapter_config.json"
                adapter_weights = [
                    cand / "adapter_model.bin",
                    cand / "adapter_model.safetensors",
                ]
                has_weights = any(path.exists() for path in adapter_weights)
                if adapter_config.exists() and has_weights:
                    return cand
                continue
            return cand

        return None


@dataclass
class ModelConfig:
    architecture: str = MISSING
    config: dict[str, Any] | None = None
    init_checkpoint: str | None = None
    attn_implementation: str | None = None
    param_validation: ParamRangeConfig | None = None

    def __post_init__(self) -> None:
        if self.config is None and self.init_checkpoint is None:
            raise ValueError(
                "Either `config` or `init_checkpoint` must be provided inside `model`."
            )


@dataclass
class DataConfig:
    path: str | None = None
    hf_name: str | None = None
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str | None = "test"

    def __post_init__(self) -> None:
        if not self.path and not self.hf_name:
            raise ValueError(
                "Either `path` or `hf_name` must be provided in data config"
            )
        if self.path and self.hf_name:
            raise ValueError("Provide either `path` or `hf_name`, not both")


@dataclass
class TokenizerConfig:
    path: str = MISSING


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
class DecodingConfig:
    strategy: str = "greedy"
    num_beams: int | None = None
    num_beam_groups: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    typical_p: float | None = None
    length_penalty: float | None = None
    early_stopping: bool | None = None
    no_repeat_ngram_size: int | None = None
    repetition_penalty: float | None = None
    num_return_sequences: int | None = None
    diversity_penalty: float | None = None
    batch_size: int | None = None

    @classmethod
    def from_any(cls, value: Any | None) -> "DecodingConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, DictConfig):
            data = OmegaConf.to_container(value, resolve=True)
        elif isinstance(value, dict):
            data = value
        else:
            raise TypeError(
                f"Unsupported decoding config type {type(value)!r}; "
                "expected mapping or DecodingConfig."
            )
        if not isinstance(data, dict):
            raise TypeError("DecodingConfig expects a mapping after resolution.")
        return cls(**data)

    def to_generate_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        strategy = self.strategy.lower()
        if strategy == "greedy":
            kwargs["do_sample"] = False
            kwargs["num_beams"] = 1
        elif strategy == "beam":
            kwargs["do_sample"] = False
            num_beams = self.num_beams or 5
            if num_beams < 1:
                raise ValueError("Beam search requires num_beams >= 1")
            kwargs["num_beams"] = num_beams
        elif strategy == "sample":
            kwargs["do_sample"] = True
            if self.num_beams:
                if self.num_beams < 1:
                    raise ValueError("Sampling requires num_beams >= 1 when provided")
                kwargs["num_beams"] = self.num_beams
            kwargs["temperature"] = (
                self.temperature if self.temperature is not None else 1.0
            )
        else:
            raise ValueError(f"Unsupported decoding strategy '{self.strategy}'")
        optional: dict[str, Any] = {
            "num_beam_groups": self.num_beam_groups,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "typical_p": self.typical_p,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "repetition_penalty": self.repetition_penalty,
            "num_return_sequences": self.num_return_sequences,
            "diversity_penalty": self.diversity_penalty,
        }
        for key, value in optional.items():
            if value is not None:
                kwargs[key] = value
        return kwargs


@dataclass
class GenerationEvalTaskConfig:
    id: str = MISSING
    dataset: FinetuneDatasetConfig = MISSING
    split: str = MISSING
    max_new_tokens: int = MISSING
    max_samples_per_lang: int | None = None
    sample_seed: int | None = None
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    fewshot: int = 0
    fewshot_split: str = "train"
    fewshot_seed: int | None = None
    fewshot_lang_match: bool = True
    fewshot_template_mode: FewshotTemplateMode = FewshotTemplateMode.SAME
    fewshot_token_budget: int | None = None
    prompt_headroom_tokens: int | None = None
    system_prompt: str | None = None


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
class HubConfig:
    enabled: bool = False
    organization: str = "anrilombard"
    private: bool = True
    push_adapter: bool = True
    push_merged: bool = False
    base_model_id: str = (
        "anrilombard/sallm-mamba-125m"  # HF model ID for base model reference
    )
    collection_slug: str | None = "anrilombard/mzansilm"  # Collection to add models to


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
