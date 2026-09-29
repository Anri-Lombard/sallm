from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import MISSING

from sallm.configs.common import to_resolved_dict
from sallm.configs.finetune import FewshotTemplateMode, FinetuneDatasetConfig
from sallm.configs.hub import WandbConfig


@dataclass
class EvaluationConfig:
    task_packs: list[str] = field(default_factory=list)
    task_pack_scope: str = "eval"
    output_dir: str = MISSING
    overrides: dict[str, Any] = field(default_factory=dict)
    wandb: WandbConfig | None = MISSING
    generation_tasks: list[GenerationEvalTaskConfig] = field(default_factory=list)


@dataclass
class ModelEvalConfig:
    # A Hub id or a local model directory; a list tries each path in order. A
    # fine-tuning output directory resolves to its `final_model`.
    checkpoint: Any = MISSING
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    tie_word_embeddings: bool | None = None
    lm_eval_model_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidates = (
            self.checkpoint
            if isinstance(self.checkpoint, list | tuple)
            else [self.checkpoint]
        )
        candidates = [str(c).rstrip("/") for c in candidates if c]
        if not candidates:
            raise ValueError("eval_model.checkpoint must be provided and non-empty")
        for candidate in candidates:
            if _is_hf_hub_id(candidate):
                self.checkpoint = candidate
                return
            path = Path(candidate).expanduser()
            if (path / "final_model").is_dir():
                path = path / "final_model"
            if path.exists():
                self.checkpoint = str(path.resolve())
                return
        raise ValueError(
            f"Checkpoint path not found. Attempted: {', '.join(candidates)}."
        )


def _is_hf_hub_id(value: str) -> bool:
    if value.startswith(("/", ".", "~")):
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(p and not p.startswith(".") for p in parts)


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
    batch_size: int | str | None = "auto:4"
    max_batch_size: int | None = 64

    @classmethod
    def from_any(cls, value: Any | None) -> DecodingConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        data = to_resolved_dict(value, name="decoding config")
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
    prompt_format: str = "chat"


@dataclass
class GeneratedExample:
    prompt_messages: list[dict[str, str]]
    prompt_text: str
    prediction: str
    reference: str
    raw_prediction: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class LanguageEvalResult:
    key: str
    metrics: dict[str, float]
    examples: list[GeneratedExample] = field(default_factory=list)


@dataclass
class GenerationEvalResult:
    metrics: dict[str, float]
    per_language: dict[str, LanguageEvalResult] = field(default_factory=dict)
