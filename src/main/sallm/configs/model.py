from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING


@dataclass
class ParamRangeConfig:
    min_params_m: float = MISSING
    max_params_m: float = MISSING


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
class TokenizerConfig:
    path: str = MISSING
