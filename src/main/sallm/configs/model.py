from __future__ import annotations

from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ModelConfig:
    architecture: str = MISSING
    init_checkpoint: str = MISSING
    attn_implementation: str | None = None


@dataclass
class TokenizerConfig:
    path: str = MISSING
