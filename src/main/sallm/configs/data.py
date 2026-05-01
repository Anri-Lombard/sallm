from __future__ import annotations

from dataclasses import dataclass


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
