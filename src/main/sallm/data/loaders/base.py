from __future__ import annotations

from typing import Protocol

from datasets import DatasetDict

from sallm.config import FinetuneDatasetConfig


class DatasetLoader(Protocol):
    def load(self, config: FinetuneDatasetConfig) -> DatasetDict:
        ...
