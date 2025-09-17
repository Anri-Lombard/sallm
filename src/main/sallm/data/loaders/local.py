from __future__ import annotations

from datasets import DatasetDict, load_from_disk

from sallm.config import DataConfig


class LocalDatasetLoader:
    def load(self, config: DataConfig) -> DatasetDict:
        dataset = load_from_disk(config.path)
        if not isinstance(dataset, DatasetDict):
            raise TypeError(
                f"Expected DatasetDict at {config.path}, found {type(dataset)}"
            )
        return dataset
