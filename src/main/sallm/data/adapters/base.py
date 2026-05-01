from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from datasets import Dataset, DatasetDict

from sallm.config import FinetuneDatasetConfig


@dataclass(frozen=True)
class RawDatasetSplits:
    train: Dataset
    validation: Dataset
    needs_language_filter: bool = False


class DatasetAdapter(Protocol):
    name: str

    def supports(self, ds_cfg: FinetuneDatasetConfig) -> bool: ...

    def load(self, ds_cfg: FinetuneDatasetConfig) -> RawDatasetSplits: ...


def extract_train_validation_splits(ds: Dataset | DatasetDict) -> RawDatasetSplits:
    if isinstance(ds, DatasetDict):
        train = ds["train"] if "train" in ds else ds[next(iter(ds.keys()))]

        if "validation" in ds:
            validation = ds["validation"]
        elif "dev" in ds:
            validation = ds["dev"]
        elif "test" in ds:
            validation = ds["test"]
        else:
            validation = train
        return RawDatasetSplits(train=train, validation=validation)

    return RawDatasetSplits(train=ds, validation=ds)


def required_languages(ds_cfg: FinetuneDatasetConfig, dataset_name: str) -> list[str]:
    lang_list_cfg = list(ds_cfg.languages or [])
    if lang_list_cfg:
        return lang_list_cfg
    if ds_cfg.subset:
        return [ds_cfg.subset]
    raise ValueError(f"{dataset_name} requires dataset.subset or dataset.languages.")
