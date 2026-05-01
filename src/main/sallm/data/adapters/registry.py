from __future__ import annotations

from datasets import Dataset

from sallm.config import FinetuneDatasetConfig
from sallm.data.adapters.base import DatasetAdapter
from sallm.data.adapters.github import GitHubAfriHGAdapter, GitHubT2XAdapter
from sallm.data.adapters.huggingface import (
    HuggingFaceAdapter,
    apply_language_filters,
)
from sallm.data.adapters.injongointent import InjongoIntentAdapter
from sallm.data.adapters.masakhaner import MasakhaNERAdapter
from sallm.data.adapters.masakhapos import MasakhaPOSAdapter

DATASET_ADAPTERS: tuple[DatasetAdapter, ...] = (
    GitHubT2XAdapter(),
    GitHubAfriHGAdapter(),
    MasakhaPOSAdapter(),
    InjongoIntentAdapter(),
    MasakhaNERAdapter(),
    HuggingFaceAdapter(),
)


def resolve_dataset_adapter(ds_cfg: FinetuneDatasetConfig) -> DatasetAdapter:
    for adapter in DATASET_ADAPTERS:
        if adapter.supports(ds_cfg):
            return adapter
    raise ValueError(f"No dataset adapter found for dataset.hf_name={ds_cfg.hf_name!r}")


def load_raw_dataset(ds_cfg: FinetuneDatasetConfig) -> tuple[Dataset, Dataset]:
    adapter = resolve_dataset_adapter(ds_cfg)
    loaded = adapter.load(ds_cfg)
    if loaded.needs_language_filter:
        return apply_language_filters(
            loaded.train,
            loaded.validation,
            ds_cfg,
            loaded.needs_language_filter,
        )
    return loaded.train, loaded.validation
