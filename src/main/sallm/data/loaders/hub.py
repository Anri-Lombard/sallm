from __future__ import annotations

from typing import Any, Callable

from datasets import DatasetDict, get_dataset_config_names, load_dataset

from sallm.config import FinetuneDatasetConfig
from sallm.data.loaders.base import DatasetLoader


class HubDatasetLoader(DatasetLoader):
    def load(self, config: FinetuneDatasetConfig) -> DatasetDict:
        splits = config.splits
        train_split = splits.get("train")
        val_split = splits.get("val")
        if not train_split or not val_split:
            raise ValueError("dataset.splits must define 'train' and 'val'")
        subset = config.subset
        load_name = subset
        filter_after_load = False
        available_configs = get_dataset_config_names(
            config.hf_name, trust_remote_code=True
        )
        if subset and subset not in available_configs:
            load_name = None
            filter_after_load = True
        dataset_dict = DatasetDict()
        dataset_dict["train"] = load_dataset(
            config.hf_name,
            name=load_name,
            split=train_split,
            trust_remote_code=True,
        )
        dataset_dict["validation"] = load_dataset(
            config.hf_name,
            name=load_name,
            split=val_split,
            trust_remote_code=True,
        )
        test_split = splits.get("test")
        if test_split:
            dataset_dict["test"] = load_dataset(
                config.hf_name,
                name=load_name,
                split=test_split,
                trust_remote_code=True,
            )
        if filter_after_load and subset:
            matcher = _subset_matcher(subset)
            for key in list(dataset_dict.keys()):
                dataset_dict[key] = dataset_dict[key].filter(matcher)
        languages = set(config.languages or [])
        if languages:
            matcher = _language_matcher(languages)
            for key in list(dataset_dict.keys()):
                dataset_dict[key] = dataset_dict[key].filter(matcher)
        return dataset_dict


def _subset_matcher(subset: str) -> Callable[[dict[str, Any]], bool]:
    def _matches(example: dict[str, Any]) -> bool:
        value = (
            example.get("lang")
            or example.get("language")
            or example.get("language_code")
        )
        return value == subset

    return _matches


def _language_matcher(languages: set[str]) -> Callable[[dict[str, Any]], bool]:
    def _matches(example: dict[str, Any]) -> bool:
        value = (
            example.get("lang")
            or example.get("language_code")
            or example.get("language")
        )
        return value in languages if value is not None else False

    return _matches
