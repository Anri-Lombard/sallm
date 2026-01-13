from __future__ import annotations

from datasets import (
    Dataset,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)

from sallm.config import FinetuneDatasetConfig
from sallm.data.loaders.base import load_split_with_fallback
from sallm.data.transforms.language_filter import (
    filter_by_language,
    filter_by_single_language,
)


def load_hf_dataset(ds_cfg: FinetuneDatasetConfig) -> tuple[Dataset, Dataset, bool]:
    """Load train/val datasets from HuggingFace with language handling.

    Args:
        ds_cfg: Dataset configuration

    Returns:
        Tuple of (train_ds, val_ds, needs_lang_filter)
    """
    try:
        available_configs = get_dataset_config_names(ds_cfg.hf_name)
    except TypeError:
        available_configs = get_dataset_config_names(ds_cfg.hf_name)
    load_name = ds_cfg.subset
    filter_after_load = False
    lang_list_cfg = list(ds_cfg.languages or [])
    splits = ds_cfg.splits

    if lang_list_cfg:
        can_multi_load = all(
            lang_code in available_configs for lang_code in lang_list_cfg
        )
        if can_multi_load:
            train_parts: list[Dataset] = []
            val_parts: list[Dataset] = []
            for lang_code in lang_list_cfg:
                tr = load_dataset(
                    ds_cfg.hf_name,
                    name=lang_code,
                    split=splits["train"],
                )
                va = load_split_with_fallback(ds_cfg.hf_name, lang_code, splits["val"])
                if "lang" not in tr.column_names:
                    tr = tr.add_column("lang", [lang_code] * len(tr))
                if "lang" not in va.column_names:
                    va = va.add_column("lang", [lang_code] * len(va))
                train_parts.append(tr)
                val_parts.append(va)
            return (
                concatenate_datasets(train_parts),
                concatenate_datasets(val_parts),
                False,
            )
        else:
            load_name = None
            filter_after_load = True
    else:
        if ds_cfg.subset not in available_configs:
            load_name = None
            filter_after_load = True

    train_raw = load_dataset(
        ds_cfg.hf_name,
        name=load_name,
        split=splits["train"],
    )
    val_raw = load_split_with_fallback(ds_cfg.hf_name, load_name, splits["val"])

    return train_raw, val_raw, filter_after_load


def apply_language_filters(
    train_ds: Dataset,
    val_ds: Dataset,
    ds_cfg: FinetuneDatasetConfig,
    filter_after_load: bool,
) -> tuple[Dataset, Dataset]:
    """Apply language filtering to datasets.

    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        ds_cfg: Dataset configuration
        filter_after_load: Whether to filter by subset language

    Returns:
        Tuple of (filtered_train_ds, filtered_val_ds)
    """
    lang_tag = ds_cfg.subset
    lang_list = set(ds_cfg.languages or [])

    if filter_after_load and lang_tag:
        train_ds = filter_by_single_language(train_ds, lang_tag)
        val_ds = filter_by_single_language(val_ds, lang_tag)

    if lang_list:
        has_lang_col = any(
            col in train_ds.column_names
            for col in ("lang", "language_code", "language")
        )
        if has_lang_col:
            train_ds = filter_by_language(train_ds, lang_list)
            val_ds = filter_by_language(val_ds, lang_list)

    return train_ds, val_ds
