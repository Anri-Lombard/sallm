from __future__ import annotations

from datasets import Dataset, concatenate_datasets, load_dataset

from sallm.config import FinetuneDatasetConfig
from sallm.data.adapters.base import RawDatasetSplits
from sallm.data.loaders.base import load_split_with_fallback
from sallm.data.transforms.language_filter import (
    filter_by_language,
    filter_by_single_language,
)

PARQUET_REVISION = "refs/convert/parquet"


def load_train_val_with_revision_fallback(
    hf_name: str,
    name: str | None,
    train_split: str,
    val_split: str,
) -> tuple[Dataset, Dataset]:
    last_err: Exception | None = None
    for revision in (None, PARQUET_REVISION):
        try:
            train_ds = load_dataset(
                hf_name,
                name=name,
                split=train_split,
                revision=revision,
            )
            val_ds = load_split_with_fallback(
                hf_name,
                name,
                val_split,
                revision,
            )
            return train_ds, val_ds
        except Exception as err:  # noqa: BLE001 - intentionally broad fallback
            last_err = err

    assert last_err is not None
    raise last_err


class HuggingFaceAdapter:
    name = "huggingface"

    def supports(self, ds_cfg: FinetuneDatasetConfig) -> bool:
        hf_name = str(ds_cfg.hf_name)
        return not hf_name.startswith(("github:", "mix:"))

    def load(self, ds_cfg: FinetuneDatasetConfig) -> RawDatasetSplits:
        train, validation, needs_filter = load_huggingface_dataset(ds_cfg)
        return RawDatasetSplits(
            train=train,
            validation=validation,
            needs_language_filter=needs_filter,
        )


def load_huggingface_dataset(
    ds_cfg: FinetuneDatasetConfig,
) -> tuple[Dataset, Dataset, bool]:
    load_name = ds_cfg.subset
    lang_list_cfg = list(ds_cfg.languages or [])
    splits = ds_cfg.splits

    if lang_list_cfg:
        train_parts: list[Dataset] = []
        val_parts: list[Dataset] = []
        try:
            for lang_code in lang_list_cfg:
                train, validation = load_train_val_with_revision_fallback(
                    ds_cfg.hf_name,
                    lang_code,
                    splits["train"],
                    splits["val"],
                )
                if "lang" not in train.column_names:
                    train = train.add_column("lang", [lang_code] * len(train))
                if "lang" not in validation.column_names:
                    validation = validation.add_column(
                        "lang", [lang_code] * len(validation)
                    )
                train_parts.append(train)
                val_parts.append(validation)

            return (
                concatenate_datasets(train_parts),
                concatenate_datasets(val_parts),
                False,
            )
        except Exception:
            # Fall back to loading one dataset and filtering by language later.
            pass

    if load_name is not None:
        try:
            return (
                *load_train_val_with_revision_fallback(
                    ds_cfg.hf_name,
                    load_name,
                    splits["train"],
                    splits["val"],
                ),
                False,
            )
        except Exception:
            train_raw, val_raw = load_train_val_with_revision_fallback(
                ds_cfg.hf_name,
                None,
                splits["train"],
                splits["val"],
            )
            return train_raw, val_raw, True

    train_raw, val_raw = load_train_val_with_revision_fallback(
        ds_cfg.hf_name,
        None,
        splits["train"],
        splits["val"],
    )
    return train_raw, val_raw, bool(lang_list_cfg)


def apply_language_filters(
    train_ds: Dataset,
    val_ds: Dataset,
    ds_cfg: FinetuneDatasetConfig,
    filter_after_load: bool,
) -> tuple[Dataset, Dataset]:
    lang_tag = ds_cfg.subset
    lang_list = set(ds_cfg.languages or [])

    if filter_after_load and lang_tag:
        train_ds = filter_by_single_language(train_ds, lang_tag)
        val_ds = filter_by_single_language(val_ds, lang_tag)

    if lang_list:
        train_has_lang_col = any(
            col in train_ds.column_names
            for col in ("lang", "language_code", "language")
        )
        val_has_lang_col = any(
            col in val_ds.column_names for col in ("lang", "language_code", "language")
        )
        if not train_has_lang_col or not val_has_lang_col:
            missing = []
            if not train_has_lang_col:
                missing.append("train")
            if not val_has_lang_col:
                missing.append("validation")
            raise ValueError(
                "dataset.languages was requested, but the loaded dataset has no "
                f"language column to filter in {', '.join(missing)} split(s). "
                "Use an explicit per-language loader or set dataset.subset to a "
                "valid dataset config."
            )
        before_train = len(train_ds)
        before_val = len(val_ds)
        train_ds = filter_by_language(train_ds, lang_list)
        val_ds = filter_by_language(val_ds, lang_list)
        if len(train_ds) == 0 or len(val_ds) == 0:
            raise ValueError(
                "dataset.languages filtering produced an empty split "
                f"(train {before_train}->{len(train_ds)}, "
                f"val {before_val}->{len(val_ds)})."
            )

    return train_ds, val_ds
