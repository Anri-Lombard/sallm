from __future__ import annotations

from urllib.request import urlopen

from datasets import (
    Dataset,
    concatenate_datasets,
    load_dataset,
)

from sallm.config import FinetuneDatasetConfig
from sallm.data.loaders.base import VALIDATION_ALIASES, load_split_with_fallback
from sallm.data.transforms.language_filter import (
    filter_by_language,
    filter_by_single_language,
)

PARQUET_REVISION = "refs/convert/parquet"
MASAKHAPOS_DATASET = "masakhane/masakhapos"
MASAKHAPOS_BASE_URL = "https://github.com/masakhane-io/masakhane-pos/raw/main/data"


def _load_train_val_with_revision_fallback(
    hf_name: str,
    name: str | None,
    train_split: str,
    val_split: str,
) -> tuple[Dataset, Dataset]:
    """Load train/val with normal revision first, then parquet fallback."""
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


def _masakhapos_split_candidates(split: str) -> list[str]:
    """Map requested split names to likely TSV filenames in masakhapos."""
    s = split.lower()
    if s == "train":
        return ["train.txt"]
    if s in VALIDATION_ALIASES:
        return ["dev.txt", "validation.txt", "val.txt", "valid.txt", "test.txt"]
    if s == "test":
        return ["test.txt", "dev.txt"]
    return [f"{split}.txt", "dev.txt", "test.txt"]


def _parse_masakhapos_conll(content: str, lang_code: str) -> Dataset:
    """Parse MasakhaPOS CoNLL-style text into a Dataset."""
    ids: list[str] = []
    tokens_batch: list[list[str]] = []
    upos_batch: list[list[str]] = []
    langs: list[str] = []

    tokens: list[str] = []
    upos_tags: list[str] = []
    guid = 0

    for line in content.splitlines():
        if line.startswith("-DOCSTART-"):
            continue
        if not line.strip():
            if tokens:
                ids.append(str(guid))
                tokens_batch.append(tokens)
                upos_batch.append(upos_tags)
                langs.append(lang_code)
                guid += 1
                tokens = []
                upos_tags = []
            continue

        splits = line.strip().split()
        if not splits:
            continue
        tokens.append(splits[0])
        upos_tags.append(splits[-1])

    if tokens:
        ids.append(str(guid))
        tokens_batch.append(tokens)
        upos_batch.append(upos_tags)
        langs.append(lang_code)

    return Dataset.from_dict(
        {
            "id": ids,
            "tokens": tokens_batch,
            "upos": upos_batch,
            "lang": langs,
        }
    )


def _load_masakhapos_split(lang_code: str, split: str) -> Dataset:
    """Load a masakhapos split directly from dataset files (no dataset script)."""
    last_err: Exception | None = None
    for filename in _masakhapos_split_candidates(split):
        try:
            url = f"{MASAKHAPOS_BASE_URL}/{lang_code}/{filename}"
            with urlopen(url, timeout=30) as response:
                content = response.read().decode("utf-8")
            return _parse_masakhapos_conll(content, lang_code)
        except Exception as err:  # noqa: BLE001 - try alternate filename candidates
            last_err = err

    assert last_err is not None
    raise last_err


def _load_masakhapos_dataset(
    ds_cfg: FinetuneDatasetConfig,
) -> tuple[Dataset, Dataset, bool]:
    """Load masakhapos with explicit language files from HF Hub."""
    splits = ds_cfg.splits
    lang_list_cfg = list(ds_cfg.languages or [])
    if not lang_list_cfg:
        if ds_cfg.subset:
            lang_list_cfg = [ds_cfg.subset]
        else:
            raise ValueError(
                "masakhane/masakhapos requires dataset.subset or dataset.languages."
            )

    train_parts: list[Dataset] = []
    val_parts: list[Dataset] = []
    for lang_code in lang_list_cfg:
        train_parts.append(_load_masakhapos_split(lang_code, splits["train"]))
        val_parts.append(_load_masakhapos_split(lang_code, splits["val"]))

    if len(train_parts) == 1:
        return train_parts[0], val_parts[0], False

    return concatenate_datasets(train_parts), concatenate_datasets(val_parts), False


def load_hf_dataset(ds_cfg: FinetuneDatasetConfig) -> tuple[Dataset, Dataset, bool]:
    """Load train/val datasets from HuggingFace with language handling.

    Args:
        ds_cfg: Dataset configuration

    Returns:
        Tuple of (train_ds, val_ds, needs_lang_filter)
    """
    if ds_cfg.hf_name == MASAKHAPOS_DATASET:
        return _load_masakhapos_dataset(ds_cfg)

    load_name = ds_cfg.subset
    lang_list_cfg = list(ds_cfg.languages or [])
    splits = ds_cfg.splits

    if lang_list_cfg:
        # First choice: explicit per-language config loading.
        train_parts: list[Dataset] = []
        val_parts: list[Dataset] = []
        try:
            for lang_code in lang_list_cfg:
                tr, va = _load_train_val_with_revision_fallback(
                    ds_cfg.hf_name,
                    lang_code,
                    splits["train"],
                    splits["val"],
                )
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
        except Exception:
            # Fall back to loading one dataset and filtering by language later.
            pass

    # Explicit subset path: keep trying with the requested config first.
    if load_name is not None:
        try:
            return (
                *_load_train_val_with_revision_fallback(
                    ds_cfg.hf_name,
                    load_name,
                    splits["train"],
                    splits["val"],
                ),
                False,
            )
        except Exception:
            # If config load fails, try loading without config and filter by subset.
            train_raw, val_raw = _load_train_val_with_revision_fallback(
                ds_cfg.hf_name,
                None,
                splits["train"],
                splits["val"],
            )
            return train_raw, val_raw, True

    # No subset specified; load default and let caller optionally language-filter.
    train_raw, val_raw = _load_train_val_with_revision_fallback(
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
