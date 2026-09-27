from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from time import sleep
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from datasets import (
    Dataset,
    concatenate_datasets,
    load_dataset,
)

from sallm.config import FinetuneDatasetConfig
from sallm.data.loaders.base import VALIDATION_ALIASES, load_split_with_fallback
from sallm.data.loaders.injongointent_split import (
    exclude_heldout_texts,
    split_injongointent_rows,
)
from sallm.data.transforms.language_filter import (
    filter_by_language,
    filter_by_single_language,
)

PARQUET_REVISION = "refs/convert/parquet"
DATASET_REVISIONS = {
    "masakhane/masakhanews": "fa3b5fff8a91d187bf0c5900a39c4271d08cf7fe",
    "anrilombard/masakhaner-x-parquet": "6aa65cdbfa22d66e5b4ed176ac525c364cda08d1",
    "Davlan/sib200": "38977a667f6fc264d5c26ec57a01e16db040b358",
}
MASAKHAPOS_DATASET = "masakhane/masakhapos"
MASAKHAPOS_REVISION = "376f4161f0425584d4bd7664122b56fa026926d3"
MASAKHAPOS_BASE_URL = (
    "https://api.github.com/repos/masakhane-io/masakhane-pos/contents/data"
)
INJONGOINTENT_DATASET = "masakhane/InjongoIntent"
INJONGOINTENT_BASE_URL = (
    "https://huggingface.co/datasets/masakhane/InjongoIntent/resolve/"
    "fe4be3882a1614161dfe231ec793197bb74f4b44"
)
MASAKHANER_DATASET = "masakhane/masakhaner2"
MASAKHANER_PARQUET_DATASET = "anrilombard/masakhaner-x-parquet"
MASAKHANER_PARQUET_LANG_DIRS = {
    "tsn": "tn",
    "xho": "xh",
    "zul": "zu",
}


def _load_train_val_with_revision_fallback(
    hf_name: str,
    name: str | None,
    train_split: str,
    val_split: str,
) -> tuple[Dataset, Dataset]:
    """Load train/val with normal revision first, then parquet fallback."""
    last_err: Exception | None = None
    pinned_revision = DATASET_REVISIONS.get(hf_name)
    revisions = (pinned_revision,) if pinned_revision else (None, PARQUET_REVISION)
    for revision in revisions:
        try:
            train_ds = cast(
                Dataset,
                load_dataset(
                    hf_name,
                    name=name,
                    split=train_split,
                    revision=revision,
                ),
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
        return ["dev.txt", "validation.txt", "val.txt", "valid.txt"]
    if s == "test":
        return ["test.txt"]
    return [f"{split}.txt"]


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


def _read_url(url: str | Request) -> bytes:
    """Read a source file, retrying only transient transport failures.

    With SALLM_SOURCE_CACHE_DIR set, commit-pinned URLs (immutable content) are
    read from and written to that directory (file name = sha256 of the URL).
    GitHub's API allows 60 unauthenticated requests per hour per IP, which
    parallel cluster jobs exhaust (HTTP 403 "rate limit exceeded").
    """
    full_url = url.full_url if isinstance(url, Request) else url
    cache_dir = os.environ.get("SALLM_SOURCE_CACHE_DIR")
    cache = None
    if cache_dir and re.search(r"[0-9a-f]{40}", full_url):
        cache = Path(cache_dir) / hashlib.sha256(full_url.encode()).hexdigest()
        if cache.is_file():
            return cache.read_bytes()
    data = _fetch_url(url)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, cache)
    return data


def _fetch_url(url: str | Request) -> bytes:
    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as response:
                return response.read()
        except HTTPError as err:
            if err.code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        sleep(2**attempt)
    raise AssertionError("unreachable")


def _load_masakhapos_split(lang_code: str, split: str) -> Dataset:
    """Load a masakhapos split directly from dataset files (no dataset script)."""
    last_err: Exception | None = None
    for filename in _masakhapos_split_candidates(split):
        try:
            url = (
                f"{MASAKHAPOS_BASE_URL}/{lang_code}/{filename}"
                f"?ref={MASAKHAPOS_REVISION}"
            )
            request = Request(
                url,
                headers={"Accept": "application/vnd.github.raw+json"},
            )
            content = _read_url(request).decode("utf-8")
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


def _injongointent_split_candidates(split: str) -> list[str]:
    """Map requested split names to likely JSONL filenames in InjongoIntent."""
    s = split.lower()
    if s == "train":
        return ["train.jsonl"]
    if s in VALIDATION_ALIASES:
        return ["dev.jsonl", "validation.jsonl"]
    if s == "test":
        return ["test.jsonl"]
    return [f"{split}.jsonl"]


def _load_injongointent_split(lang_code: str, split: str) -> Dataset:
    """Load an InjongoIntent split directly from dataset files on the Hub."""
    last_err: Exception | None = None
    for filename in _injongointent_split_candidates(split):
        try:
            url = f"{INJONGOINTENT_BASE_URL}/{lang_code}/{filename}"
            rows = [
                {
                    **json.loads(line),
                    "lang": lang_code,
                }
                for line in _read_url(url).decode("utf-8").splitlines()
                if line.strip()
            ]
            return Dataset.from_list(rows)
        except Exception as err:  # noqa: BLE001 - try alternate filename candidates
            last_err = err

    assert last_err is not None
    raise last_err


def _load_injongointent_dataset(
    ds_cfg: FinetuneDatasetConfig,
) -> tuple[Dataset, Dataset, bool]:
    """Load InjongoIntent directly from Hub-hosted JSONL files."""
    splits = ds_cfg.splits
    lang_list_cfg = list(ds_cfg.languages or [])
    if not lang_list_cfg:
        if ds_cfg.subset:
            lang_list_cfg = [ds_cfg.subset]
        else:
            raise ValueError(
                "masakhane/InjongoIntent requires dataset.subset or dataset.languages."
            )

    train_parts: list[Dataset] = []
    val_parts: list[Dataset] = []
    for lang_code in lang_list_cfg:
        train_ds = _load_injongointent_split(lang_code, splits["train"])
        test_ds = _load_injongointent_split(lang_code, "test")
        train_ds = Dataset.from_list(
            exclude_heldout_texts(train_ds.to_list(), test_ds.to_list())
        )

        val_split = splits["val"]
        if val_split.lower() in VALIDATION_ALIASES:
            train_rows, val_rows = split_injongointent_rows(train_ds.to_list())
            if not val_rows:
                raise ValueError(
                    f"Could not derive InjongoIntent validation rows for {lang_code}."
                )
            train_ds = Dataset.from_list(train_rows)
            val_ds = Dataset.from_list(val_rows)
        else:
            val_ds = _load_injongointent_split(lang_code, val_split)

        train_parts.append(train_ds)
        val_parts.append(val_ds)

    if len(train_parts) == 1:
        return train_parts[0], val_parts[0], False

    return concatenate_datasets(train_parts), concatenate_datasets(val_parts), False


def _requested_languages(ds_cfg: FinetuneDatasetConfig, dataset_name: str) -> list[str]:
    """Resolve a required language list for language-scoped datasets."""
    lang_list_cfg = list(ds_cfg.languages or [])
    if lang_list_cfg:
        return lang_list_cfg
    if ds_cfg.subset:
        return [ds_cfg.subset]
    raise ValueError(f"{dataset_name} requires dataset.subset or dataset.languages.")


def _masakhaner_data_files(lang_code: str) -> dict[str, str]:
    """Return parquet file paths for the mirrored MasakhaNER language."""
    lang_dir = MASAKHANER_PARQUET_LANG_DIRS.get(lang_code)
    if lang_dir is None:
        supported = ", ".join(sorted(MASAKHANER_PARQUET_LANG_DIRS))
        raise ValueError(
            f"Unsupported MasakhaNER language '{lang_code}'. "
            f"Supported languages: {supported}."
        )
    return {
        "train": f"data/{lang_dir}/train.parquet",
        "validation": f"data/{lang_dir}/validation.parquet",
        "test": f"data/{lang_dir}/test.parquet",
    }


def _load_masakhaner_dataset(
    ds_cfg: FinetuneDatasetConfig,
) -> tuple[Dataset, Dataset, bool]:
    """Load MasakhaNER from explicit per-language parquet files.

    The upstream `masakhane/masakhaner2` dataset can fall back to a cached
    default config without language columns. That silently turns an xho/zul/tsn
    run into an all-language run, so use the mirrored parquet files with
    explicit language paths instead.
    """
    splits = ds_cfg.splits
    train_parts: list[Dataset] = []
    val_parts: list[Dataset] = []

    for lang_code in _requested_languages(ds_cfg, MASAKHANER_DATASET):
        data_files = _masakhaner_data_files(lang_code)
        train_ds = cast(
            Dataset,
            load_dataset(
                MASAKHANER_PARQUET_DATASET,
                data_files=data_files,
                split=splits["train"],
                revision=DATASET_REVISIONS[MASAKHANER_PARQUET_DATASET],
            ),
        )
        val_ds = load_split_with_fallback(
            MASAKHANER_PARQUET_DATASET,
            None,
            splits["val"],
            revision=DATASET_REVISIONS[MASAKHANER_PARQUET_DATASET],
            data_files=data_files,
        )
        if "lang" not in train_ds.column_names:
            train_ds = train_ds.add_column("lang", [lang_code] * len(train_ds))
        if "lang" not in val_ds.column_names:
            val_ds = val_ds.add_column("lang", [lang_code] * len(val_ds))
        train_parts.append(train_ds)
        val_parts.append(val_ds)

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
    if ds_cfg.hf_name == INJONGOINTENT_DATASET:
        return _load_injongointent_dataset(ds_cfg)
    if ds_cfg.hf_name == MASAKHANER_DATASET:
        return _load_masakhaner_dataset(ds_cfg)

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
