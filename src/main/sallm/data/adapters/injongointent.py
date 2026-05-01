from __future__ import annotations

import json
from urllib.request import urlopen

from datasets import Dataset, concatenate_datasets

from sallm.config import FinetuneDatasetConfig
from sallm.data.adapters.base import RawDatasetSplits, required_languages
from sallm.data.loaders.base import VALIDATION_ALIASES
from sallm.data.loaders.injongointent_split import split_injongointent_rows

INJONGOINTENT_DATASET = "masakhane/InjongoIntent"
INJONGOINTENT_BASE_URL = (
    "https://huggingface.co/datasets/masakhane/InjongoIntent/resolve/main"
)


class InjongoIntentAdapter:
    name = "injongointent"

    def supports(self, ds_cfg: FinetuneDatasetConfig) -> bool:
        return ds_cfg.hf_name == INJONGOINTENT_DATASET

    def load(self, ds_cfg: FinetuneDatasetConfig) -> RawDatasetSplits:
        splits = ds_cfg.splits
        train_parts: list[Dataset] = []
        val_parts: list[Dataset] = []
        for lang_code in required_languages(ds_cfg, INJONGOINTENT_DATASET):
            train_ds = load_injongointent_split(lang_code, splits["train"])

            val_split = splits["val"]
            if val_split.lower() in VALIDATION_ALIASES:
                try:
                    val_ds = load_injongointent_split(lang_code, val_split)
                except Exception:
                    train_rows, val_rows = split_injongointent_rows(train_ds.to_list())
                    if not val_rows:
                        raise
                    train_ds = Dataset.from_list(train_rows)
                    val_ds = Dataset.from_list(val_rows)
            else:
                val_ds = load_injongointent_split(lang_code, val_split)

            train_parts.append(train_ds)
            val_parts.append(val_ds)

        if len(train_parts) == 1:
            return RawDatasetSplits(train_parts[0], val_parts[0])

        return RawDatasetSplits(
            concatenate_datasets(train_parts),
            concatenate_datasets(val_parts),
        )


def injongointent_split_candidates(split: str) -> list[str]:
    s = split.lower()
    if s == "train":
        return ["train.jsonl"]
    if s in VALIDATION_ALIASES:
        return ["dev.jsonl", "validation.jsonl"]
    if s == "test":
        return ["test.jsonl"]
    return [f"{split}.jsonl"]


def load_injongointent_split(lang_code: str, split: str) -> Dataset:
    last_err: Exception | None = None
    for filename in injongointent_split_candidates(split):
        try:
            url = f"{INJONGOINTENT_BASE_URL}/{lang_code}/{filename}"
            with urlopen(url, timeout=30) as response:
                rows = [
                    {
                        **json.loads(line),
                        "lang": lang_code,
                    }
                    for line in response.read().decode("utf-8").splitlines()
                    if line.strip()
                ]
            return Dataset.from_list(rows)
        except Exception as err:  # noqa: BLE001 - try alternate filename candidates
            last_err = err

    assert last_err is not None
    raise last_err
