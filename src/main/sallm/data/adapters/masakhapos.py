from __future__ import annotations

from urllib.request import urlopen

from datasets import Dataset, concatenate_datasets

from sallm.config import FinetuneDatasetConfig
from sallm.data.adapters.base import RawDatasetSplits, required_languages
from sallm.data.loaders.base import VALIDATION_ALIASES

MASAKHAPOS_DATASET = "masakhane/masakhapos"
MASAKHAPOS_BASE_URL = "https://github.com/masakhane-io/masakhane-pos/raw/main/data"


class MasakhaPOSAdapter:
    name = "masakhapos"

    def supports(self, ds_cfg: FinetuneDatasetConfig) -> bool:
        return ds_cfg.hf_name == MASAKHAPOS_DATASET

    def load(self, ds_cfg: FinetuneDatasetConfig) -> RawDatasetSplits:
        splits = ds_cfg.splits
        train_parts: list[Dataset] = []
        val_parts: list[Dataset] = []
        for lang_code in required_languages(ds_cfg, MASAKHAPOS_DATASET):
            train_parts.append(load_masakhapos_split(lang_code, splits["train"]))
            val_parts.append(load_masakhapos_split(lang_code, splits["val"]))

        if len(train_parts) == 1:
            return RawDatasetSplits(train_parts[0], val_parts[0])

        return RawDatasetSplits(
            concatenate_datasets(train_parts),
            concatenate_datasets(val_parts),
        )


def masakhapos_split_candidates(split: str) -> list[str]:
    s = split.lower()
    if s == "train":
        return ["train.txt"]
    if s in VALIDATION_ALIASES:
        return ["dev.txt", "validation.txt", "val.txt", "valid.txt"]
    if s == "test":
        return ["test.txt"]
    return [f"{split}.txt"]


def parse_masakhapos_conll(content: str, lang_code: str) -> Dataset:
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


def load_masakhapos_split(lang_code: str, split: str) -> Dataset:
    last_err: Exception | None = None
    for filename in masakhapos_split_candidates(split):
        try:
            url = f"{MASAKHAPOS_BASE_URL}/{lang_code}/{filename}"
            with urlopen(url, timeout=30) as response:
                content = response.read().decode("utf-8")
            return parse_masakhapos_conll(content, lang_code)
        except Exception as err:  # noqa: BLE001 - try alternate filename candidates
            last_err = err

    assert last_err is not None
    raise last_err
