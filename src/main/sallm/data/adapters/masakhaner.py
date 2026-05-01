from __future__ import annotations

from datasets import Dataset, concatenate_datasets, load_dataset

from sallm.config import FinetuneDatasetConfig
from sallm.data.adapters.base import RawDatasetSplits, required_languages
from sallm.data.loaders.base import load_split_with_fallback

MASAKHANER_DATASET = "masakhane/masakhaner2"
MASAKHANER_PARQUET_DATASET = "anrilombard/masakhaner-x-parquet"
MASAKHANER_PARQUET_LANG_DIRS = {
    "tsn": "tn",
    "xho": "xh",
    "zul": "zu",
}


class MasakhaNERAdapter:
    name = "masakhaner"

    def supports(self, ds_cfg: FinetuneDatasetConfig) -> bool:
        return ds_cfg.hf_name == MASAKHANER_DATASET

    def load(self, ds_cfg: FinetuneDatasetConfig) -> RawDatasetSplits:
        splits = ds_cfg.splits
        train_parts: list[Dataset] = []
        val_parts: list[Dataset] = []

        for lang_code in required_languages(ds_cfg, MASAKHANER_DATASET):
            data_files = masakhaner_data_files(lang_code)
            train_ds = load_dataset(
                MASAKHANER_PARQUET_DATASET,
                data_files=data_files,
                split=splits["train"],
            )
            val_ds = load_split_with_fallback(
                MASAKHANER_PARQUET_DATASET,
                None,
                splits["val"],
                None,
                data_files=data_files,
            )
            if "lang" not in train_ds.column_names:
                train_ds = train_ds.add_column("lang", [lang_code] * len(train_ds))
            if "lang" not in val_ds.column_names:
                val_ds = val_ds.add_column("lang", [lang_code] * len(val_ds))
            train_parts.append(train_ds)
            val_parts.append(val_ds)

        if len(train_parts) == 1:
            return RawDatasetSplits(train_parts[0], val_parts[0])

        return RawDatasetSplits(
            concatenate_datasets(train_parts),
            concatenate_datasets(val_parts),
        )


def masakhaner_data_files(lang_code: str) -> dict[str, str]:
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
