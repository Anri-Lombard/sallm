from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import requests
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/dadelani/AfriHG/main"


def _load_csv_entries(entries: list[tuple[str, str]]) -> Dataset:
    """Load CSV files while preserving each source language."""
    datasets: list[Dataset] = []
    for language, path in entries:
        dataset = cast(DatasetDict, load_dataset("csv", data_files=path))["train"]
        if "lang" not in dataset.column_names:
            dataset = dataset.add_column("lang", [language] * len(dataset))
        datasets.append(dataset)
    return concatenate_datasets(datasets)


def load_afrihg_from_github(
    languages: list[str] | None = None, cache_dir: str | None = None
) -> DatasetDict:
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "data", "afrihg_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if languages is None:
        wanted = ["xho", "zul"]
    else:
        wanted = list(languages)
    session = requests.Session()
    splits: dict[str, list[tuple[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for code in wanted:
        lang_dir = f"data/{code}"
        for split_name in ["train", "dev", "validation", "test"]:
            filename = f"{lang_dir}/{split_name}.csv"
            url = f"{GITHUB_RAW_BASE}/{filename}"
            resp = session.get(url, stream=True)
            if resp.status_code != 200:
                continue
            dataset_split = (
                "validation" if split_name in ("dev", "validation") else split_name
            )
            dest = Path(cache_dir) / f"{code}_{split_name}.csv"
            if not dest.exists():
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(8192):
                        fh.write(chunk)
            if dataset_split == "validation":
                splits["validation"].append((code, str(dest)))
            elif dataset_split == "test":
                splits["test"].append((code, str(dest)))
            else:
                splits["train"].append((code, str(dest)))

    dataset_dict = DatasetDict()
    if splits["train"]:
        dataset_dict["train"] = _load_csv_entries(splits["train"])
    if splits["validation"]:
        dataset_dict["validation"] = _load_csv_entries(splits["validation"])
    if splits["test"]:
        dataset_dict["test"] = _load_csv_entries(splits["test"])

    if not dataset_dict:
        raise RuntimeError(f"No AFriHG CSVs found on GitHub for languages={wanted}.")

    return dataset_dict
