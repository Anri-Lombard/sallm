from __future__ import annotations

import os
from pathlib import Path

import requests
from datasets import DatasetDict, load_dataset

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/dadelani/AfriHG/main"


def load_afrihg_from_github(
    languages: list[str] | None = None, cache_dir: str | None = None
) -> DatasetDict:
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "data", "afrihg_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    supported = {"xho", "zul"}
    if languages is None:
        wanted = ["xho", "zul"]
    else:
        unknown = [lang for lang in languages if lang not in supported]
        if unknown:
            raise ValueError(
                f"Unsupported AFriHG languages: {unknown}. "
                f"Supported: {sorted(supported)}"
            )
        wanted = list(languages)
    session = requests.Session()
    splits: dict[str, list[str]] = {"train": [], "validation": [], "test": []}

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
                splits["validation"].append(str(dest))
            elif dataset_split == "test":
                splits["test"].append(str(dest))
            else:
                splits["train"].append(str(dest))

    dataset_dict = DatasetDict()
    if splits["train"]:
        dataset_dict["train"] = load_dataset("csv", data_files=splits["train"])["train"]
    if splits["validation"]:
        dataset_dict["validation"] = load_dataset(
            "csv", data_files=splits["validation"]
        )["train"]
    if splits["test"]:
        dataset_dict["test"] = load_dataset("csv", data_files=splits["test"])["train"]

    if len(wanted) == 1:
        lang_code = wanted[0]
        for split_name, ds in list(dataset_dict.items()):
            if "lang" not in ds.column_names:
                dataset_dict[split_name] = ds.add_column("lang", [lang_code] * len(ds))

    if not dataset_dict:
        requested = languages if languages is not None else sorted(list(supported))
        raise RuntimeError(f"No AFriHG CSVs found on GitHub for languages={requested}.")

    return dataset_dict
