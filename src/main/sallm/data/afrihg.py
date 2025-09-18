from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
from datasets import DatasetDict, load_dataset

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/dadelani/AfriHG/main"


def load_afrihg_from_github(
    languages: list[str] | None = None, cache_dir: str | None = None
) -> DatasetDict:
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "data", "afrihg_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    wanted = ["xho", "zul"]
    if languages:
        wanted = [lang for lang in languages if lang in ("xho", "zul")]
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

    if not dataset_dict:
        hf = load_dataset("dadelani/AfriHG", trust_remote_code=True)
        if isinstance(hf, DatasetDict):
            for k in hf.keys():
                target_k = "validation" if k in ("dev", "validation", "val") else k
                dataset_dict[target_k] = hf[k]
        else:
            dataset_dict["train"] = hf
        if languages:
            for split in list(dataset_dict.keys()):
                ds = dataset_dict[split]

                def _in_lang(ex: dict[str, Any]) -> bool:
                    code = (
                        ex.get("lang") or ex.get("language") or ex.get("language_code")
                    )
                    return code in languages if code is not None else False

                dataset_dict[split] = ds.filter(_in_lang)

    return dataset_dict
