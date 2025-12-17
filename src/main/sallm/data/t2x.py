from __future__ import annotations

import os
from pathlib import Path

import requests
from datasets import Dataset, DatasetDict

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/francois-meyer/t2x/main"


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return [ln.rstrip("\n") for ln in fh]


def _make_dataset_from_files(data_file: str, text_file: str) -> Dataset:
    data_lines = _read_lines(data_file)
    text_lines = _read_lines(text_file)
    n = min(len(data_lines), len(text_lines))
    records = [{"source": data_lines[i], "target": text_lines[i]} for i in range(n)]
    return Dataset.from_list(records)


def load_t2x_from_github(cache_dir: str | None = None) -> DatasetDict:
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "data", "t2x_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    files = {
        "train": ("train.data", "train.text"),
        "validation": ("valid.data", "valid.text"),
        "test": ("test.data", "test.text"),
    }
    ds = DatasetDict()
    session = requests.Session()

    for split, (data_name, text_name) in files.items():
        dest_data = Path(cache_dir) / data_name
        dest_text = Path(cache_dir) / text_name

        for url, dest in [
            (f"{GITHUB_RAW_BASE}/{data_name}", dest_data),
            (f"{GITHUB_RAW_BASE}/{text_name}", dest_text),
        ]:
            if not dest.exists():
                resp = session.get(url, stream=True)
                if resp.status_code != 200:
                    break
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(8192):
                        fh.write(chunk)

        if dest_data.exists() and dest_text.exists():
            ds[split] = _make_dataset_from_files(str(dest_data), str(dest_text))

    if not ds:
        raise RuntimeError("Failed to download T2X dataset from GitHub")

    return ds
