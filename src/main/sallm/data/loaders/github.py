from __future__ import annotations

from datasets import DatasetDict

from sallm.config import FinetuneDatasetConfig
from sallm.data.afrihg import load_afrihg_from_github
from sallm.data.loaders.base import DatasetLoader
from sallm.data.t2x import load_t2x_from_github


class GithubDatasetLoader(DatasetLoader):
    def load(self, config: FinetuneDatasetConfig) -> DatasetDict:
        hf_name = config.hf_name
        if not isinstance(hf_name, str):
            raise TypeError("dataset.hf_name must be a string when using GitHub loader")
        ref = hf_name[len("github:") :]
        if "francois-meyer/t2x" in ref or ref.strip().endswith("/t2x"):
            raw = load_t2x_from_github()
        else:
            raw = load_afrihg_from_github(languages=config.languages)
        if isinstance(raw, DatasetDict):
            return raw
        dataset_dict = DatasetDict()
        dataset_dict["train"] = raw
        dataset_dict["validation"] = raw
        return dataset_dict

