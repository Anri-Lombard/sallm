from __future__ import annotations

from sallm.config import FinetuneDatasetConfig
from sallm.data.adapters.base import RawDatasetSplits, extract_train_validation_splits
from sallm.data.afrihg import load_afrihg_from_github
from sallm.data.t2x import load_t2x_from_github


def github_ref(ds_cfg: FinetuneDatasetConfig) -> str:
    hf_name = str(ds_cfg.hf_name)
    return hf_name[len("github:") :] if hf_name.startswith("github:") else hf_name


class GitHubT2XAdapter:
    name = "github-t2x"

    def supports(self, ds_cfg: FinetuneDatasetConfig) -> bool:
        ref = github_ref(ds_cfg).strip()
        return str(ds_cfg.hf_name).startswith("github:") and (
            "francois-meyer/t2x" in ref or ref.endswith("/t2x")
        )

    def load(self, ds_cfg: FinetuneDatasetConfig) -> RawDatasetSplits:
        return extract_train_validation_splits(load_t2x_from_github())


class GitHubAfriHGAdapter:
    name = "github-afrihg"

    def supports(self, ds_cfg: FinetuneDatasetConfig) -> bool:
        return str(ds_cfg.hf_name).startswith("github:")

    def load(self, ds_cfg: FinetuneDatasetConfig) -> RawDatasetSplits:
        languages = [ds_cfg.subset] if ds_cfg.subset else ds_cfg.languages
        return extract_train_validation_splits(
            load_afrihg_from_github(languages=languages)
        )
