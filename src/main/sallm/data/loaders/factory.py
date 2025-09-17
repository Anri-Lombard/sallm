from __future__ import annotations

from sallm.config import FinetuneDatasetConfig

from .base import DatasetLoader
from .github import GithubDatasetLoader
from .hub import HubDatasetLoader


def get_loader(config: FinetuneDatasetConfig) -> DatasetLoader:
    hf_name = config.hf_name
    if isinstance(hf_name, str) and hf_name.startswith("github:"):
        return GithubDatasetLoader()
    return HubDatasetLoader()
