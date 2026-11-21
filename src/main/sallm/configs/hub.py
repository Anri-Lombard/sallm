from __future__ import annotations

from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class WandbConfig:
    project: str = MISSING
    entity: str | None = None
    group: str | None = None
    name: str | None = MISSING
    id: str | None = None


@dataclass
class HubConfig:
    enabled: bool = False
    organization: str | None = None
    repo_id: str | None = None
    private: bool = True
    collection_slug: str | None = None
