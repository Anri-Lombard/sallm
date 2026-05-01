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
    organization: str = "anrilombard"
    private: bool = True
    push_adapter: bool = True
    push_merged: bool = False
    base_model_id: str = "anrilombard/mzansilm-125m"
    collection_slug: str | None = "anrilombard/mzansilm-69635ca7b60efedb9dfcb09e"
