"""Process and W&B setup shared by training and fine-tuning runs."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import wandb
from omegaconf import OmegaConf

from sallm.configs import to_resolved_dict

logger = logging.getLogger(__name__)


def is_main_process() -> bool:
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    except ValueError:
        return True
    return local_rank in (-1, 0)


def is_hpo_run(config: Any) -> bool:
    """HPO trials receive a W&B id containing ``sweep`` (see hpo/trial.py)."""
    wb_id = getattr(getattr(config, "wandb", None), "id", None)
    return isinstance(wb_id, str) and "sweep" in wb_id


def set_local_cuda_device() -> None:
    """Pin each distributed process to its logical CUDA device."""
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank == -1:
            return
        torch.cuda.set_device(local_rank)
    except Exception as exc:
        logger.warning("Could not set the local CUDA device: %s", exc)


def init_wandb(config: Any) -> None:
    """Start the W&B run on the main process, except for HPO trials."""
    wb = config.wandb
    if not (wb and wb.project and is_main_process() and not is_hpo_run(config)):
        return
    wandb.init(
        project=wb.project,
        entity=wb.entity,
        group=wb.group,
        name=wb.name,
        id=wb.id,
        config=to_resolved_dict(OmegaConf.structured(config), name="wandb config"),
        settings=wandb.Settings(init_timeout=120),
    )
    if wb.id:
        wandb.config.update({"resume": "allow"})
