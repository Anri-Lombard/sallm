import logging
import os
from collections.abc import Mapping
from typing import Any, cast

import wandb
from omegaconf import OmegaConf

from sallm.config import ExperimentConfig, to_resolved_dict
from sallm.data.factory import build_datasets
from sallm.models.factory import build_model, build_tokenizer
from sallm.training.factory import build_trainer

logger = logging.getLogger(__name__)


def _is_main_process() -> bool:
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    except Exception:
        return True
    return local_rank in (-1, 0)


def _assert_train_batch_length_if_requested(trainer: Any) -> None:
    """Check a real training batch when the canary explicitly requests it."""
    raw_expected = os.environ.get("SALLM_ASSERT_TRAIN_BATCH_LENGTH")
    if raw_expected is None:
        return
    try:
        expected = int(raw_expected)
    except ValueError as exc:
        raise ValueError(
            "SALLM_ASSERT_TRAIN_BATCH_LENGTH must be a positive integer, "
            f"got {raw_expected!r}."
        ) from exc
    if expected <= 0:
        raise ValueError(
            "SALLM_ASSERT_TRAIN_BATCH_LENGTH must be a positive integer, "
            f"got {raw_expected!r}."
        )

    try:
        batch = next(iter(trainer.get_train_dataloader()))
    except StopIteration as exc:
        raise ValueError(
            "Cannot assert packed batch length: training data is empty."
        ) from exc
    if not isinstance(batch, Mapping) or "input_ids" not in batch:
        raise ValueError(
            "Cannot assert packed batch length: the real training batch has no "
            "`input_ids`."
        )
    input_ids = batch["input_ids"]
    shape = tuple(getattr(input_ids, "shape", ()))
    if len(shape) < 2:
        raise ValueError(
            "Cannot assert packed batch length: `input_ids` must be rank two or "
            f"higher, got shape {tuple(shape)!r}."
        )
    actual = int(shape[-1])
    if actual != expected:
        raise AssertionError(
            "Real training batch length does not match "
            f"SALLM_ASSERT_TRAIN_BATCH_LENGTH: expected {expected}, got {actual}."
        )
    logger.info("Verified real training batch length: %d tokens.", actual)


def run(config: ExperimentConfig) -> None:
    is_hpo_run = config.wandb.id is not None and "sweep" in config.wandb.id

    i_am_main = _is_main_process()

    if config.wandb and config.wandb.project and (not is_hpo_run) and i_am_main:
        cfg_for_wandb = OmegaConf.to_container(
            OmegaConf.structured(config), resolve=True, throw_on_missing=True
        )
        cfg_for_wandb = to_resolved_dict(cfg_for_wandb, name="wandb config")
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=config.wandb.name,
            id=config.wandb.id,
            config=cfg_for_wandb,
        )
        if config.wandb.id:
            wandb.config.update({"resume": "allow"})

    tokenizer = build_tokenizer(config)
    model = build_model(config, tokenizer)

    cast(Any, model).tokenizer = tokenizer

    train_ds, val_ds, test_ds = build_datasets(config, tokenizer, is_hpo=is_hpo_run)

    trainer = build_trainer(config, model, tokenizer, train_ds, val_ds)
    _assert_train_batch_length_if_requested(trainer)
    trainer.train(
        resume_from_checkpoint=(config.training or {}).get("resume_from_checkpoint")
    )

    if not is_hpo_run:
        out = os.path.join(str(trainer.args.output_dir), "final_model")
        trainer.accelerator.wait_for_everyone()
        if trainer.args.should_save:
            trainer.save_model(out)
            logger.info(f"Saved model → {out}")
        trainer.accelerator.wait_for_everyone()

        if config.hub and config.hub.enabled and i_am_main:
            if config.model is None:
                raise ValueError("Hub push requires a `model` config block.")
            repo_id = f"{config.hub.organization}/sallm-{config.model.architecture}"
            logger.info(f"Pushing base model to HuggingFace Hub: {repo_id}")
            cast(Any, model).push_to_hub(repo_id, private=config.hub.private)
            tokenizer.push_to_hub(repo_id, private=config.hub.private)

    # TODO: do per language
    if test_ds:
        res = trainer.predict(cast(Any, test_ds))
        logger.info(f"Test metrics: {res.metrics}")
