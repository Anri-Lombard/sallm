from __future__ import annotations

import logging
import os
from typing import Any, cast

from sallm.config import ExperimentConfig
from sallm.data.factory import build_datasets
from sallm.models.factory import build_model, build_tokenizer
from sallm.runtime import init_wandb, is_hpo_run, is_main_process
from sallm.training.factory import build_trainer

logger = logging.getLogger(__name__)


def run(config: ExperimentConfig) -> None:
    hpo_run = is_hpo_run(config)
    i_am_main = is_main_process()
    init_wandb(config)

    tokenizer = build_tokenizer(config)
    model = build_model(config, tokenizer)
    cast(Any, model).tokenizer = tokenizer

    train_ds, val_ds, test_ds = build_datasets(config, tokenizer, is_hpo=hpo_run)

    trainer = build_trainer(config, model, tokenizer, train_ds, val_ds)
    trainer.train(
        resume_from_checkpoint=(config.training or {}).get("resume_from_checkpoint")
    )

    if not hpo_run:
        out = os.path.join(str(trainer.args.output_dir), "final_model")
        trainer.accelerator.wait_for_everyone()
        if trainer.args.should_save:
            trainer.save_model(out)
            logger.info("Saved model -> %s", out)
        trainer.accelerator.wait_for_everyone()

        if config.hub and config.hub.enabled and i_am_main:
            if config.model is None:
                raise ValueError("Hub push requires a `model` config block.")
            if config.hub.repo_id:
                repo_id = config.hub.repo_id.strip().rstrip("/")
            else:
                org = (config.hub.organization or "").strip()
                if not org:
                    raise ValueError(
                        "Set `hub.organization` or `hub.repo_id` when "
                        "`hub.enabled=true`."
                    )
                repo_id = f"{org}/sallm-{config.model.architecture}"
            logger.info("Pushing base model to HuggingFace Hub: %s", repo_id)
            cast(Any, model).push_to_hub(repo_id, private=config.hub.private)
            tokenizer.push_to_hub(repo_id, private=config.hub.private)

    if test_ds:
        res = trainer.predict(cast(Any, test_ds))
        logger.info("Test metrics: %s", res.metrics)
