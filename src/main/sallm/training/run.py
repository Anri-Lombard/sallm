import logging
import os

from sallm.config import ExperimentConfig
from sallm.data.factory import build_datasets
from sallm.models.factory import build_model, build_tokenizer
from sallm.training.factory import build_trainer

logger = logging.getLogger(__name__)


def run(config: ExperimentConfig) -> None:
    is_hpo_run = "WANDB_SWEEP_ID" in os.environ

    if not is_hpo_run:
        if config.wandb.project:
            os.environ["WANDB_PROJECT"] = config.wandb.project
        if config.wandb.name:
            os.environ["WANDB_RUN_NAME"] = config.wandb.name
    if config.wandb.id:
        os.environ["WANDB_RUN_ID"] = config.wandb.id
        os.environ["WANDB_RESUME"] = "allow"

    tokenizer = build_tokenizer(config)
    model = build_model(config, tokenizer)
    train_ds, val_ds, test_ds = build_datasets(config, is_hpo=is_hpo_run)

    trainer = build_trainer(config, model, tokenizer, train_ds, val_ds)
    trainer.train(resume_from_checkpoint=config.training.get("resume_from_checkpoint"))

    if not is_hpo_run:
        out = os.path.join(trainer.args.output_dir, "final_model")
        trainer.save_model(out)
        logger.info(f"Saved model → {out}")

    # TODO: do per language
    if test_ds:
        res = trainer.predict(test_ds)
        logger.info(f"Test metrics: {res.metrics}")
