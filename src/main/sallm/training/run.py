import logging
import wandb
import os
from omegaconf import OmegaConf

from sallm.config import ExperimentConfig
from sallm.data.factory import build_datasets
from sallm.models.factory import build_model, build_tokenizer
from sallm.training.factory import build_trainer

logger = logging.getLogger(__name__)


def run(config: ExperimentConfig) -> None:
    is_hpo_run = config.wandb.id is not None and "sweep" in config.wandb.id

    sel = OmegaConf.select(config, "runtime.is_main")
    i_am_main = bool(True if sel is None else sel)

    if config.wandb and config.wandb.project and (not is_hpo_run) and i_am_main:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=config.wandb.name,
            id=config.wandb.id,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        )
        if config.wandb.id:
            wandb.config.update({"resume": "allow"})

    tokenizer = build_tokenizer(config)
    model = build_model(config, tokenizer)

    model.tokenizer = tokenizer

    train_ds, val_ds, test_ds = build_datasets(config, tokenizer, is_hpo=is_hpo_run)

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
