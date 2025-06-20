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

    logger.info("Building tokenizer...")
    tokenizer = build_tokenizer(config)

    logger.info("Building model...")
    model = build_model(config, tokenizer)

    logger.info("Building datasets...")
    train_dataset, eval_dataset, test_dataset = build_datasets(
        config, is_hpo=is_hpo_run
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    logger.info(f"Evaluation dataset features: {eval_dataset.features}")
    if test_dataset:
        logger.info(f"Test dataset size: {len(test_dataset)}")
    elif not is_hpo_run:
        logger.warning("No test set found or specified for this final training run.")

    logger.info("Building trainer...")
    trainer = build_trainer(config, model, tokenizer, train_dataset, eval_dataset)

    resume_from_checkpoint = config.training.get("resume_from_checkpoint", None)

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logger.info("Training finished.")

    if not is_hpo_run:
        final_model_path = os.path.join(trainer.args.output_dir, "final_model")
        logger.info(f"Saving final model to {final_model_path}...")
        trainer.save_model(final_model_path)
        logger.info("Model saved.")
    else:
        logger.info("HPO trial finished. Skipping final model save to conserve space.")

    if test_dataset:
        logger.info("Starting final evaluation on the test set...")
        test_results = trainer.predict(test_dataset)
        logger.info(f"Test results: {test_results.metrics}")

    logger.info("Run finished.")
