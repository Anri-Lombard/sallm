import logging

import wandb

from sallm.config import ExperimentConfig
from sallm.data.factory import build_datasets
from sallm.models.factory import build_model, build_tokenizer
from sallm.training.factory import build_trainer

logger = logging.getLogger(__name__)


def run(config: ExperimentConfig) -> None:
    """
    Executes a training run, which can be a standalone run or part of an HPO sweep.
    """
    run_instance = wandb.init(**config.wandb.model_dump(exclude_none=True))

    is_hpo_run = run_instance.sweep_id is not None
    if is_hpo_run:
        logger.info("Detected HPO run (part of a wandb sweep).")
        for key, value in wandb.config.items():
            if hasattr(config.training, key):
                logger.info(f"  - Sweep override: {key} = {value}")
                setattr(config.training, key, value)
    else:
        logger.info("Detected a single, standalone training run.")

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
    if test_dataset:
        logger.info(f"Test dataset size: {len(test_dataset)}")
    elif not is_hpo_run:
        logger.warning("No test set found or specified for this final training run.")

    logger.info("Building trainer...")
    trainer = build_trainer(config, model, tokenizer, train_dataset, eval_dataset)

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    logger.info("Training finished.")

    if not is_hpo_run:
        final_model_path = f"{config.training.output_dir}/final_model"
        logger.info(f"Saving final model to {final_model_path}...")
        trainer.save_model(final_model_path)
        logger.info("Model saved.")
    else:
        logger.info("HPO trial finished. Skipping final model save to conserve space.")

    if test_dataset:
        logger.info("Starting final evaluation on the test set...")
        test_results = trainer.predict(test_dataset)

        test_metrics = {f"test_{k}": v for k, v in test_results.metrics.items()}
        wandb.log(test_metrics)
        logger.info(f"Test results: {test_metrics}")

    wandb.finish()
    logger.info("Run finished.")
