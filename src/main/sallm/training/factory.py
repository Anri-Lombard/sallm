import logging
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from sallm.config import ExperimentConfig
from sallm.training.callbacks import ShowCompletionsCallback

logger = logging.getLogger(__name__)


def build_trainer(
    config: ExperimentConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> SFTTrainer:
    training_args_dict = OmegaConf.to_container(config.training, resolve=True)

    training_args = SFTConfig(
        **training_args_dict,
        max_seq_length=config.dataset.max_seq_length if config.dataset else None,
        dataset_text_field="text",
        completion_only_loss=True,
        packing=False,
    )

    if training_args.local_rank <= 0:
        logger.info("--- Effective Training Arguments ---")
        logger.info(training_args)
        logger.info("------------------------------------")

    completions_callback = ShowCompletionsCallback(
        eval_dataset=eval_dataset, tokenizer=tokenizer, num_samples=5
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[completions_callback],
    )

    trainer.tokenizer = tokenizer

    return trainer
