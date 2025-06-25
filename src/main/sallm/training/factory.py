import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from datasets import Dataset

from sallm.config import ExperimentConfig
from sallm.training.trainer import CustomTrainer

logger = logging.getLogger(__name__)


def build_trainer(
    config: ExperimentConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> CustomTrainer:
    training_args = TrainingArguments(**config.training)

    if training_args.local_rank <= 0:  # Print only on the main process
        logger.info("--- Effective Training Arguments ---")
        # The __str__ method of TrainingArguments provides a nice, readable format.
        logger.info(training_args)
        logger.info("------------------------------------")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # TODO: depricated, so use a different method eventually
    trainer.tokenizer = tokenizer

    return trainer
