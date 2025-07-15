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

    # TODO validation
    packing = getattr(config.dataset, "packing", False)
    assistant_only_loss = getattr(config.dataset, "assistant_only_loss", True)

    if packing and assistant_only_loss:
        logger.warning(
            "You are using `packing=True` with `assistant_only_loss=True`. "
            "Due to a known bug in TRL, this combination is not effective, and the loss will be "
            "computed on all tokens. See: https://github.com/huggingface/trl/issues/3728"
        )

    training_args = SFTConfig(
        **training_args_dict,
        max_seq_length=config.dataset.max_seq_length,
        packing=packing,
        assistant_only_loss=assistant_only_loss,
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
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[completions_callback],
    )

    trainer.processing_class = tokenizer

    return trainer
