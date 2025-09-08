import logging
from omegaconf import OmegaConf
import inspect
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from sallm.config import ExperimentConfig, RunMode
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

    # TODO implement cleaner logic for this
    if config.mode == RunMode.FINETUNE:
        if config.dataset is None:
            raise ValueError("`dataset` config block must be provided for fine-tuning.")
        max_seq_length = config.dataset.max_seq_length
        packing = bool(getattr(config.dataset, "packing", False))
        assistant_only_loss = bool(getattr(config.dataset, "assistant_only_loss", True))
    else:
        if "max_seq_length" in training_args_dict:
            max_seq_length = training_args_dict.pop("max_seq_length")
        else:
            max_seq_length = 2048
            logger.warning(
                f"SFTConfig `max_seq_length` not found. Falling back to {max_seq_length}. "
                "Please add `max_seq_length` to your training config."
            )
        packing = False
        assistant_only_loss = False

    # Ensure we don't accidentally pass duplicates for values we set explicitly
    for _k in ("max_seq_length", "packing", "assistant_only_loss"):
        training_args_dict.pop(_k, None)

    sft_sig = inspect.signature(SFTConfig)
    allowed_keys = set(sft_sig.parameters.keys())
    provided_keys = set(training_args_dict.keys())
    unknown = sorted(list(provided_keys - allowed_keys))
    if unknown:
        raise ValueError(
            "Found unsupported training config keys that SFTConfig does not accept: "
            f"{unknown}.\nPlease remove these keys from your `training` config or move them to the appropriate place. "
            "Allowed keys for SFTConfig are: " + ", ".join(sorted(list(allowed_keys)))
        )

    training_args = SFTConfig(
        **training_args_dict,
        max_seq_length=max_seq_length,
        packing=packing,
        assistant_only_loss=assistant_only_loss,
    )

    if training_args.local_rank <= 0:
        logger.info("--- Effective Training Arguments ---")
        logger.info(training_args)
        logger.info("------------------------------------")

    callbacks = []
    if config.mode == RunMode.FINETUNE:
        completions_callback = ShowCompletionsCallback(
            eval_dataset=eval_dataset, tokenizer=tokenizer, num_samples=5
        )
        callbacks.append(completions_callback)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        processing_class=tokenizer,
    )

    trainer.processing_class = tokenizer

    return trainer
