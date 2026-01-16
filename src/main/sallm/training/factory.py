import inspect
import logging

from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import SFTConfig

from sallm.config import ExperimentConfig, FinetuneTaskType, RunMode
from sallm.training.callbacks import (
    ClassificationMetricsCallback,
    EnsureStaticGraphCallback,
    GenerationMetricsCallback,
    ShowCompletionsCallback,
)
from sallm.training.trainer import CustomSFTTrainer

logger = logging.getLogger(__name__)


class _DatasetEpochCallback(TrainerCallback):
    def __init__(self, dataset) -> None:
        self._dataset = dataset

    def on_train_begin(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if hasattr(self._dataset, "set_epoch"):
            self._dataset.set_epoch(0)
        return control

    def on_epoch_begin(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if hasattr(self._dataset, "set_epoch"):
            epoch_index = int(state.epoch or 0)
            self._dataset.set_epoch(epoch_index)
        return control


def build_trainer(
    config: ExperimentConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset | TorchDataset,
    eval_dataset: Dataset | TorchDataset,
) -> CustomSFTTrainer:
    training_raw = config.training or {}
    if isinstance(training_raw, DictConfig):
        training_args_dict = OmegaConf.to_container(training_raw, resolve=True)
    elif isinstance(training_raw, dict):
        training_args_dict = dict(training_raw)
    else:
        raise TypeError(f"Unsupported type for training config: {type(training_raw)!r}")

    # TODO implement cleaner logic for this
    if config.mode == RunMode.FINETUNE:
        if config.dataset is None:
            raise ValueError("`dataset` config block must be provided for fine-tuning.")
        max_length = config.dataset.max_seq_length
        packing = bool(getattr(config.dataset, "packing", False))
        assistant_only_loss = bool(getattr(config.dataset, "assistant_only_loss", True))
    else:
        if "max_seq_length" in training_args_dict:
            max_length = training_args_dict.pop("max_seq_length")
        elif "max_length" in training_args_dict:
            max_length = training_args_dict.pop("max_length")
        else:
            max_length = 2048
            logger.warning(
                "SFTConfig `max_length` not found. Falling back to %s. "
                "Please add `max_length` to your training config.",
                max_length,
            )
        packing = False
        assistant_only_loss = False

    # Ensure we don't accidentally pass duplicates for values we set explicitly
    for _k in ("max_seq_length", "max_length", "packing", "assistant_only_loss"):
        training_args_dict.pop(_k, None)

    sft_sig = inspect.signature(SFTConfig)
    allowed_keys = set(sft_sig.parameters.keys())
    provided_keys = set(training_args_dict.keys())
    unknown = sorted(list(provided_keys - allowed_keys))
    if unknown:
        raise ValueError(
            (
                "Found unsupported training config keys that SFTConfig does not "
                "accept: {}.\nPlease remove these keys from your `training` "
                "config or move them to the appropriate place. Allowed keys "
                "for SFTConfig are: {}"
            ).format(unknown, ", ".join(sorted(list(allowed_keys))))
        )

    training_args = SFTConfig(
        **training_args_dict,
        max_length=max_length,
        packing=packing,
        assistant_only_loss=assistant_only_loss,
    )

    if getattr(training_args, "gradient_checkpointing", False) and not getattr(
        training_args, "gradient_checkpointing_kwargs", None
    ):
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if training_args.local_rank <= 0:
        logger.info("--- Effective Training Arguments ---")
        logger.info(training_args)
        logger.info("------------------------------------")

    callbacks = []
    if config.mode == RunMode.FINETUNE:
        completions_callback = ShowCompletionsCallback(
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            num_samples=5,
            decoding=config.generation_decoding,
        )
        callbacks.append(completions_callback)

        task_type = getattr(config.dataset, "task", None)
        if task_type == FinetuneTaskType.CLASSIFICATION:
            callbacks.append(
                ClassificationMetricsCallback(
                    eval_dataset=eval_dataset,
                    tokenizer=tokenizer,
                    max_new_tokens=32,
                    max_samples_per_lang=64,
                    decoding=config.generation_decoding,
                )
            )
        if task_type in (
            FinetuneTaskType.INSTRUCTION,
            FinetuneTaskType.NAMED_ENTITY_RECOGNITION,
            FinetuneTaskType.POS_TAGGING,
        ):
            callbacks.append(
                GenerationMetricsCallback(
                    eval_dataset=eval_dataset,
                    tokenizer=tokenizer,
                    max_new_tokens=64,
                    decoding=config.generation_decoding,
                )
            )

    if training_args.gradient_checkpointing and training_args.world_size > 1:
        callbacks.append(EnsureStaticGraphCallback())

    if hasattr(train_dataset, "set_epoch"):
        callbacks.append(_DatasetEpochCallback(train_dataset))

    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        processing_class=tokenizer,
    )

    trainer.processing_class = tokenizer

    return trainer
