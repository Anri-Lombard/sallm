import logging
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from sallm.fine_tune.metrics import classification_metrics

from sallm.config import ExperimentConfig, TaskType
from sallm.training.callbacks import ShowCompletionsCallback

logger = logging.getLogger(__name__)


def build_trainer(
    config: ExperimentConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> SFTTrainer:
    eos_id = tokenizer.eos_token_id

    def _strip_eos(batch):
        inputs = batch["input_ids"]
        labels = batch.get("labels", inputs)
        new_inputs, new_labels = [], []
        for inp, lab in zip(inputs, labels):
            if inp and inp[-1] == eos_id:
                inp = inp[:-1]
            if lab and lab[-1] == eos_id:
                lab = lab[:-1]
            new_inputs.append(inp)
            new_labels.append(lab)
        return {"input_ids": new_inputs, "labels": new_labels}

    train_dataset = train_dataset.map(
        _strip_eos,
        batched=True,
        remove_columns=[
            col
            for col in train_dataset.column_names
            if col not in ("input_ids", "labels")
        ],
    )
    eval_dataset = eval_dataset.map(
        _strip_eos,
        batched=True,
        remove_columns=[
            col
            for col in eval_dataset.column_names
            if col not in ("input_ids", "labels")
        ],
    )

    training_args_dict = OmegaConf.to_container(config.training, resolve=True)

    training_args = SFTConfig(
        **training_args_dict,
        max_seq_length=config.dataset.max_seq_length if config.dataset else None,
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

    compute_metrics = None
    if config.dataset and config.dataset.task_type == TaskType.CLASSIFICATION:
        compute_metrics = classification_metrics

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[completions_callback],
    )

    trainer.tokenizer = tokenizer

    return trainer
