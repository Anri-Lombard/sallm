from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

from sallm.config import ExperimentConfig
from sallm.training.callbacks import PerLanguageEvalCallback
from sallm.utils import compute_metrics


def build_trainer(
    config: ExperimentConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> Trainer:
    training_args = TrainingArguments(**config.training.model_dump())
    callbacks = [PerLanguageEvalCallback()]

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
