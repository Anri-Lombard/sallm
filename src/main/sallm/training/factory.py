from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

from sallm.config import ExperimentConfig
from sallm.utils import compute_metrics


def build_trainer(
    config: ExperimentConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> Trainer:
    """Builds and returns a Hugging Face Trainer instance."""
    training_args = TrainingArguments(**config.training.model_dump())

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
