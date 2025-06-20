from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from datasets import Dataset

from sallm.config import ExperimentConfig
from sallm.training.trainer import CustomTrainer


def build_trainer(
    config: ExperimentConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> CustomTrainer:
    training_args = TrainingArguments(**config.training)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
