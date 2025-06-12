import torch
import wandb
from transformers import PreTrainedTokenizerFast, DataCollatorForTokenClassification
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import List

from sallm_finetuner.config import AppConfig, DataConfig, get_config
from sallm_finetuner.data.processing import (
    load_and_combine_datasets,
    create_tokenization_function,
    process_dataset,
)
from sallm_finetuner.models.heads import GPTForTokenClassification
from sallm_finetuner.training.trainer import FineTuner
from sallm_finetuner.training.metrics import MetricsComputer


@dataclass
class NerDataConfig(DataConfig):
    language_subsets: List[str] = field(default_factory=lambda: ["tsn", "xho", "zul"])
    test_language_subsets: List[str] = field(
        default_factory=lambda: ["tsn", "xho", "zul"]
    )
    ner_tags: List[str] = field(
        default_factory=lambda: [
            "O",
            "B-PER",
            "I-PER",
            "B-ORG",
            "I-ORG",
            "B-LOC",
            "I-LOC",
            "B-DATE",
            "I-DATE",
        ]
    )


@dataclass
class NerAppConfig(AppConfig):
    data: NerDataConfig = field(default_factory=NerDataConfig)


def main():
    config = get_config(NerAppConfig)

    wandb.init(
        project=config.training.wandb_project,
        entity=config.training.wandb_entity,
        config=asdict(config),
    )
    if config.hpo:
        config.update_from_wandb(wandb.config)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.model.tokenizer_path)
    tokenizer.pad_token = "<|endoftext|>"
    config.model.vocab_size = tokenizer.vocab_size

    label_to_id = {label: i for i, label in enumerate(config.data.ner_tags)}
    tokenization_fn = create_tokenization_function(
        tokenizer, label_to_id, config.data.max_seq_length, config.data.pad_token_id
    )

    raw_datasets = load_and_combine_datasets(
        config.data.dataset_name, config.data.language_subsets
    )
    processed_datasets = process_dataset(raw_datasets, tokenization_fn)

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding="longest"
    )
    train_dl = DataLoader(
        processed_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=config.training.batch_size,
    )
    val_dl = DataLoader(
        processed_datasets["validation"],
        collate_fn=data_collator,
        batch_size=config.training.batch_size,
    )
    test_dl = DataLoader(
        processed_datasets["test"],
        collate_fn=data_collator,
        batch_size=config.training.batch_size,
    )

    model = GPTForTokenClassification(config)

    id_to_label = {i: label for i, label in enumerate(config.data.ner_tags)}
    metrics_computer = MetricsComputer(id_to_label, config.data.pad_token_id)
    metrics_computer.tokenization_function = tokenization_fn

    tuner = FineTuner(
        config, model, tokenizer, train_dl, val_dl, test_dl, metrics_computer
    )

    best_model_path = tuner.train()
    tuner.final_evaluation(best_model_path)

    print("\nFine-tuning complete.")
    wandb.finish()


if __name__ == "__main__":
    main()
