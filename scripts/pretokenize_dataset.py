#!/usr/bin/env python3
"""Pre-tokenize the SALLM dataset to avoid distributed training timeouts."""
import logging
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    input_path = Path("data/sallm_dataset")
    output_path = Path("data/sallm_dataset_tokenized")
    tokenizer_path = Path("tokenizer/sallm_bpe_tokenizer")

    logger.info(f"Loading dataset from {input_path}")
    dataset = load_from_disk(str(input_path))
    logger.info(f"Dataset loaded: {dataset}")

    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    max_seq_length = 2048

    def tokenize_function(examples):
        """Tokenize the text column."""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    logger.info("Tokenizing dataset...")
    logger.info(f"Max sequence length: {max_seq_length}")

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=6,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    logger.info(f"Tokenized dataset: {tokenized_dataset}")

    logger.info(f"Saving tokenized dataset to {output_path}")
    tokenized_dataset.save_to_disk(str(output_path))

    logger.info("✓ Pre-tokenization complete!")
    logger.info(f"Tokenized dataset saved to: {output_path}")
    logger.info("You can now rsync this to the cluster")


if __name__ == "__main__":
    main()
