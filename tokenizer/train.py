import argparse
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import yaml

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


# TODO restructure library to use pydantic models for loading configs
def load_config(config_path: Path) -> Any:
    with config_path.open("r") as f:
        data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise TypeError("Config file is not a dictionary.")
        return data


def train_tokenizer(config: dict) -> None:
    """
    Orchestrate the tokenizer training process based on a config dict.
    """
    model_config = config["tokenizer_training"]["bpe_model"]
    path_config = config["tokenizer_training"]["paths"]

    vocab_size = model_config["vocab_size"]
    special_tokens = model_config["special_tokens"]
    train_dir = Path(path_config["train_data_file"])
    output_dir = Path(path_config["output_file"])

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # TODO these necessary/useful according to research?
    tokenizer.normalizer = Sequence([NFD()])

    # Sentencepiece is language agnostic, this is an attempt to mimic their behaviour
    # without needing to change libraries. Tokens will now be bytes which has been
    # shown to help with the morphologically rich languages we're working with
    tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )

    print(f"Loading dataset from {train_dir}...")
    if not train_dir.exists():
        raise FileNotFoundError(f"Training data directory not found at: {train_dir}")

    hf_dataset = datasets.load_from_disk(str(train_dir))

    def batch_iterator(batch_size: int = 1000) -> Iterator[list[str]]:
        for i in range(0, len(hf_dataset), batch_size):
            yield hf_dataset[i : i + batch_size]["text"]

    print("Starting tokenizer training...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    print("Training complete.")

    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")

    # TODO this template correct?
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] [BOS] $B [EOS]",
        special_tokens=[
            ("[BOS]", bos_token_id),
            ("[EOS]", eos_token_id),
        ],
    )

    print("Wrapping tokenizer for Hugging Face compatibility...")
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cast(Any, wrapped_tokenizer).save_pretrained(str(output_dir))
    print(f"Tokenizer saved to {output_dir}")

    print("\n--- Verification ---")
    reloaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(str(output_dir))
    text = "Kunjani? This is a test for isiZulu & English."
    encoded = reloaded_tokenizer.encode(text)

    print(f"Original: {text}")
    print(f"Encoded (tokens): {reloaded_tokenizer.convert_ids_to_tokens(encoded)}")
    print(f"Encoded (IDs): {encoded}")
    decoded = reloaded_tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"Decoded: {decoded}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tokenizers/bpe.yaml",
        help="Path to the tokenizer training configuration YAML file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    print(f"Loading configuration from: {config_path}")
    config_data = load_config(config_path)
    train_tokenizer(config_data)
