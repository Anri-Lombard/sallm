from typing import Any
import yaml
import argparse
from pathlib import Path
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing

from .dataset import stream_training_data


# TODO restructure library to use pydantic models for loading configs
def load_config(config_path: Path) -> Any:
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def train_tokenizer(config: dict) -> None:
    """
    Orchestrate the tokenizer training process based on a config dict.
    """
    model_config = config["tokenizer_training"]["bpe_model"]
    path_config = config["tokenizer_training"]["paths"]

    vocab_size = model_config["vocab_size"]
    special_tokens = model_config["special_tokens"]
    train_file = Path(path_config["train_data_file"])
    output_file = Path(path_config["output_file"])

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # TODO is this needed?
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=False,
    )

    print("Starting tokenizer training...")
    data_iterator = stream_training_data(train_file)
    tokenizer.train_from_iterator(data_iterator, trainer=trainer)
    print("Training complete.")

    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] [BOS] $B [EOS]",
        special_tokens=[
            ("[BOS]", bos_token_id),
            ("[EOS]", eos_token_id),
        ],
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_file))
    print(f"Tokenizer saved to {output_file}")

    print("\n--- Verification ---")
    reloaded_tokenizer = Tokenizer.from_file(str(output_file))
    text = "Kunjani? This is a test for isiZulu & English."
    encoded = reloaded_tokenizer.encode(text)

    print(f"Original: {text}")
    print(f"Encoded (tokens): {encoded.tokens}")
    print(f"Encoded (IDs): {encoded.ids}")
    decoded = reloaded_tokenizer.decode(encoded.ids)
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
