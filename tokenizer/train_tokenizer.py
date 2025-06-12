import os
import json
import argparse
from typing import List, Iterator
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing


def get_training_corpus(root_dir: str) -> Iterator[List[str]]:
    """Yields batches of text from JSONL files for training."""
    batch_size = 1000
    current_batch = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jsonl"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            text = data.get("text", "").strip()
                            if text:
                                current_batch.append(text)
                                if len(current_batch) >= batch_size:
                                    yield current_batch
                                    current_batch = []
                        except json.JSONDecodeError:
                            continue

    if current_batch:  # Yield any remaining texts
        yield current_batch


def train_tokenizer(args):
    """Train a BPE tokenizer with similar settings to GPT-2."""
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<|UNKNOWN|>"))

    # Configure pre-tokenization and decoder
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()

    # Configure special tokens and trainer
    trainer = BpeTrainer(
        vocab_size=50257,  # Same as GPT-2
        min_frequency=2,
        special_tokens=[
            "<|endoftext|>",
            "<|UNKNOWN|>",
            "<|START|>",
            "<|END|>",
        ],
        show_progress=True,
    )

    # Train the tokenizer first so we can get token IDs
    print(f"Training tokenizer on files in {args.root_dir}...")
    tokenizer.train_from_iterator(get_training_corpus(args.root_dir), trainer)

    # Now configure post-processor with the trained token IDs
    start_token_id = tokenizer.token_to_id("<|START|>")
    end_token_id = tokenizer.token_to_id("<|END|>")

    if start_token_id is not None and end_token_id is not None:
        tokenizer.post_processor = TemplateProcessing(
            single="<|START|> $A <|END|>",
            special_tokens=[
                ("<|START|>", start_token_id),
                ("<|END|>", end_token_id),
            ],
        )

    # Save the tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(output_path)
    print(f"\nTokenizer training complete! Saved to: {output_path}")

    # Print some statistics
    vocab_size = tokenizer.get_vocab_size()
    print(f"\nVocabulary size: {vocab_size}")

    # Test the tokenizer on a sample text
    if args.test_text:
        print("\nTesting tokenizer on sample text:")
        encoded = tokenizer.encode(args.test_text)
        decoded = tokenizer.decode(encoded.ids)
        print(f"Original: {args.test_text}")
        print(f"Encoded: {encoded.tokens}")
        print(f"Decoded: {decoded}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a custom BPE tokenizer similar to GPT-2"
    )
    parser.add_argument(
        "--root-dir", required=True, help="Directory containing JSONL files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for tokenizer files"
    )
    parser.add_argument(
        "--test-text",
        default="Hello, world! This is a test.",
        help="Sample text to test the trained tokenizer",
    )

    args = parser.parse_args()
    train_tokenizer(args)


if __name__ == "__main__":
    main()
