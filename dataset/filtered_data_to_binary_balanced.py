import os
import json
import argparse
import hashlib
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import tiktoken
import logging
import gc
from typing import Iterator, List, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LanguageNormalizer:
    LANGUAGE_MAPPINGS = {
        "af": "af",
        "afr": "af",
        "ns": "nso",
        "nso": "nso",
        "nr": "nr",
        "ss": "ssw",
        "ssw": "ssw",
        "st": "st",
        "sot": "st",
        "tn": "tsn",
        "tsn": "tsn",
        "ts": "tso",
        "tso": "tso",
        "ve": "ven",
        "ven": "ve",
        "xh": "xho",
        "xho": "xh",
        "zu": "zul",
        "zul": "zu",
    }

    @classmethod
    def normalize(cls, lang_code: str) -> str:
        return cls.LANGUAGE_MAPPINGS.get(lang_code.lower(), lang_code.lower())


class DataBalancer:
    def __init__(self, balance_factor: float = 15.0):
        self.balance_factor = balance_factor
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens["<|endoftext|>"]

    def write_shard(
        self, tokens: List[int], output_dir: str, split: str, shard_num: int
    ):
        filename = os.path.join(output_dir, f"balanced_{split}_{shard_num:06d}.bin")
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520
        header[1] = 1
        header[2] = len(tokens)

        tokens_np = np.array(tokens, dtype=np.uint16)
        with open(filename, "wb") as f:
            f.write(header.tobytes())
            f.write(tokens_np.tobytes())

        del tokens_np
        gc.collect()

    def count_tokens_in_file(self, filepath: str) -> Tuple[str, int]:
        normalized_lang = LanguageNormalizer.normalize(
            os.path.basename(filepath).split(".")[0]
        )
        total_chars = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        total_chars += len(text)
                except json.JSONDecodeError:
                    continue

        # Rough estimation of tokens
        estimated_tokens = total_chars // 4
        return normalized_lang, estimated_tokens

    def file_line_iterator(
        self, filepath: str, target_ratio: float, seed: int
    ) -> Iterator[Tuple[str, str, bool]]:
        """
        Iterator that yields (language, line, is_upsample) tuples.
        is_upsample indicates whether this is an upsampled line.
        """
        normalized_lang = LanguageNormalizer.normalize(
            os.path.basename(filepath).split(".")[0]
        )

        # Read all valid lines
        valid_lines = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    valid_lines.append(line)

        if not valid_lines:
            return

        # Create deterministic seed
        file_seed = (seed + hash(normalized_lang + filepath)) % (2**32 - 1)
        rng = np.random.RandomState(file_seed)

        if target_ratio <= 1.0:  # Downsampling case
            num_samples = int(len(valid_lines) * target_ratio)
            selected_indices = rng.choice(
                len(valid_lines), size=num_samples, replace=False
            )
            for idx in selected_indices:
                yield normalized_lang, valid_lines[idx], False
        else:  # Upsampling case
            # First yield all original lines
            for line in valid_lines:
                yield normalized_lang, line, False

            # Then yield additional upsampled lines
            additional_samples = int((target_ratio - 1) * len(valid_lines))
            if additional_samples > 0:
                upsampled_indices = rng.choice(
                    len(valid_lines), size=additional_samples, replace=True
                )
                for idx in upsampled_indices:
                    yield normalized_lang, valid_lines[idx], True

        del valid_lines
        gc.collect()

    def process_balanced_dataset(
        self,
        root_dir: str,
        output_dir: str,
        val_size: float,
        shard_size: int,
        seed: int,
    ):
        # Get list of files
        all_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".jsonl"):
                    all_files.append(os.path.join(root, file))

        logging.info(f"Found {len(all_files)} JSONL files")

        # First pass: count tokens
        logging.info("Counting tokens...")
        lang_token_counts = defaultdict(int)
        for filepath in tqdm(all_files, desc="Analyzing files"):
            lang, tokens = self.count_tokens_in_file(filepath)
            lang_token_counts[lang] += tokens

        # Calculate target size (15x the smallest non-zero dataset)
        min_tokens = min(count for count in lang_token_counts.values() if count > 0)
        target_tokens = min_tokens * self.balance_factor

        # Calculate sampling ratios (allowing upsampling)
        sampling_ratios = {
            lang: target_tokens / count if count > 0 else 0.0
            for lang, count in lang_token_counts.items()
        }

        # Print statistics
        print("\nDataset Statistics:")
        print(f"Target tokens per language: {target_tokens:,}")
        for lang, count in sorted(lang_token_counts.items(), key=lambda x: -x[1]):
            ratio = sampling_ratios[lang]
            final_tokens = int(count * ratio)
            action = "upsampled" if ratio > 1 else "downsampled"
            print(
                f"{lang}: {count:,} -> {final_tokens:,} tokens ({ratio:.2f}x {action})"
            )

        # Initialize output
        os.makedirs(output_dir, exist_ok=True)
        buffers = {"train": [], "val": []}
        shard_counts = {"train": 0, "val": 0}

        # Process files
        logging.info("Processing files...")
        for filepath in tqdm(all_files, desc="Processing files"):
            lang = LanguageNormalizer.normalize(
                os.path.basename(filepath).split(".")[0]
            )
            ratio = sampling_ratios[lang]

            if ratio == 0:
                continue

            for lang, line, is_upsample in self.file_line_iterator(
                filepath, ratio, seed
            ):
                try:
                    data = json.loads(line)
                    text = data["text"]

                    tokens = [self.eot]
                    tokens.extend(self.enc.encode_ordinary(text))
                    tokens.append(self.eot)
                    tokens.extend(self.enc.encode_ordinary(lang))
                    tokens.append(self.eot)

                    # Deterministic train/val split based on original (non-upsampled) data
                    if not is_upsample:
                        hash_str = line.encode()
                        hash_int = int(hashlib.sha256(hash_str).hexdigest(), 16)
                        split_seed = (seed + hash_int) % (2**32)
                        np.random.seed(split_seed)
                        is_val = np.random.rand() < val_size
                    # Upsampled data follows same split as original
                    split = "val" if is_val else "train"

                    buffers[split].extend(tokens)

                    # Write full shards
                    for split_type in ["train", "val"]:
                        while len(buffers[split_type]) >= shard_size:
                            shard_tokens = buffers[split_type][:shard_size]
                            self.write_shard(
                                shard_tokens,
                                output_dir,
                                split_type,
                                shard_counts[split_type],
                            )
                            buffers[split_type] = buffers[split_type][shard_size:]
                            shard_counts[split_type] += 1

                except Exception as e:
                    logging.warning(f"Error processing line in {filepath}: {str(e)}")
                    continue

                # Force garbage collection periodically
                if np.random.rand() < 0.01:  # 1% chance each iteration
                    gc.collect()

        # Write remaining tokens
        for split_type in ["train", "val"]:
            if buffers[split_type]:
                self.write_shard(
                    buffers[split_type],
                    output_dir,
                    split_type,
                    shard_counts[split_type],
                )


def main():
    parser = argparse.ArgumentParser(
        description="Process JSONL files into balanced tokenized binaries with train/val split."
    )
    parser.add_argument(
        "--root-dir", required=True, help="Directory containing JSONL files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for binary files"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.2, help="Validation set proportion"
    )
    parser.add_argument(
        "--shard-size", type=int, default=10**7, help="Tokens per shard"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        balancer = DataBalancer()
        balancer.process_balanced_dataset(
            args.root_dir, args.output_dir, args.val_size, args.shard_size, args.seed
        )
        print(
            f"\nProcessing complete! Balanced binary files saved to: {args.output_dir}"
        )

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
