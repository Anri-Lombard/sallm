import os
import json
import argparse
import hashlib
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import tiktoken


def write_datafile(filename, tokens):
    """Write tokens to binary file with custom header format."""
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # Magic number
    header[1] = 1  # Version
    header[2] = len(tokens)  # Token count

    tokens_np = np.array(tokens, dtype=np.uint16)
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_np.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="Process JSONL files into tokenized binaries with train/val split."
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
    args = parser.parse_args()

    # First pass: collect language statistics and total count
    lang_counts = defaultdict(int)
    total_lines = 0
    print("Counting lines and languages...")
    for root, _, files in os.walk(args.root_dir):
        for file in files:
            if file.endswith(".jsonl"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    for line in f:
                        total_lines += 1
                        try:
                            data = json.loads(line.strip())
                            lang_counts[data.get("language", "")] += 1
                        except json.JSONDecodeError:
                            continue

    # Initialize tokenizer and split buffers
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize shard buffers and counters
    buffers = {"train": [], "val": []}
    shard_counts = {"train": 0, "val": 0}

    # Second pass: process and split data
    print("\nProcessing files and generating splits:")
    with tqdm(total=total_lines, desc="Processing", unit="line") as pbar:
        for root, _, files in os.walk(args.root_dir):
            for file in files:
                if file.endswith(".jsonl"):
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                pbar.update(1)
                                continue

                            try:
                                data = json.loads(line)
                                text = data.get("text", "")
                                lang = data.get("language", "")

                                # Deterministic split using hashed content
                                hash_str = f"{lang}_{text}".encode()
                                hash_int = int(hashlib.sha256(hash_str).hexdigest(), 16)
                                valid_seed = (args.seed + hash_int) % (2**32)
                                np.random.seed(valid_seed)
                                split = (
                                    "val"
                                    if np.random.rand() < args.val_size
                                    else "train"
                                )

                                # Tokenize with EOT markers
                                tokens = [eot]
                                tokens.extend(enc.encode_ordinary(text))
                                tokens.append(eot)
                                tokens.extend(enc.encode_ordinary(lang))
                                tokens.append(eot)

                                # Add to buffer
                                buffers[split].extend(tokens)

                                # Write shard if buffer full
                                for split_type in ["train", "val"]:
                                    while len(buffers[split_type]) >= args.shard_size:
                                        shard_tokens = buffers[split_type][
                                            : args.shard_size
                                        ]
                                        filename = os.path.join(
                                            args.output_dir,
                                            f"custom_{split_type}_{shard_counts[split_type]:06d}.bin",
                                        )
                                        write_datafile(filename, shard_tokens)
                                        buffers[split_type] = buffers[split_type][
                                            args.shard_size :
                                        ]
                                        shard_counts[split_type] += 1

                                pbar.update(1)
                            except json.JSONDecodeError:
                                pbar.update(1)
                                continue

    # Write remaining tokens
    for split_type in ["train", "val"]:
        if buffers[split_type]:
            filename = os.path.join(
                args.output_dir,
                f"custom_{split_type}_{shard_counts[split_type]:06d}.bin",
            )
            write_datafile(filename, buffers[split_type])

    print(f"\nProcessing complete! Binary files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
