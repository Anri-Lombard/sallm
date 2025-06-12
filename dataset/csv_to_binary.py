import os
import argparse
import numpy as np
import pandas as pd
import tiktoken
from tqdm import tqdm


def write_datafile(filename, toks):
    """
    Saves token data as a .bin file, matching the format expected by the training script.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large"  # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic number matching the training script
    header[1] = 1  # version
    header[2] = len(toks)  # number of tokens

    # Convert tokens to numpy array if needed
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        maxtok = 2**16
        assert all(
            0 <= t < maxtok for t in toks
        ), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks

    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


def process_csv(input_file, output_dir, split_name, shard_size=10**7):
    """Process a single CSV file and convert it to binary format"""
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize token collection
    current_shard = []
    shard_index = 0

    # Process each row
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        # Format: <|endoftext|>text<|endoftext|>language<|endoftext|>
        tokens = []
        tokens.append(eot)  # Start with EOT
        tokens.extend(enc.encode_ordinary(str(row["text"])))  # Text content
        tokens.append(eot)  # EOT between text and language
        tokens.extend(enc.encode_ordinary(str(row["language"])))  # Language
        tokens.append(eot)  # End with EOT

        # Add to current shard
        current_shard.extend(tokens)

        # If shard is full, write it
        if len(current_shard) >= shard_size:
            filename = os.path.join(
                output_dir, f"custom_{split_name}_{shard_index:06d}.bin"
            )
            write_datafile(filename, current_shard[:shard_size])
            # Keep remainder for next shard
            current_shard = current_shard[shard_size:]
            shard_index += 1

    # Write final shard if there are remaining tokens
    if current_shard:
        filename = os.path.join(
            output_dir, f"custom_{split_name}_{shard_index:06d}.bin"
        )
        write_datafile(filename, current_shard)


def main():
    parser = argparse.ArgumentParser(
        description="Convert headline generation CSV datasets to binary format for GPT training"
    )
    parser.add_argument("--train", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--val", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--test", type=str, required=False, help="Path to test CSV")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for binary files",
    )
    parser.add_argument(
        "--shard-size", type=int, default=10**7, help="Size of each shard in tokens"
    )

    args = parser.parse_args()

    # Process train and val splits
    process_csv(args.train, args.output_dir, "train", args.shard_size)
    process_csv(args.val, args.output_dir, "val", args.shard_size)

    # Only process test split if provided
    if args.test:
        process_csv(args.test, args.output_dir, "test", args.shard_size)


if __name__ == "__main__":
    main()
