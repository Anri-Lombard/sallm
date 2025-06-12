import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from tokenizers import Tokenizer

# Language code mappings remain the same
LANG_MAPPINGS = {
    "af": "afr",
    "afr": "afr",
    "xh": "xho",
    "xho": "xho",
    "zu": "zul",
    "zul": "zul",
    "ns": "nso",
    "nso": "nso",
    "st": "sot",
    "sot": "sot",
    "ss": "ssw",
    "ssw": "ssw",
    "tn": "tsn",
    "tsn": "tsn",
    "ts": "tso",
    "tso": "tso",
    "ve": "ven",
    "ven": "ven",
    "en": "eng",
    "nr": "nbl",
}


def normalize_language_code(file_path, file_field):
    """Extract and normalize language code from file path and file field"""
    if file_field:
        base_code = file_field.lower().split(".")[0]
        if base_code in LANG_MAPPINGS:
            return LANG_MAPPINGS[base_code]
    filename = os.path.basename(file_path)
    base_code = filename.split(".")[0].lower()
    return LANG_MAPPINGS.get(base_code, "unknown")


def print_distribution(counts, total_lines, description=""):
    """Print distribution of languages in the dataset"""
    print(f"\n{description}")
    for lang, count in sorted(counts.items(), key=lambda x: -x[1]):
        percentage = (count / total_lines) * 100
        print(f"{lang}: {count} ({percentage:.2f}%)")


def consolidate_data(root_dir, output_file, target_percentage=0.25):
    """Consolidate all JSONL files into a single file with downsampling"""
    initial_lang_counts = defaultdict(int)
    total_lines = 0

    # First pass: count lines and languages
    print("Counting lines and languages...")
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        total_lines += 1
                        try:
                            data = json.loads(line.strip())
                            lang = normalize_language_code(file_path, data.get("file"))
                            initial_lang_counts[lang] += 1
                        except json.JSONDecodeError:
                            continue

    # Calculate target counts for Afrikaans and English
    other_languages_count = sum(
        count
        for lang, count in initial_lang_counts.items()
        if lang not in ["afr", "eng"]
    )
    denominator = 1 - 2 * target_percentage
    afr_target_count = (
        int(other_languages_count * target_percentage / denominator)
        if denominator != 0
        else 0
    )
    eng_target_count = (
        int(other_languages_count * target_percentage / denominator)
        if denominator != 0
        else 0
    )

    # Second pass: write consolidated file with downsampling
    current_afr_count = 0
    current_eng_count = 0
    final_lang_counts = defaultdict(int)

    print("\nConsolidating files...")
    with open(output_file, "w", encoding="utf-8") as outf:
        with tqdm(total=total_lines, desc="Processing", unit="line") as pbar:
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file.endswith(".jsonl"):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    data = json.loads(line.strip())
                                    text = data.get("text", "").strip()
                                    lang = normalize_language_code(
                                        file_path, data.get("file")
                                    )

                                    if not text:
                                        pbar.update(1)
                                        continue

                                    # Apply downsampling for Afrikaans and English
                                    if lang == "afr":
                                        if current_afr_count >= afr_target_count:
                                            pbar.update(1)
                                            continue
                                        current_afr_count += 1
                                    elif lang == "eng":
                                        if current_eng_count >= eng_target_count:
                                            pbar.update(1)
                                            continue
                                        current_eng_count += 1

                                    # Write to consolidated file
                                    output_data = {
                                        "text": text,
                                        "lang": lang,
                                        "file": data.get("file"),
                                    }
                                    outf.write(json.dumps(output_data) + "\n")
                                    final_lang_counts[lang] += 1

                                except json.JSONDecodeError:
                                    pass
                                pbar.update(1)

    # Print distributions
    total_initial = sum(initial_lang_counts.values())
    total_final = sum(final_lang_counts.values())

    print_distribution(
        initial_lang_counts, total_initial, "Distribution BEFORE downsampling:"
    )
    print_distribution(
        final_lang_counts, total_final, "Distribution AFTER downsampling:"
    )

    return final_lang_counts


def split_and_shuffle_data(input_file, output_dir, val_size, seed):
    """Split data into train/val/test and shuffle training data"""
    os.makedirs(output_dir, exist_ok=True)

    # Count total lines and collect line offsets
    print("Collecting line offsets...")
    line_offsets = []
    with open(input_file, "rb") as f:
        offset = 0
        for line in tqdm(f):
            line_offsets.append(offset)
            offset += len(line)

    total_lines = len(line_offsets)

    # Group offsets by language
    print("Grouping by language...")
    lang_offsets = defaultdict(list)
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f)):
            data = json.loads(line)
            lang_offsets[data["lang"]].append(line_offsets[i])

    # Split and write data
    print("Splitting and writing data...")
    train_offsets = []

    # Counters for split distributions
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)

    # Create language-specific validation and test files
    for lang, offsets in lang_offsets.items():
        np.random.seed(seed)
        indices = np.random.permutation(len(offsets))

        val_count = int(len(offsets) * val_size)

        val_indices = indices[:val_count]
        train_indices = indices[val_count:]

        # Write validation data (unshuffled)
        split_name, split_indices, counts_dict = "val", val_indices, val_counts
        with (
            open(input_file, "rb") as inf,
            open(os.path.join(output_dir, f"{split_name}_{lang}.jsonl"), "wb") as outf,
        ):
            for idx in split_indices:
                inf.seek(line_offsets[idx])
                line = inf.readline()
                outf.write(line)
                counts_dict[lang] += 1

        # Collect training offsets
        for idx in train_indices:
            train_offsets.append(line_offsets[idx])

    # Shuffle and write training data
    print("Shuffling and writing training data...")
    np.random.seed(seed)
    np.random.shuffle(train_offsets)

    with (
        open(input_file, "rb") as inf,
        open(os.path.join(output_dir, "train.jsonl"), "wb") as outf,
    ):
        for offset in tqdm(train_offsets):
            inf.seek(offset)
            line = inf.readline()
            data = json.loads(line)
            train_counts[data["lang"]] += 1
            outf.write(line)

    # Print split distributions
    print_distribution(
        train_counts, sum(train_counts.values()), "Distribution in TRAIN split:"
    )
    print_distribution(
        val_counts, sum(val_counts.values()), "Distribution in VALIDATION split:"
    )


def process_text(text, lang, tokenizer):
    """Process a single text entry with explicit language marking"""
    eot_token = tokenizer.token_to_id("<|endoftext|>")
    if eot_token is None:
        raise ValueError("Could not find <|endoftext|> token in tokenizer vocabulary")

    tokens = []
    tokens.append(eot_token)
    encoded = tokenizer.encode(text)
    tokens.extend(encoded.ids)
    tokens.append(eot_token)
    lang_marker = tokenizer.encode(f"<{lang}>")
    tokens.extend(lang_marker.ids)
    tokens.append(eot_token)
    return tokens


def write_datafile(filename, tokens):
    """Write tokens to binary file with header"""
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # Magic number
    header[1] = 1  # Version
    header[2] = len(tokens)  # Token count
    tokens_np = np.array(tokens, dtype=np.uint16)
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_np.tobytes())


def write_sharded_data(input_file, prefix, output_dir, shard_size, tokenizer):
    """Write data to sharded binary files"""
    shard_count = 0
    current_shard_tokens = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Processing {prefix}"):
            data = json.loads(line)
            tokens = process_text(data["text"], data["lang"], tokenizer)

            current_shard_tokens.extend(tokens)

            if len(current_shard_tokens) >= shard_size:
                filename = os.path.join(output_dir, f"{prefix}_{shard_count:06d}.bin")
                write_datafile(filename, current_shard_tokens)
                current_shard_tokens = []
                shard_count += 1

    # Write final shard if there are remaining tokens
    if current_shard_tokens:
        filename = os.path.join(output_dir, f"{prefix}_{shard_count:06d}.bin")
        write_datafile(filename, current_shard_tokens)
        shard_count += 1

    return shard_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir", required=True, help="Directory containing JSONL files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for binary files"
    )
    parser.add_argument(
        "--tokenizer-path", required=True, help="Path to your trained tokenizer.json"
    )
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--shard-size", type=int, default=10**7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temp-dir", default="temp", help="Directory for temporary files"
    )
    args = parser.parse_args()

    # Create temporary directory
    os.makedirs(args.temp_dir, exist_ok=True)
    consolidated_file = os.path.join(args.temp_dir, "consolidated.jsonl")

    # Step 1: Consolidate all data into a single file
    print("Step 1: Consolidating data...")
    final_lang_counts = consolidate_data(args.root_dir, consolidated_file)

    # Step 2: Split and shuffle data
    print("\nStep 2: Splitting and shuffling data...")
    split_and_shuffle_data(consolidated_file, args.temp_dir, args.val_size, args.seed)

    # Step 3: Process and write sharded binary files
    print("\nStep 3: Creating binary shards...")
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # Process training data
    train_shards = write_sharded_data(
        os.path.join(args.temp_dir, "train.jsonl"),
        "custom_train",
        args.output_dir,
        args.shard_size,
        tokenizer,
    )

    # Process validation data for each language
    val_shards = {}
    for lang in final_lang_counts.keys():
        val_file = os.path.join(args.temp_dir, f"val_{lang}.jsonl")

        if os.path.exists(val_file):
            val_shards[lang] = write_sharded_data(
                val_file,
                f"custom_val_{lang}",
                args.output_dir,
                args.shard_size,
                tokenizer,
            )

    # Print final statistics
    print("\nProcessing complete!")
    print(f"Binary files saved to: {args.output_dir}")
    print(f"Total training shards: {train_shards}")
    print("\nValidation shards per language:")
    for lang, count in val_shards.items():
        print(f"{lang}: {count} shards")

    # Cleanup temporary files
    if os.path.exists(args.temp_dir):
        print(f"\nCleaning up temporary files in {args.temp_dir}...")
        for file in os.listdir(args.temp_dir):
            os.remove(os.path.join(args.temp_dir, file))
        os.rmdir(args.temp_dir)


if __name__ == "__main__":
    main()
