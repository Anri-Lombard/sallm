import torch
from pathlib import Path


def count_tokens_in_file(file_path):
    header = torch.from_file(str(file_path), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    return num_tokens


def count_dataset_tokens(pattern):
    files = sorted(Path(pattern).parent.glob(Path(pattern).name))
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return 0, 0

    total_tokens = 0
    for file in files:
        tokens = count_tokens_in_file(file)
        print(f"{file.name}: {tokens:,} tokens")
        total_tokens += tokens

    print(f"\nTotal tokens: {total_tokens:,}")
    print(f"Number of files: {len(files)}")
    print("-" * 50)
    return total_tokens, len(files)


# Count training tokens
print("Training Dataset:")
train_pattern = "./filtered_data_binary_custom_tokenizer/custom_train_*.bin"
train_tokens, train_files = count_dataset_tokens(train_pattern)

# Count validation tokens
print("\nValidation Dataset:")
val_pattern = "./filtered_data_binary_custom_tokenizer/custom_val_*.bin"
val_tokens, val_files = count_dataset_tokens(val_pattern)

print("\nSummary:")
print(f"Training tokens: {train_tokens:,}")
print(f"Validation tokens: {val_tokens:,}")
print(f"Total tokens: {train_tokens + val_tokens:,}")
