import torch
import tiktoken
from collections import defaultdict


def extract_language_and_text(tokens, enc):
    """Extract language and text from token sequence between EOTs"""
    eot_positions = (tokens == 50256).nonzero().flatten().tolist()
    sequences = []
    for i in range(0, len(eot_positions) - 2, 3):
        if i + 2 >= len(eot_positions):
            break
        text_start = eot_positions[i] + 1
        text_end = eot_positions[i + 1]
        lang_start = eot_positions[i + 1] + 1
        lang_end = eot_positions[i + 2]
        if text_end > text_start and lang_end > lang_start:
            text = enc.decode(tokens[text_start:text_end].tolist())
            lang = enc.decode(tokens[lang_start:lang_end].tolist())
            # Validate language format
            if lang.startswith("<") and lang.endswith(">"):
                sequences.append((text, lang))
            else:
                print(f"Warning: Invalid language format found: {lang}")
    return sequences


def inspect_binary_file(file_path, num_examples=5):
    # Load the tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Read the header
    header = torch.from_file(file_path, False, 256, dtype=torch.int32)
    print("Header info:")
    print(f"Magic number: {header[0]}")
    print(f"Version: {header[1]}")
    print(f"Number of tokens: {header[2]}")
    print("-" * 80)

    # Read the tokens
    tokens = torch.empty(header[2], dtype=torch.uint16)
    with open(file_path, "rb") as f:
        f.seek(256 * 4)
        f.readinto(tokens.numpy())

    # Find EOT patterns
    eot_positions = (tokens == 50256).nonzero().flatten().tolist()
    print(f"Number of EOT tokens found: {len(eot_positions)}")
    print(f"First few EOT positions: {eot_positions[:10]}")
    print("-" * 80)

    # Analyze distances between EOT tokens
    if len(eot_positions) > 1:
        distances = [
            eot_positions[i + 1] - eot_positions[i]
            for i in range(len(eot_positions) - 1)
        ]
        print("Distances between consecutive EOT tokens:")
        print(f"Min: {min(distances)}")
        print(f"Max: {max(distances)}")
        print(f"First few distances: {distances[:10]}")
        print("-" * 80)

    # Extract and display examples
    print(f"\nFirst {num_examples} examples:")
    sequences = extract_language_and_text(tokens, enc)

    language_counts = defaultdict(int)
    for i, (text, lang) in enumerate(sequences):
        language_counts[lang] += 1
        if i < num_examples:
            print(f"\nExample {i + 1}:")
            print(f"Language: {lang}")
            print(f"Text preview: {text[:200]}...")
            print("-" * 40)

    # Display language statistics
    print("\nLanguage distribution:")
    total_sequences = len(sequences)
    for lang, count in sorted(language_counts.items(), key=lambda x: -x[1]):
        percentage = (count / total_sequences * 100) if total_sequences > 0 else 0
        print(f"{lang}: {count} ({percentage:.2f}%)")


if __name__ == "__main__":
    file_path = "./filtered_data_binary/custom_val_000000.bin"
    inspect_binary_file(file_path)
