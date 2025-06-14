import json
from pathlib import Path
from typing import Iterator
from tqdm import tqdm
import sys

def stream_training_data(file_path: Path) -> Iterator[str]:
    """
    Creates a memory-efficient generator that streams text from a large
    JSONL file.
    """
    if not file_path.exists():
        print(f"Error: Training file not found at {file_path}", file=sys.stderr)
        print("Please ensure you have run the data preparation script first.", file=sys.stderr)
        return

    with file_path.open("r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)

    with file_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=num_lines, desc="Streaming training data")):
            try:
                yield json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError) as e:
                error_message = f"""
                \n\n--- FATAL DATA INTEGRITY ERROR ---
                The tokenizer training was aborted due to a corrupted record in the training data.

                File:           {file_path}
                Line Number:    {i + 1}
                Error:          {e}
                Line Content:   {line.strip()}

                This indicates a bug in the upstream data preparation script. Please fix the
                source of this corrupted data before attempting to train the tokenizer again.
                """
                raise RuntimeError(error_message) from e