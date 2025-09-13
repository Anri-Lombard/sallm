import json
import sys
from collections.abc import Iterator
from pathlib import Path

from tqdm import tqdm


def stream_training_data(file_path: Path) -> Iterator[str]:
    """
    Creates a memory-efficient generator that streams text from a large
    JSONL file.
    """
    if not file_path.exists():
        print(f"Error: Training file not found at {file_path}", file=sys.stderr)
        print(
            "Please ensure you have run the data preparation script first.",
            file=sys.stderr,
        )
        return

    with file_path.open("r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)

    with file_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(
            tqdm(f, total=num_lines, desc="Streaming training data")
        ):
            try:
                yield json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError) as e:
                error_message = (
                    "\n\n--- FATAL DATA INTEGRITY ERROR ---\n"
                    "The tokenizer training was aborted due to a corrupted record "
                    "in the training data.\n\n"
                    f"File:           {file_path}\n"
                    f"Line Number:    {i + 1}\n"
                    f"Error:          {e}\n"
                    f"Line Content:   {line.strip()}\n\n"
                    "This indicates a bug in the upstream data preparation script. "
                    "Please fix the source of this corrupted data before attempting "
                    "to train the tokenizer again."
                )
                raise RuntimeError(error_message) from e
