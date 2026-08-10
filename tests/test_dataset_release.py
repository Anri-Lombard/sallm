import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from data.prepare_datasets import DataProcessor, FilePointer  # noqa: E402


def test_release_manifest_rejects_missing_source(tmp_path: Path) -> None:
    processor = DataProcessor.__new__(DataProcessor)
    processor.root_dir = tmp_path
    processor.expected_source_files = ["wura/en.jsonl"]

    with pytest.raises(ValueError, match="wura/en.jsonl"):
        processor._validate_source_files()


def test_release_split_sizes_are_exact() -> None:
    processor = DataProcessor.__new__(DataProcessor)
    processor.expected_split_sizes = {"train": 1, "validation": 1, "test": 1}
    pointer = FilePointer(Path("source.jsonl"), 0, 1)

    with pytest.raises(ValueError, match="expected"):
        processor._validate_split_sizes(
            {"train": [pointer], "validation": [pointer], "test": []}
        )
