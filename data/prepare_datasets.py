# TODO split into modular files as part of restructuring
import json
import argparse
import random
import yaml
from pathlib import Path
from collections import defaultdict
from itertools import groupby
from typing import List, Dict, NamedTuple, Generator, Any
from tqdm import tqdm
import datasets
from datasets import DatasetDict


class FilePointer(NamedTuple):
    file_path: Path
    offset: int
    char_count: int


class DataProcessor:
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

    def __init__(self, config: dict) -> None:
        paths = config["paths"]
        splits = config["splits"]
        settings = config["settings"]

        self.root_dir = Path(paths["source_directory"])
        self.output_dir = Path(paths["output_directory"])

        self.val_size = splits["validation_size"]
        self.test_size = splits["test_size"]
        self.split_cap_tokens = splits["capping"]["max_tokens"]
        self.chars_per_token_estimate = splits["capping"]["chars_per_token_estimate"]

        self.seed = settings["seed"]
        random.seed(self.seed)

    def run(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Source directory not found: {self.root_dir}\n"
                "Please update the 'source_directory' path in your config file."
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        language_indexes = self._build_index()
        split_pointers = self._calculate_splits(language_indexes)
        self._create_and_save_splits(split_pointers)

        print("\nProcessing complete.")
        print(f"Final dataset saved to: {self.output_dir}")

    def _prepare_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for split in ["train", "validation", "test"]:
            (self.output_dir / split).mkdir(exist_ok=True)

    def _normalize_lang_code(self, file_path: Path, file_field: str) -> str:
        if file_field:
            base_code = file_field.lower().split(".")[0]
            if base_code in self.LANG_MAPPINGS:
                return self.LANG_MAPPINGS[base_code]
        filename = file_path.name
        base_code = filename.split(".")[0].lower()
        return self.LANG_MAPPINGS.get(base_code, "unknown")

    def _build_index(self) -> Dict[str, List[FilePointer]]:
        print("Phase 1: Indexing source files...")
        language_indexes = defaultdict(list)
        jsonl_files = list(self.root_dir.rglob("*.jsonl"))

        for file_path in tqdm(jsonl_files, desc="Indexing files"):
            with file_path.open("rb") as f:
                offset = 0
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        lang = self._normalize_lang_code(
                            file_path, data.get("file", "")
                        )
                        if text and lang != "unknown":
                            pointer = FilePointer(file_path, offset, len(text))
                            language_indexes[lang].append(pointer)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                    offset += len(line)
        return language_indexes

    def _calculate_splits(
        self, language_indexes: Dict[str, List[FilePointer]]
    ) -> Dict[str, List[FilePointer]]:
        print("Phase 2: Calculating splits with capping...")
        split_pointers = defaultdict(list)

        for lang, pointers in tqdm(language_indexes.items(), desc="Calculating splits"):
            random.shuffle(pointers)

            n_total = len(pointers)
            n_val = int(self.val_size * n_total)
            n_test = int(self.test_size * n_total)

            lang_test_pointers = pointers[:n_test]
            lang_val_pointers = pointers[n_test : n_test + n_val]
            lang_train_pointers = pointers[n_test + n_val :]

            for split_name, lang_split_pointers in [
                ("validation", lang_val_pointers),
                ("test", lang_test_pointers),
            ]:
                current_char_count = 0
                capped_pointers = []
                spillover_pointers = []

                for pointer in lang_split_pointers:
                    estimated_tokens = (
                        current_char_count + pointer.char_count
                    ) / self.chars_per_token_estimate
                    if estimated_tokens < self.split_cap_tokens:
                        capped_pointers.append(pointer)
                        current_char_count += pointer.char_count
                    else:
                        spillover_pointers.append(pointer)

                split_pointers[split_name].extend(capped_pointers)
                lang_train_pointers.extend(spillover_pointers)

            split_pointers["train"].extend(lang_train_pointers)

        return split_pointers

    def _generate_samples(
        self, pointers: List[FilePointer]
    ) -> Generator[Dict[str, Any], None, None]:
        pointers.sort(key=lambda p: (p.file_path, p.offset))

        for file_path, group in groupby(pointers, key=lambda p: p.file_path):
            with file_path.open("rb") as in_f:
                for pointer in group:
                    in_f.seek(pointer.offset)
                    line = in_f.readline()
                    data = json.loads(line)
                    lang = self._normalize_lang_code(
                        pointer.file_path, data.get("file", "")
                    )
                    yield {"text": data["text"], "lang": lang}

    def _create_and_save_splits(
        self, split_pointers: Dict[str, List[FilePointer]]
    ) -> None:
        print("Phase 3: Generating and saving Hugging Face DatasetDict...")

        dataset_splits = {}
        for split_name, pointers in split_pointers.items():
            if not pointers:
                print(f"  - No data for '{split_name}' split. Skipping.")
                continue

            print(f"  - Generating '{split_name}' split...")
            dataset = datasets.Dataset.from_generator(
                self._generate_samples, gen_kwargs={"pointers": pointers}
            )

            if split_name == "train":
                print("  - Shuffling training set...")
                dataset = dataset.shuffle(seed=self.seed)

            dataset_splits[split_name] = dataset

        if not dataset_splits:
            print("No data was processed into any split. Halting.")
            return

        final_dataset_dict = DatasetDict(dataset_splits)

        print(f"\nSaving DatasetDict to {self.output_dir}...")
        final_dataset_dict.save_to_disk(str(self.output_dir))
        print("DatasetDict saved successfully.")
        print(f"Final dataset info:\n{final_dataset_dict}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a multilingual dataset from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset/sallm_dataset.yaml",
        help="Path to the dataset preparation configuration YAML file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    print(f"Loading configuration from: {config_path}")
    with config_path.open("r") as f:
        config_data = yaml.safe_load(f)

    processor = DataProcessor(config_data["dataset_preparation"])
    processor.run()


if __name__ == "__main__":
    main()
