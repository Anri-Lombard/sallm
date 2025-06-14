import os
import json
import argparse
import random
import yaml
from pathlib import Path
from collections import defaultdict
from itertools import groupby
from typing import List, Dict, NamedTuple
from tqdm import tqdm


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
        self.temp_dir = self.output_dir / "tmp"

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
        self._prepare_directories()

        language_indexes = self._build_index()
        split_pointers = self._calculate_splits(language_indexes)
        temp_train_path = self._write_unshuffled_splits(split_pointers)
        self._external_shuffle(temp_train_path)

        self._cleanup(temp_train_path)
        print("\nProcessing complete.")
        print(f"Final dataset saved to: {self.output_dir}")

    def _prepare_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
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

    def _write_unshuffled_splits(
        self, split_pointers: Dict[str, List[FilePointer]]
    ) -> Path:
        print("Phase 3: Writing splits via sorted index...")
        temp_train_path = self.temp_dir / "train.tmp.jsonl"

        for split_name, pointers in split_pointers.items():
            pointers.sort(key=lambda p: (p.file_path, p.offset))
            output_path = (
                temp_train_path
                if split_name == "train"
                else self.output_dir / split_name / "data-00000-of-00001.jsonl"
            )

            with (
                output_path.open("wb") as out_f,
                tqdm(total=len(pointers), desc=f"Writing {split_name} split") as pbar,
            ):
                for file_path, group in groupby(pointers, key=lambda p: p.file_path):
                    with file_path.open("rb") as in_f:
                        for pointer in group:
                            in_f.seek(pointer.offset)
                            line = in_f.readline()
                            data = json.loads(line)
                            lang = self._normalize_lang_code(
                                pointer.file_path, data.get("file", "")
                            )
                            output_data = {"text": data["text"], "lang": lang}
                            out_f.write(
                                (json.dumps(output_data) + "\n").encode("utf-8")
                            )
                            pbar.update(1)
        return temp_train_path

    def _external_shuffle(self, temp_train_path: Path) -> None:
        print("Phase 4: Performing external shuffle on training set...")
        line_offsets = []
        file_size = temp_train_path.stat().st_size
        with (
            temp_train_path.open("rb") as f,
            tqdm(
                total=file_size,
                desc="  - Indexing train.tmp",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            offset = 0
            for line in f:
                line_offsets.append(offset)
                line_len = len(line)
                offset += line_len
                pbar.update(line_len)

        final_train_path = self.output_dir / "train" / "data-00000-of-00001.jsonl"
        with final_train_path.open("wb") as out_f, temp_train_path.open("rb") as in_f:
            random.shuffle(line_offsets)
            for offset in tqdm(line_offsets, desc="  - Shuffling train set"):
                in_f.seek(offset)
                out_f.write(in_f.readline())

    def _cleanup(self, temp_train_path: Path) -> None:
        print("Cleaning up temporary files...")
        temp_train_path.unlink()
        try:
            self.temp_dir.rmdir()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a multilingual dataset from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets/sallm_dataset.yaml",
        help="Path to the dataset preparation configuration YAML file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    # TODO restructure library to use pydantic models for loading configs
    print(f"Loading configuration from: {config_path}")
    with config_path.open("r") as f:
        config_data = yaml.safe_load(f)

    processor = DataProcessor(config_data["dataset_preparation"])
    processor.run()


if __name__ == "__main__":
    main()
