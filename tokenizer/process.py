# TODO reorganize tokenizer code to be DRY across files
# TODO calculate tokenization distribution
import yaml
import argparse
from pathlib import Path
import os
from itertools import chain
from typing import Any, Dict

import datasets
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r") as f:
        data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise TypeError("Config file is not a dictionary.")
        return data


def process_dataset(config: dict) -> None:
    path_config = config["dataset_processing"]["paths"]
    settings_config = config["dataset_processing"]["settings"]

    source_dir = Path(path_config["source_dataset_dir"])
    tokenizer_path = Path(path_config["tokenizer_path"])
    output_dir = Path(path_config["output_dir"])
    max_seq_length = settings_config["max_seq_length"]
    num_proc = settings_config["num_proc"]
    seed = settings_config["seed"]

    # TODO pydantic models
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must have an EOS token defined.")

    print(f"Loading dataset from: {source_dir}")
    raw_datasets = datasets.load_from_disk(str(source_dir))

    def tokenize_and_chunk(examples: dict) -> Dict[str, list[list[int]]]:
        outputs = tokenizer(
            examples["text"], add_special_tokens=False, truncation=False
        )
        concatenated_ids = list(
            chain.from_iterable(
                tokens + [eos_token_id] for tokens in outputs["input_ids"]
            )
        )
        total_length = len(concatenated_ids)
        total_length = (total_length // max_seq_length) * max_seq_length
        result = {
            "input_ids": [
                concatenated_ids[i : i + max_seq_length]
                for i in range(0, total_length, max_seq_length)
            ]
        }
        return result

    unique_langs = raw_datasets["train"].unique("lang")
    print(f"\nFound {len(unique_langs)} languages to process: {unique_langs}")

    final_processed_splits = {}
    original_columns = raw_datasets["train"].column_names

    for split_name, split_dataset in raw_datasets.items():
        print(f"\n--- Processing split: {split_name} ---")

        processed_language_splits = []
        pbar = tqdm(unique_langs, desc=f"Processing languages for '{split_name}'")

        for lang in pbar:
            pbar.set_postfix_str(f"lang={lang}")

            lang_dataset = split_dataset.filter(
                lambda x: x["lang"] == lang, num_proc=num_proc
            )
            if len(lang_dataset) == 0:
                continue

            tokenized_dataset = lang_dataset.map(
                tokenize_and_chunk,
                batched=True,
                remove_columns=original_columns,
                num_proc=num_proc,
            )

            tokenized_dataset = tokenized_dataset.add_column(
                "lang", [lang] * len(tokenized_dataset)
            )
            processed_language_splits.append(tokenized_dataset)

        if not processed_language_splits:
            print(f"Split '{split_name}' is empty after processing. Skipping.")
            continue

        final_split = datasets.concatenate_datasets(processed_language_splits)

        if split_name == "train":
            print(
                f"Shuffling the '{split_name}' split to mix languages for training..."
            )
            final_split = final_split.shuffle(seed=seed)

        final_processed_splits[split_name] = final_split

    processed_dataset_dict = datasets.DatasetDict(final_processed_splits)

    print("\nProcessing complete. Saving datasets to disk...")
    processed_dataset_dict.save_to_disk(str(output_dir))

    print(f"\nFinal processed dataset saved to: {output_dir}")
    print("--- Verification ---")
    print(f"Processed dataset info:\n{processed_dataset_dict}")
    if "train" in processed_dataset_dict:
        train_sample = processed_dataset_dict["train"][0]
        print(
            f"Example 'train' sample: {{'input_ids': [shape: {len(train_sample['input_ids'])}], 'lang': '{train_sample['lang']}'}}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize and chunk a dataset for language model pre-training."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset/sallm_processed.yaml",
        help="Path to the data processing configuration YAML file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    print(f"Loading configuration from: {config_path}")
    config_data = load_config(config_path)
    process_dataset(config_data)


if __name__ == "__main__":
    main()
