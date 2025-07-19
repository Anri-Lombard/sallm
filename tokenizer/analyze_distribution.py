# TODO merge the codebase
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict

import datasets
from tabulate import tabulate
from tqdm import tqdm

DistributionResult = Dict[str, Dict[str, int]]


def analyze_token_distribution(dataset_dir: Path) -> None:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Processed dataset directory not found at: {dataset_dir}"
        )

    print(f"Loading processed dataset from: {dataset_dir}")
    processed_dataset_dict = datasets.load_from_disk(str(dataset_dir))

    try:
        any_split_key = next(iter(processed_dataset_dict))
        max_seq_length = len(processed_dataset_dict[any_split_key][0]["input_ids"])
        print(f"Inferred max_seq_length from data: {max_seq_length}")
    except (StopIteration, IndexError, KeyError):
        print("Dataset appears to be empty. Cannot determine sequence length.")
        return

    results: DistributionResult = defaultdict(dict)
    totals: Dict[str, int] = defaultdict(int)
    all_langs = set()

    for split_name, split_dataset in processed_dataset_dict.items():
        print(f"\nAnalyzing split: '{split_name}'...")
        if "lang" not in split_dataset.column_names:
            print(
                f"Warning: 'lang' column not found in '{split_name}' split. Skipping."
            )
            continue

        unique_langs_in_split = sorted(split_dataset.unique("lang"))
        all_langs.update(unique_langs_in_split)

        pbar = tqdm(
            unique_langs_in_split, desc=f"Processing languages for '{split_name}'"
        )
        for lang in pbar:
            pbar.set_postfix_str(f"lang={lang}")

            lang_dataset = split_dataset.filter(lambda x: x["lang"] == lang, num_proc=4)

            num_tokens = len(lang_dataset) * max_seq_length

            results[lang][split_name] = num_tokens
            totals[split_name] += num_tokens

    if not results:
        print("\nNo languages found or data was empty. Analysis complete.")
        return

    sorted_langs = sorted(list(all_langs))
    split_names = sorted(totals.keys())

    headers = ["Language"]
    for split in split_names:
        headers.append(f"{split.capitalize()} Tokens")
        headers.append(f"{split.capitalize()} (%)")

    table_data = []
    for lang in sorted_langs:
        row = [lang]
        for split in split_names:
            count = results[lang].get(split, 0)
            percentage = (count / totals[split] * 100) if totals[split] > 0 else 0
            row.append(f"{count:,}")
            row.append(f"{percentage:.2f}%")
        table_data.append(row)

    totals_row = ["TOTAL"]
    for split in split_names:
        total_count = totals[split]
        totals_row.append(f"{total_count:,}")
        totals_row.append("100.00%")

    print("\n" + "=" * 80)
    print(" " * 25 + "Token Distribution Analysis")
    print("=" * 80)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print(tabulate([totals_row], tablefmt="grid"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze token distribution per language for a chunked dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        default="data/sallm_processed",
        type=str,
        help="Path to the processed dataset directory (output of process.py).",
    )
    args = parser.parse_args()

    analyze_token_distribution(Path(args.dataset_dir))


if __name__ == "__main__":
    main()
