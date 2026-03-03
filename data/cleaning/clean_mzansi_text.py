#!/usr/bin/env python3
"""
MzansiText data cleaning pipeline
=================================

Exact script used to produce the filtered MzansiText corpus for the
LREC 2026 paper. Adapted from HuggingFaceFW/fineweb.

Run with:
    python data/cleaning/clean_mzansi_text.py \\
        --config data/cleaning/config.toml \\
        --concat
"""

import argparse
import dataclasses

import toml
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.typeshelper import StatHints

# This creates documents that are ~1000 tokens (very roughly).
CONCATENATE_LIMIT = 150


class BaseFormatter(PipelineStep):
    """An identity formatter needed to define the concatenator."""

    type = "FORMAT"

    def __init__(self) -> None:
        super().__init__()

    def format(self, text: str) -> str:
        return text

    def run(self, data, rank: int = 0, world_size: int = 1):
        for doc in data:
            self.stat_update(StatHints.total)
            with self.track_time():
                doc.text = self.format(doc.text)
            yield doc


class Concatenator(BaseFormatter):
    """Combine small documents to roughly 1000 tokens."""

    def __init__(self) -> None:
        super().__init__()

    def format(self, text: str) -> str:
        return text

    def run(self, data, rank: int = 0, world_size: int = 1):
        current_tokens = 0
        current_doc = None
        for doc in data:
            self.stat_update(StatHints.total)

            with self.track_time():
                if current_doc is None:
                    current_doc = doc
                    current_doc.text += " "
                    current_tokens = len(doc.text.split())
                else:
                    current_doc.text += doc.text + " "
                    current_tokens += len(doc.text.split())

                if current_tokens >= CONCATENATE_LIMIT:
                    yield current_doc
                    current_doc = None


def custom_adapter(_, document):
    """Use the original input filename as the output filename stem."""
    data = {key: val for key, val in dataclasses.asdict(document).items() if val}

    input_file = data["metadata"]["file_path"].split("/")[-1].split(".")[0]
    data.pop("metadata", None)
    data["file"] = input_file

    return data


def create_filter_executor(config: dict, concat: bool) -> LocalPipelineExecutor:
    try:
        input_folder = config["input_folder"]
        output_folder = config["output_folder"]
        tasks = config["tasks"]
        workers = config["workers"]
    except Exception as exc:
        print("Failed to read values from config file")
        print(exc)
        raise SystemExit(1) from exc

    print(f"Executing {tasks} tasks with {workers} workers...")

    intermediate_pipeline = [
        JsonlReader(f"{input_folder}"),
        GopherRepetitionFilter(
            exclusion_writer=JsonlWriter(
                f"{output_folder}/removed/gopher_repetition", compression=None
            ),
            dup_line_frac=0.5,
            dup_para_frac=0.5,
            dup_line_char_frac=0.5,
            dup_para_char_frac=0.5,
            top_n_grams=((2, 0.75), (3, 0.75), (4, 0.75)),
            dup_n_grams=(
                (5, 0.3),
                (6, 0.28),
                (7, 0.26),
                (8, 0.24),
                (9, 0.22),
                (10, 0.2),
            ),
        ),
        GopherQualityFilter(
            min_doc_words=10,
            min_stop_words=0,
            max_non_alpha_words_ratio=0.5,
            exclusion_writer=JsonlWriter(
                f"{output_folder}/removed/gopher_quality", compression=None
            ),
        ),
        C4QualityFilter(
            filter_no_terminal_punct=False,
            min_num_sentences=2,
            exclusion_writer=JsonlWriter(
                f"{output_folder}/removed/c4_quality", compression=None
            ),
        ),
        FineWebQualityFilter(
            line_punct_exclude_zero=True,
            short_line_thr=0.8,
            exclusion_writer=JsonlWriter(
                f"{output_folder}/removed/fineweb_quality", compression=None
            ),
        ),
    ]

    if concat:
        intermediate_pipeline.append(Concatenator())

    intermediate_pipeline.append(
        JsonlWriter(
            f"{output_folder}",
            compression=None,
            adapter=custom_adapter,
            output_filename="${rank}.jsonl",
        )
    )

    return LocalPipelineExecutor(
        pipeline=intermediate_pipeline,
        skip_completed=False,
        tasks=tasks,
        workers=workers,
        logging_dir=f"{output_folder}/filter_logs",
    )


def main() -> None:
    parser = argparse.ArgumentParser("pipeline")
    parser.add_argument(
        "-c",
        "--concat",
        help="Whether to concatenate smaller documents",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--config",
        help="Where to look for the config file",
        type=str,
        default="config.toml",
    )
    args = parser.parse_args()

    config = toml.load(args.config)

    filter_executor = create_filter_executor(config, args.concat)
    filter_executor.run()


if __name__ == "__main__":
    main()
