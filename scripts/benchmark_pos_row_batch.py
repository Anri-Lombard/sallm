#!/usr/bin/env python3
"""Compare serial and row-batched full-prefix POS validation scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset
from run_pure_gdn_hpo_correction_canary import validation_dataset
from sallm.config import ModelEvalConfig
from sallm.evaluation.constrained_label_scoring import decode_pos_tuple_contract
from sallm.evaluation.harness import load_model_and_tokenizer
from sallm.evaluation.pos_batched_scoring import decode_pos_tuple_contract_batch
from sallm.evaluation.pos_metrics import (
    UPOS_LABELS,
    aggregate_pos_cell_accuracies,
    parse_pos_tuple_completion,
)

BATCH_SIZE = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_row_per_cell(dataset: Dataset) -> tuple[Dataset, list[int]]:
    indices: list[int] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(dataset):
        cell = (str(row["lang"]), str(row["template_id"]))
        if cell not in seen:
            seen.add(cell)
            indices.append(index)
    if len(indices) != 12:
        raise ValueError(f"Expected 12 POS cells, found {sorted(seen)}")
    return dataset.select(indices), indices


def score_rows(
    *, model: Any, tokenizer: Any, dataset: Dataset, batched: bool
) -> dict[str, Any]:
    device = getattr(model, "device", torch.device("cuda"))
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("POS scoring requires a pad or EOS token id.")
    samples = list(dataset)
    parsed = []
    for sample in samples:
        messages = cast(list[dict[str, str]], sample["messages"])
        tokens, gold = parse_pos_tuple_completion(messages[-1]["content"])
        parsed.append((messages, tokens, gold))

    results: list[tuple[list[str], float]] = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    if batched:
        for start in range(0, len(parsed), BATCH_SIZE):
            chunk = parsed[start : start + BATCH_SIZE]
            results.extend(
                decode_pos_tuple_contract_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_messages_batch=[messages[:-1] for messages, _, _ in chunk],
                    tokens_batch=[tokens for _, tokens, _ in chunk],
                    labels=UPOS_LABELS,
                    score_mode="mean",
                    pad_token_id=int(pad_token_id),
                    pad_to_multiple_of=64,
                    device=device,
                )
            )
    else:
        for messages, tokens, _ in parsed:
            results.append(
                decode_pos_tuple_contract(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_messages=messages[:-1],
                    tokens=tokens,
                    labels=UPOS_LABELS,
                    score_mode="mean",
                    pad_token_id=int(pad_token_id),
                    pad_to_multiple_of=64,
                    device=device,
                )
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    rows: list[dict[str, Any]] = []
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row_index, (sample, (_, tokens, gold), result) in enumerate(
        zip(samples, parsed, results, strict=True)
    ):
        predictions, selected_score = result
        cell = (str(sample["lang"]), str(sample["template_id"]))
        correct = sum(
            prediction == reference
            for prediction, reference in zip(predictions, gold, strict=True)
        )
        counts[cell][0] += correct
        counts[cell][1] += len(gold)
        rows.append(
            {
                "row": row_index,
                "cell": f"{cell[0]}/{cell[1]}",
                "tokens": len(tokens),
                "predictions": predictions,
                "gold": gold,
                "correct": correct,
                "selected_sequence_score": selected_score,
            }
        )
    immutable_counts = {cell: tuple(values) for cell, values in counts.items()}
    accuracy, cell_metrics = aggregate_pos_cell_accuracies(immutable_counts)
    return {
        "implementation": "row_batch_full_prefix_v1" if batched else "full_prefix_v1",
        "row_batch_size": BATCH_SIZE if batched else 1,
        "elapsed_seconds": elapsed,
        "rows": rows,
        "cell_counts": {
            f"{language}/{template}": {"correct": correct, "total": total}
            for (language, template), (correct, total) in sorted(
                immutable_counts.items()
            )
        },
        "cell_metrics": cell_metrics,
        "all_token_accuracy": accuracy,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Equivalence gate requires exactly one visible CUDA device.")
    validation = validation_dataset("gdn_pure_pos_all_hpo_r2")
    subset, subset_indices = first_row_per_cell(validation)
    model, tokenizer = load_model_and_tokenizer(
        ModelEvalConfig(
            checkpoint=str(args.checkpoint),
            peft_adapter=str(args.adapter),
            dtype="bfloat16",
            device="cuda",
            merge_lora=False,
        )
    )
    model.eval()
    return {
        "schema": "sallm_pos_row_batch_equivalence/v1",
        "data_boundary": "validation-only; no held-out split loaded or scored",
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_job_gres": os.getenv("SLURM_JOB_GRES"),
        "checkpoint": str(args.checkpoint),
        "adapter": str(args.adapter),
        "adapter_hashes": {
            str(path.relative_to(args.adapter)): sha256_file(path)
            for path in sorted(args.adapter.rglob("*"))
            if path.is_file()
        },
        "validation_rows": len(validation),
        "subset_indices": subset_indices,
        "results": {
            "serial": score_rows(
                model=model, tokenizer=tokenizer, dataset=subset, batched=False
            ),
            "batched": score_rows(
                model=model, tokenizer=tokenizer, dataset=subset, batched=True
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    args.output.with_suffix(".json.sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(f"Wrote {args.output} ({digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
