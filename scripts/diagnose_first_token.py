#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any

import torch
from sallm.config import ModelEvalConfig
from sallm.evaluation.harness import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare EOS and reference first-token probabilities."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--compare-cache", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item()) + 1


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    summary: dict[str, float | int] = {
        "n": len(rows),
        "eos_top1_rate": fmean(row["eos_rank"] == 1 for row in rows),
        "eos_mean_probability": fmean(row["eos_probability"] for row in rows),
        "eos_median_rank": median(row["eos_rank"] for row in rows),
        "reference_top1_rate": fmean(
            row["reference_first_token_rank"] == 1 for row in rows
        ),
        "reference_mean_probability": fmean(
            row["reference_first_token_probability"] for row in rows
        ),
        "reference_median_rank": median(
            row["reference_first_token_rank"] for row in rows
        ),
    }
    if rows and "cached_prediction" in rows[0]:
        summary.update(
            {
                "cached_empty_rate": fmean(
                    not row["cached_prediction"] for row in rows
                ),
                "uncached_empty_rate": fmean(
                    not row["uncached_prediction"] for row in rows
                ),
                "cache_predictions_equal_rate": fmean(
                    row["cached_prediction"] == row["uncached_prediction"]
                    for row in rows
                ),
            }
        )
    return summary


def load_examples(path: Path, limit: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[:limit]


def score(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer = load_model_and_tokenizer(
        ModelEvalConfig(
            checkpoint=args.checkpoint,
            peft_adapter=args.adapter,
            merge_lora=False,
            dtype="bfloat16",
            device="cuda:0",
        )
    )
    model.eval()
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no EOS token")

    rows = []
    with torch.no_grad():
        for example in load_examples(args.examples, args.limit):
            messages = example["prompt_messages"]
            reference = example["reference"]
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            full_ids = tokenizer.apply_chat_template(
                [*messages, {"role": "assistant", "content": reference}],
                tokenize=True,
                add_generation_prompt=False,
            )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError("Assistant reference does not extend prompt tokens")
            suffix_ids = full_ids[len(prompt_ids) :]
            lexical_offset = next(
                (
                    index
                    for index, token_id in enumerate(suffix_ids)
                    if tokenizer.decode([token_id]).strip()
                ),
                None,
            )
            if lexical_offset is None:
                raise ValueError("Assistant reference has no lexical token")
            structural_prefix = suffix_ids[:lexical_offset]
            reference_id = suffix_ids[lexical_offset]

            context_limit = int(getattr(model.config, "max_position_embeddings", 2048))
            scored_ids = [*prompt_ids, *structural_prefix][-(context_limit - 1) :]
            input_ids = torch.tensor([scored_ids], device=model.device)
            logits = model(input_ids=input_ids).logits[0, -1].float()
            probabilities = torch.softmax(logits, dim=-1)
            top_ids = torch.topk(logits, k=5).indices.tolist()
            rows.append(
                {
                    "example_index": example.get("example_index", len(rows)),
                    "reference": reference,
                    "generated_prediction": example["prediction"],
                    "forced_structural_prefix": tokenizer.decode(structural_prefix),
                    "eos_probability": float(probabilities[eos_id].item()),
                    "eos_rank": token_rank(logits, eos_id),
                    "reference_first_token": tokenizer.decode([reference_id]),
                    "reference_first_token_id": reference_id,
                    "reference_first_token_probability": float(
                        probabilities[reference_id].item()
                    ),
                    "reference_first_token_rank": token_rank(logits, reference_id),
                    "top_tokens": [
                        {
                            "token": tokenizer.decode([token_id]),
                            "token_id": token_id,
                            "probability": float(probabilities[token_id].item()),
                        }
                        for token_id in top_ids
                    ],
                }
            )
            if args.compare_cache:
                generation_kwargs = {
                    "max_new_tokens": 8,
                    "do_sample": False,
                    "num_beams": 1,
                    "pad_token_id": tokenizer.pad_token_id or eos_id,
                    "eos_token_id": eos_id,
                }
                generation_prompt_ids = prompt_ids[-(context_limit - 8) :]
                prompt_tensor = torch.tensor(
                    [generation_prompt_ids], device=model.device
                )
                for use_cache, key in (
                    (True, "cached_prediction"),
                    (False, "uncached_prediction"),
                ):
                    generated = model.generate(
                        input_ids=prompt_tensor,
                        use_cache=use_cache,
                        **generation_kwargs,
                    )
                    rows[-1][key] = tokenizer.decode(
                        generated[0, len(generation_prompt_ids) :],
                        skip_special_tokens=True,
                    ).strip()

    return {
        "checkpoint": args.checkpoint,
        "adapter": args.adapter,
        "examples": str(args.examples),
        "summary": summarize(rows),
        "rows": rows,
    }


def self_check() -> None:
    logits = torch.tensor([0.0, 2.0, 1.0])
    assert token_rank(logits, 1) == 1
    assert token_rank(logits, 2) == 2
    summary = summarize(
        [
            {
                "eos_rank": 1,
                "eos_probability": 0.75,
                "reference_first_token_rank": 2,
                "reference_first_token_probability": 0.25,
            }
        ]
    )
    assert summary["eos_top1_rate"] == 1.0
    assert summary["reference_median_rank"] == 2


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        print("SELF_CHECK_OK")
        return
    result = score(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
