#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tiny out-of-template BabyLM entity-state probe."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/scratch/alombard/babylm-gdn/diagnostics/entity_custom_state_probe.json"
        ),
    )
    parser.add_argument(
        "--model", action="append", default=[], help="name=path_or_hf_repo"
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def examples() -> list[dict[str, Any]]:
    return [
        {
            "id": "crate_removed_apple",
            "kind": "empty",
            "prompt": (
                "Crate 2 held apple. Apple was removed from crate 2. "
                "Crate 2 now contains "
            ),
            "options": ["nothing.", "apple.", "key.", "book."],
            "gold": 0,
        },
        {
            "id": "bag_taken_key",
            "kind": "empty",
            "prompt": (
                "A red bag had key. Then key was taken out of the red bag. "
                "The red bag has "
            ),
            "options": ["nothing.", "key.", "cup.", "bell."],
            "gold": 0,
        },
        {
            "id": "shelf_moved_cup",
            "kind": "empty",
            "prompt": (
                "Shelf 5 started with cup. Cup was moved away from shelf 5. "
                "Shelf 5 currently has "
            ),
            "options": ["nothing.", "cup.", "hat.", "letter."],
            "gold": 0,
        },
        {
            "id": "crate_stayed_cup",
            "kind": "nonempty",
            "prompt": (
                "Crate 1 held cup. Crate 2 held key. Cup stayed in crate 1. "
                "Crate 1 now contains "
            ),
            "options": ["nothing.", "cup.", "key.", "bottle."],
            "gold": 1,
        },
        {
            "id": "bag_still_bell",
            "kind": "nonempty",
            "prompt": (
                "A blue bag had bell. Key was removed from a different bag. "
                "The blue bag still has "
            ),
            "options": ["nothing.", "bell.", "key.", "shell."],
            "gold": 1,
        },
        {
            "id": "shelf_kept_book",
            "kind": "nonempty",
            "prompt": (
                "Shelf 4 started with book. Shelf 5 was emptied instead. "
                "Shelf 4 currently has "
            ),
            "options": ["nothing.", "book.", "coat.", "paper."],
            "gold": 1,
        },
    ]


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def summarize(
    rows: list[dict[str, Any]], prediction_key: str
) -> dict[str, float | int]:
    empty = [row for row in rows if row["kind"] == "empty"]
    nonempty = [row for row in rows if row["kind"] == "nonempty"]
    return {
        "n": len(rows),
        "accuracy": pct(
            sum(row[f"{prediction_key}_correct"] for row in rows), len(rows)
        ),
        "empty_accuracy": pct(
            sum(row[f"{prediction_key}_correct"] for row in empty), len(empty)
        ),
        "nonempty_accuracy": pct(
            sum(row[f"{prediction_key}_correct"] for row in nonempty), len(nonempty)
        ),
        "predicted_empty_rate": pct(
            sum(row[f"{prediction_key}_prediction"] == "nothing." for row in rows),
            len(rows),
        ),
    }


def default_models() -> list[str]:
    root = "/scratch/alombard/babylm-gdn"
    return [
        f"gdn12_s256={root}/outputs/gdn-512x12-packed-strict-small-20260629-173135",
        f"synthentity2k_seed13={root}/outputs/gdn-h512l12-s256-synthentity2k-strict-small-20260630",
        f"synthentity2k_seed29={root}/outputs/gdn-h512l12-s256-synthentity2k-seed29-strict-small-20260630",
    ]


def parse_model_specs(specs: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for spec in specs or default_models():
        if "=" not in spec:
            raise ValueError(f"Model spec must be name=path, got {spec!r}")
        parsed.append(tuple(spec.split("=", 1)))
    return parsed


def score_model(model_name: str, model_path: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(
        device
    )
    model.eval()

    rows = []
    with torch.no_grad():
        for example in examples():
            sum_scores = []
            mean_scores = []
            prompt_ids = tokenizer.encode(example["prompt"], add_special_tokens=False)
            for option in example["options"]:
                full_ids = tokenizer.encode(
                    example["prompt"] + option, add_special_tokens=False
                )
                inputs = torch.tensor([full_ids[:-1]], device=device)
                targets = torch.tensor(full_ids[1:], device=device)
                logits = model(input_ids=inputs).logits[0]
                log_probs = F.log_softmax(logits, dim=-1)
                start = max(0, len(prompt_ids) - 1)
                token_scores = (
                    log_probs[start:, :]
                    .gather(1, targets[start:].unsqueeze(1))
                    .squeeze(1)
                )
                sum_scores.append(float(token_scores.sum().cpu()))
                mean_scores.append(float(token_scores.mean().cpu()))
            sum_prediction_idx = max(range(len(sum_scores)), key=sum_scores.__getitem__)
            mean_prediction_idx = max(
                range(len(mean_scores)), key=mean_scores.__getitem__
            )
            gold_idx = int(example["gold"])
            best_wrong_sum = max(
                score for idx, score in enumerate(sum_scores) if idx != gold_idx
            )
            best_wrong_mean = max(
                score for idx, score in enumerate(mean_scores) if idx != gold_idx
            )
            rows.append(
                {
                    "id": example["id"],
                    "kind": example["kind"],
                    "gold": example["options"][gold_idx],
                    "sum_prediction": example["options"][sum_prediction_idx],
                    "sum_correct": sum_prediction_idx == gold_idx,
                    "sum_gold_margin": sum_scores[gold_idx] - best_wrong_sum,
                    "mean_prediction": example["options"][mean_prediction_idx],
                    "mean_correct": mean_prediction_idx == gold_idx,
                    "mean_gold_margin": mean_scores[gold_idx] - best_wrong_mean,
                    "sum_scores": sum_scores,
                    "mean_scores": mean_scores,
                    "option_token_lengths": [
                        len(
                            tokenizer.encode(
                                example["prompt"] + option, add_special_tokens=False
                            )
                        )
                        - len(prompt_ids)
                        for option in example["options"]
                    ],
                    "options": example["options"],
                }
            )

    return {
        "model": model_name,
        "model_path": model_path,
        "sum_summary": summarize(rows, "sum"),
        "mean_summary": summarize(rows, "mean"),
        "rows": rows,
    }


def self_check() -> None:
    rows = [
        {"kind": "empty", "sum_correct": True, "sum_prediction": "nothing."},
        {"kind": "nonempty", "sum_correct": False, "sum_prediction": "nothing."},
    ]
    summary = summarize(rows, "sum")
    assert len(examples()) == 6
    assert summary["accuracy"] == 50.0
    assert summary["predicted_empty_rate"] == 100.0


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        print("SELF_CHECK_OK")
        return

    result = {"examples": examples(), "models": []}
    for model_name, model_path in parse_model_specs(args.model):
        print(f"probing {model_name}: {model_path}", flush=True)
        result["models"].append(score_model(model_name, model_path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
