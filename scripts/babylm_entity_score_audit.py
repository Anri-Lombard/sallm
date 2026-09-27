#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit BabyLM entity-tracking option score margins."
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/scratch/alombard/babylm-gdn/repos/babylm-eval/strict"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/scratch/alombard/babylm-gdn/diagnostics/entity_nothing_score_audit.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--model", action="append", default=[], help="name=path_or_hf_repo"
    )
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    values = sorted(values)
    p10_idx = int(0.1 * (len(values) - 1))
    p90_idx = int(0.9 * (len(values) - 1))
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": values[p10_idx],
        "p90": values[p90_idx],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold_empty = [row for row in rows if row["gold_empty"]]
    gold_nonempty = [row for row in rows if not row["gold_empty"]]
    with_empty_option = [row for row in rows if row["empty_option_index"] is not None]
    return {
        "n": len(rows),
        "accuracy": pct(sum(row["correct"] for row in rows), len(rows)),
        "gold_empty_n": len(gold_empty),
        "gold_empty_accuracy": pct(
            sum(row["correct"] for row in gold_empty), len(gold_empty)
        ),
        "gold_nonempty_accuracy": pct(
            sum(row["correct"] for row in gold_nonempty), len(gold_nonempty)
        ),
        "predicted_empty_n": sum(row["predicted_empty"] for row in rows),
        "predicted_empty_rate": pct(
            sum(row["predicted_empty"] for row in rows), len(rows)
        ),
        "gold_margin": stats([row["gold_margin"] for row in rows]),
        "gold_empty_margin": stats([row["gold_margin"] for row in gold_empty]),
        "empty_option_margin": stats(
            [row["empty_margin"] for row in with_empty_option]
        ),
        "gold_empty_close_miss_rate_margin_gt_neg1": pct(
            sum(
                (not row["correct"]) and row["gold_margin"] > -1.0 for row in gold_empty
            ),
            len(gold_empty),
        ),
        "gold_empty_catastrophic_miss_rate_margin_lt_neg5": pct(
            sum(
                (not row["correct"]) and row["gold_margin"] < -5.0 for row in gold_empty
            ),
            len(gold_empty),
        ),
    }


def default_models() -> list[str]:
    root = "/scratch/alombard/babylm-gdn"
    return [
        "gpt2=BabyLM-community/gpt2-baseline-BabyLM-2026-Strict-Small",
        f"gdn12_s256={root}/outputs/gdn-512x12-packed-strict-small-20260629-173135",
        f"gdn12_s512={root}/outputs/gdn-h512l12-s512-packed-strict-small-20260629-ctx512",
        f"gdn24_s256={root}/outputs/gdn-h512l24-s256-packed-strict-small-20260629-depth24",
    ]


def parse_model_specs(specs: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for spec in specs or default_models():
        if "=" not in spec:
            raise ValueError(f"Model spec must be name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        parsed.append((name, path))
    return parsed


def run_model(
    eval_root: Path,
    model_name: str,
    model_path: str,
    batch_size: int,
    max_examples: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, str(eval_root))
    from evaluation_pipeline.sentence_zero_shot.dataset import get_dataloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = SimpleNamespace(
        data_path=eval_root / "evaluation_data" / "fast_eval" / "entity_tracking_fast",
        task="entity_tracking",
        model_path_or_name=model_path,
        backend="causal",
        revision_name=None,
        batch_size=batch_size,
        images_path=None,
        image_split=None,
        image_template=None,
        full_sentence_scores=False,
    )

    dataloader = get_dataloader(args)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(
        device
    )
    model.eval()
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for (
            raw_sentences,
            sentence_dict,
            labels,
            _metadatas,
            uids,
            _images,
        ) in dataloader:
            option_count = len(
                [key for key in sentence_dict if key.endswith("_attn_mask")]
            )
            scores = []
            for option_idx in range(option_count):
                prefix = f"sentence_{option_idx}"
                logits = model(
                    input_ids=sentence_dict[f"{prefix}_inputs"].to(device),
                    attention_mask=sentence_dict[f"{prefix}_attn_mask"].to(device),
                )
                logits = logits[0] if isinstance(logits, tuple) else logits["logits"]
                log_probs = F.log_softmax(logits, dim=-1)
                targets = sentence_dict[f"{prefix}_targets"].to(device)
                mask = sentence_dict[f"{prefix}_phrase_mask"].to(device)
                target_log_probs = torch.gather(
                    log_probs, -1, targets.unsqueeze(-1)
                ).squeeze(-1)
                scores.append(torch.sum(target_log_probs * mask, dim=1).cpu())

            stacked = torch.stack(scores, dim=1)
            chosen = torch.argmax(stacked, dim=1)
            for item_idx, raw in enumerate(raw_sentences):
                completions = raw["completions"]
                option_scores = [float(value) for value in stacked[item_idx].tolist()]
                gold_idx = int(labels[item_idx])
                chosen_idx = int(chosen[item_idx])
                best_wrong = max(
                    score for idx, score in enumerate(option_scores) if idx != gold_idx
                )
                empty_idx = next(
                    (
                        idx
                        for idx, option in enumerate(completions)
                        if option.strip() == "nothing."
                    ),
                    None,
                )
                empty_margin = None
                if empty_idx is not None:
                    best_nonempty = max(
                        score
                        for idx, score in enumerate(option_scores)
                        if idx != empty_idx
                    )
                    empty_margin = option_scores[empty_idx] - best_nonempty
                rows.append(
                    {
                        "id": (
                            f"{uids[item_idx]}_"
                            f"{sum(row['uid'] == uids[item_idx] for row in rows)}"
                        ),
                        "uid": uids[item_idx],
                        "gold": completions[gold_idx],
                        "prediction": completions[chosen_idx],
                        "correct": chosen_idx == gold_idx,
                        "gold_empty": completions[gold_idx].strip() == "nothing.",
                        "predicted_empty": completions[chosen_idx].strip()
                        == "nothing.",
                        "empty_option_index": empty_idx,
                        "gold_margin": option_scores[gold_idx] - best_wrong,
                        "empty_margin": empty_margin,
                        "scores": option_scores,
                        "options": completions,
                    }
                )
                if max_examples and len(rows) >= max_examples:
                    break
            if max_examples and len(rows) >= max_examples:
                break

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "model": model_name,
        "model_path": model_path,
        "summary": summarize(rows),
        "rows": rows,
    }


def self_check() -> None:
    rows = [
        {
            "correct": True,
            "gold_empty": True,
            "predicted_empty": True,
            "gold_margin": 2.0,
            "empty_margin": 2.0,
            "empty_option_index": 0,
        },
        {
            "correct": False,
            "gold_empty": True,
            "predicted_empty": False,
            "gold_margin": -0.5,
            "empty_margin": -0.5,
            "empty_option_index": 0,
        },
        {
            "correct": True,
            "gold_empty": False,
            "predicted_empty": False,
            "gold_margin": 1.0,
            "empty_margin": -3.0,
            "empty_option_index": 2,
        },
    ]
    summary = summarize(rows)
    assert summary["accuracy"] == 100.0 * 2 / 3
    assert summary["gold_empty_accuracy"] == 50.0
    assert summary["gold_empty_close_miss_rate_margin_gt_neg1"] == 50.0


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        print("SELF_CHECK_OK")
        return

    # ponytail: entity/causal only; add other tasks when a result needs it.
    result = {"models": []}
    for model_name, model_path in parse_model_specs(args.model):
        print(f"auditing {model_name}: {model_path}", flush=True)
        result["models"].append(
            run_model(
                args.eval_root,
                model_name,
                model_path,
                args.batch_size,
                args.max_examples,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
