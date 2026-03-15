#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


LANG_MAP = {
    "tn": "tsn",
    "xh": "xho",
    "zu": "zul",
}


def _unwrap_singleton_lists(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score rerank manifest with lm-eval outputs"
    )
    parser.add_argument(
        "--manifest", required=True, help="CSV emitted by hpo_submit_rerank.sh"
    )
    parser.add_argument("--out", required=True, help="Scored candidate CSV")
    parser.add_argument(
        "--winners-yaml", required=True, help="Best-hparams YAML output"
    )
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isnan(f):
            return None
        return f
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            f = float(text)
        except ValueError:
            return None
        if math.isnan(f):
            return None
        return f
    return None


def _infer_task_type(task_config: str) -> str:
    lower = task_config.lower()
    if "_ner_" in lower:
        return "ner"
    if "_pos_" in lower:
        return "pos"
    return "other"


def _find_results_file(eval_output_dir: str) -> Path | None:
    if not eval_output_dir:
        return None
    root = Path(eval_output_dir)
    if not root.exists():
        return None
    candidates = sorted(root.rglob("results.json"))
    if not candidates:
        return None
    return candidates[0]


def _select_metric(task_type: str, metrics: dict[str, Any]) -> tuple[str, float] | None:
    numeric: list[tuple[str, float]] = []
    for key, value in metrics.items():
        if key == "alias" or key.endswith("stderr"):
            continue
        val = _to_float(value)
        if val is None:
            continue
        numeric.append((key, val))
    if not numeric:
        return None

    def pick(predicate):
        for key, val in numeric:
            if predicate(key):
                return key, val
        return None

    if task_type == "ner":
        chosen = pick(lambda k: k.lower().startswith("f1"))
        if chosen:
            return chosen
    elif task_type == "pos":
        chosen = pick(lambda k: k.lower().startswith("token_accuracy"))
        if chosen:
            return chosen
        chosen = pick(lambda k: k.lower().startswith("acc"))
        if chosen:
            return chosen

    priority = [
        "token_accuracy",
        "f1",
        "acc",
        "all_chrf",
        "all_rougel",
        "exact_match",
        "bleu",
    ]
    for prefix in priority:
        chosen = pick(lambda k, p=prefix: k.lower().startswith(p))
        if chosen:
            return chosen

    return numeric[0]


def _infer_language(task_name: str) -> str | None:
    name = task_name.lower()

    for pattern in [
        r"masakhapos_(tsn|xho|zul)(?:_val)?_prompt",
        r"sallm_masakhapos_(tsn|xho|zul)(?:_val)?_prompt",
        r"masakhaner_(tn|xh|zu)(?:_val)?_prompt",
        r"sib_([a-z]{3})",
        r"injongointent_([a-z]{3})",
        r"afrihg_([a-z]{3})",
        r"masakhanews_([a-z]{3})",
    ]:
        m = re.search(pattern, name)
        if m:
            raw = m.group(1)
            return LANG_MAP.get(raw, raw)

    if "_prompt_" in name:
        before_prompt = name.split("_prompt_", 1)[0]
        last = before_prompt.rsplit("_", 1)[-1]
        if len(last) in {2, 3, 4}:
            return LANG_MAP.get(last, last)

    return None


def _aggregate(entries: list[tuple[str, float]], task_config: str) -> float | None:
    if not entries:
        return None
    if task_config.endswith("_all"):
        by_lang: dict[str, list[float]] = defaultdict(list)
        fallback: list[float] = []
        for task_name, score in entries:
            lang = _infer_language(task_name)
            if lang is None:
                fallback.append(score)
            else:
                by_lang[lang].append(score)

        lang_means: list[float] = []
        for values in by_lang.values():
            lang_means.append(sum(values) / len(values))
        if fallback:
            lang_means.append(sum(fallback) / len(fallback))
        if not lang_means:
            return None
        return sum(lang_means) / len(lang_means)

    scores = [score for _, score in entries]
    return sum(scores) / len(scores)


def _compute_pos_score_from_samples(payload: dict[str, Any]) -> list[tuple[str, float]]:
    samples_by_task = payload.get("samples")
    if not isinstance(samples_by_task, dict):
        return []

    task_scores: list[tuple[str, float]] = []
    for task_name, samples in samples_by_task.items():
        if not isinstance(samples, list) or not samples:
            continue

        sample_scores: list[float] = []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            ref = _unwrap_singleton_lists(sample.get("target"))
            pred = _unwrap_singleton_lists(sample.get("filtered_resps"))

            if not isinstance(ref, list):
                sample_scores.append(0.0)
                continue
            if not isinstance(pred, list):
                sample_scores.append(0.0)
                continue
            if not ref:
                sample_scores.append(0.0)
                continue

            aligned = min(len(ref), len(pred))
            correct = sum(1 for i in range(aligned) if str(ref[i]) == str(pred[i]))
            sample_scores.append(correct / len(ref))

        if sample_scores:
            task_scores.append((task_name, sum(sample_scores) / len(sample_scores)))

    return task_scores


def _better(metric_name: str, lhs: float, rhs: float) -> bool:
    metric = metric_name.lower()
    if "loss" in metric:
        return lhs < rhs
    return lhs > rhs


def main() -> int:
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    scored_rows: list[dict[str, Any]] = []
    winner_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        out = dict(row)
        task_config = row.get("task_config", "")
        task_type = _infer_task_type(task_config)
        result_file = _find_results_file(row.get("eval_output_dir", ""))
        out["results_file"] = str(result_file) if result_file else ""
        out["score_metric"] = ""
        out["score"] = ""

        if result_file is None:
            out["status"] = "missing_results"
            scored_rows.append(out)
            continue

        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            out["status"] = "invalid_results_json"
            scored_rows.append(out)
            continue

        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, dict) or not results:
            out["status"] = "results_empty"
            scored_rows.append(out)
            continue

        task_scores: list[tuple[str, float]] = []
        metric_names: list[str] = []
        if task_type == "pos":
            task_scores = _compute_pos_score_from_samples(payload)
            if task_scores:
                metric_names = ["token_accuracy,flexible-extract"] * len(task_scores)

        if not task_scores:
            for task_name, metrics in results.items():
                if not isinstance(metrics, dict):
                    continue
                chosen = _select_metric(task_type, metrics)
                if not chosen:
                    continue
                metric_name, value = chosen
                task_scores.append((task_name, value))
                metric_names.append(metric_name)

        score = _aggregate(task_scores, task_config)
        if score is None:
            out["status"] = "metric_not_found"
            scored_rows.append(out)
            continue

        primary_metric = (
            Counter(metric_names).most_common(1)[0][0] if metric_names else ""
        )
        out["score_metric"] = primary_metric
        out["score"] = f"{score:.10f}"
        out["status"] = "scored"
        scored_rows.append(out)
        winner_candidates[task_config].append(out)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(scored_rows[0].keys()) if scored_rows else []
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(scored_rows)

    winners: dict[str, Any] = {}
    for task_config, candidates in sorted(winner_candidates.items()):
        scored = [c for c in candidates if c.get("score")]
        if not scored:
            continue

        scored.sort(
            key=lambda c: (
                float(c["score"]),
                -float(c.get("objective_value") or 0.0),
            ),
            reverse=True,
        )

        best = scored[0]
        metric_name = best.get("score_metric", "")
        for candidate in scored[1:]:
            lhs = float(candidate["score"])
            rhs = float(best["score"])
            if _better(metric_name, lhs, rhs):
                best = candidate

        try:
            params = json.loads(best.get("parameter_json") or "{}")
            if not isinstance(params, dict):
                params = {}
        except Exception:
            params = {}

        winners[task_config] = {
            "sweep_id": best.get("sweep_id", ""),
            "sweep_name": best.get("sweep_name", ""),
            "run_id": best.get("run_id", ""),
            "run_name": best.get("run_name", ""),
            "rank": int(best.get("rank") or 0),
            "hpo_objective": best.get("objective_metric", ""),
            "hpo_objective_goal": best.get("objective_goal", ""),
            "topk_metric": best.get("score_metric", ""),
            "topk_score": float(best.get("score", "nan")),
            "finetune_config": f"finetune/{task_config}",
            "selected_hparams": params,
            "override_args": best.get("override_args", ""),
            "eval_output_dir": best.get("eval_output_dir", ""),
            "results_file": best.get("results_file", ""),
        }

    winners_payload = {
        "version": 1,
        "defaults": {
            "k_ner_pos": 10,
            "k_other": 6,
            "aggregation": "macro-over-languages for *_all; mean-over-prompts per language",
        },
        "winners": winners,
    }

    winners_path = Path(args.winners_yaml)
    winners_path.parent.mkdir(parents=True, exist_ok=True)
    winners_path.write_text(
        yaml.safe_dump(winners_payload, sort_keys=False), encoding="utf-8"
    )

    sheet_path = Path("outputs/hpo/llama_optimization_sheet.csv")
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    with sheet_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_config",
                "sweep_id",
                "hpo_objective",
                "topk_metric",
                "winner_run",
                "selected_hparams",
                "final_eval_metric",
            ],
        )
        writer.writeheader()
        for task_config, winner in sorted(winners.items()):
            writer.writerow(
                {
                    "task_config": task_config,
                    "sweep_id": winner.get("sweep_id", ""),
                    "hpo_objective": winner.get("hpo_objective", ""),
                    "topk_metric": winner.get("topk_metric", ""),
                    "winner_run": winner.get("run_id", ""),
                    "selected_hparams": json.dumps(
                        winner.get("selected_hparams", {}),
                        sort_keys=True,
                    ),
                    "final_eval_metric": winner.get("topk_score", ""),
                }
            )

    print(f"Scored candidates written to: {out_path}")
    print(f"Winners YAML written to: {winners_path}")
    print(f"Spreadsheet CSV written to: {sheet_path}")
    print(f"Winners found: {len(winners)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
