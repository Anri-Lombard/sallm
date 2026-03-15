#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb


@dataclass
class SweepObjective:
    name: str
    goal: str


def _load_mapping(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}
    if hasattr(raw, "_json_dict"):
        parsed = getattr(raw, "_json_dict")
        if isinstance(parsed, dict):
            return dict(parsed)
    if hasattr(raw, "as_dict"):
        try:
            parsed = raw.as_dict()
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return dict(parsed)
    if hasattr(raw, "items"):
        try:
            return {str(key): value for key, value in raw.items()}
        except Exception:
            pass
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _normalize_sweep_ref(project: str, sweep: str) -> str:
    cleaned = sweep.strip()
    if cleaned.count("/") >= 2:
        return cleaned
    return f"{project}/{cleaned}"


def _extract_base_config(sweep_cfg: dict[str, Any]) -> str:
    command = sweep_cfg.get("command", [])
    if not isinstance(command, list):
        return ""
    for idx, token in enumerate(command):
        if token == "--base-config" and idx + 1 < len(command):
            return str(command[idx + 1])
    return ""


def _infer_task_config(base_config: str, sweep_name: str) -> str:
    base = base_config.strip()
    if base.startswith("finetune/"):
        return base.split("/", 1)[1]
    if base.startswith("finetune") and "/" in base:
        return base.rsplit("/", 1)[-1]

    name = sweep_name.strip().replace("hpo-", "")
    name = name.replace("-", "_")
    if not name.startswith("llama_"):
        name = f"llama_{name}"
    return name


def _metric_from_summary(summary: dict[str, Any], metric_name: str) -> float | None:
    candidates = [metric_name]
    if "/" in metric_name:
        candidates.append(metric_name.replace("/", "_"))
    for key in candidates:
        value = summary.get(key)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            return float(value)
    return None


def _render_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_./:-]+", text):
        return text
    return json.dumps(text)


def _build_override_args(params: dict[str, Any]) -> str:
    overrides = []
    for key in sorted(params):
        overrides.append(f"finetune.{key}={_render_scalar(params[key])}")
    return " ".join(overrides)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select top-K completed sweep runs from W&B for re-ranking"
    )
    parser.add_argument("--project", required=True, help="entity/project")
    parser.add_argument("--sweep", required=True, help="sweep id or full sweep path")
    parser.add_argument("--k", type=int, required=True, help="number of runs to keep")
    parser.add_argument("--out", required=True, help="output CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.k <= 0:
        raise SystemExit("--k must be > 0")

    sweep_ref = _normalize_sweep_ref(args.project, args.sweep)
    api = wandb.Api()
    sweep = api.sweep(sweep_ref)

    sweep_cfg = _load_mapping(getattr(sweep, "config", {}))
    objective_cfg = _load_mapping(sweep_cfg.get("metric"))
    objective = SweepObjective(
        name=str(objective_cfg.get("name", "eval/loss")),
        goal=str(objective_cfg.get("goal", "minimize")).lower(),
    )
    if objective.goal not in {"minimize", "maximize"}:
        objective.goal = "minimize"

    sweep_name = str(getattr(sweep, "name", args.sweep))
    parameter_cfg = _load_mapping(sweep_cfg.get("parameters"))
    parameter_keys = sorted(parameter_cfg.keys())
    base_config = _extract_base_config(sweep_cfg)
    task_config = _infer_task_config(base_config, sweep_name)

    rows: list[dict[str, Any]] = []
    for run in getattr(sweep, "runs", []):
        state = str(getattr(run, "state", "")).lower()
        if state != "finished":
            continue

        summary = _load_mapping(getattr(run, "summary", {}))
        metric_value = _metric_from_summary(summary, objective.name)
        if metric_value is None:
            continue

        run_cfg = _load_mapping(getattr(run, "config", {}))
        selected_params = {
            key: run_cfg[key]
            for key in parameter_keys
            if key in run_cfg and not str(key).startswith("_")
        }
        rows.append(
            {
                "project": args.project,
                "sweep_id": str(getattr(sweep, "id", args.sweep)),
                "sweep_name": sweep_name,
                "task_config": task_config,
                "base_config": base_config,
                "run_id": str(getattr(run, "id", "")),
                "run_name": str(getattr(run, "name", "")),
                "objective_metric": objective.name,
                "objective_goal": objective.goal,
                "objective_value": metric_value,
                "parameter_json": json.dumps(selected_params, sort_keys=True),
                "override_args": _build_override_args(selected_params),
            }
        )

    reverse = objective.goal == "maximize"
    rows.sort(key=lambda item: float(item["objective_value"]), reverse=reverse)
    selected = rows[: args.k]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "project",
        "sweep_id",
        "sweep_name",
        "task_config",
        "base_config",
        "run_id",
        "run_name",
        "objective_metric",
        "objective_goal",
        "objective_value",
        "parameter_json",
        "override_args",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(selected, start=1):
            row_with_rank = dict(row)
            row_with_rank["rank"] = idx
            writer.writerow(row_with_rank)

    print(
        f"Wrote {len(selected)} runs (from {len(rows)} completed) to {out_path}",
        file=sys.stderr,
    )
    print(
        f"Sweep={sweep_ref} objective={objective.name} ({objective.goal}) task={task_config}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
