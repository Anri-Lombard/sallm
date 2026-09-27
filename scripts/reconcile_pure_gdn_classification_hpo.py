#!/usr/bin/env python3
"""Reconcile pure-GDN classification HPO using validation macro-F1 only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import wandb

ENTITY = "anri-lombard"
PROJECT = "sallm-ft"
FAMILIES = {
    "news": {
        "languages": ["eng", "xho"],
        "runs": [
            {"run_id": "ldtnnu63", "job_id": "1183134", "lr": 3e-5},
            {"run_id": "d46ssqps", "job_id": "1183160", "lr": 8e-5},
            {"run_id": "y4go6aeh", "job_id": "1183161", "lr": 1.5e-4},
        ],
        "excluded_trials": [
            {
                "job_id": "1183162",
                "reason": "deliberately cancelled; no validation metric",
            }
        ],
    },
    "sib": {
        "languages": [
            "afr_Latn",
            "eng_Latn",
            "nso_Latn",
            "sot_Latn",
            "xho_Latn",
            "zul_Latn",
        ],
        "runs": [
            {"run_id": "86xxffgr", "job_id": "1189730", "lr": 3e-5},
            {"run_id": "v2b6kqxe", "job_id": "1189838", "lr": 8e-5},
            {"run_id": "web3ll2q", "job_id": "1189839", "lr": 1.5e-4},
        ],
        "excluded_trials": [],
    },
    "intent": {
        "languages": ["eng", "sot", "xho", "zul"],
        "runs": [
            {"run_id": "t7smhrtn", "job_id": "1192257", "lr": 3e-5},
            {"run_id": "r9qqor27", "job_id": "1192267", "lr": 8e-5},
            {"run_id": "ug1czuc8", "job_id": "1192268", "lr": 1.5e-4},
        ],
        "excluded_trials": [],
    },
}
EXPECTED_WINNERS = {
    "news": {
        "run_id": "ldtnnu63",
        "job_id": "1183134",
        "checkpoint": 543,
        "macro_f1": 0.0736133408810401,
    },
    "sib": {
        "run_id": "86xxffgr",
        "job_id": "1189730",
        "checkpoint": 526,
        "macro_f1": 0.05760368663594471,
    },
    "intent": {
        "run_id": "r9qqor27",
        "job_id": "1192267",
        "checkpoint": 2643,
        "macro_f1": 0.002501957982855507,
    },
}


def metric_rows(run: Any, languages: list[str]) -> list[dict[str, Any]]:
    language_keys = [f"classification/{language}_macro_f1" for language in languages]
    rows: list[dict[str, Any]] = []
    for history_row in run.scan_history():
        values = [history_row.get(key) for key in language_keys]
        if all(value is None for value in values):
            continue
        if any(value is None or not math.isfinite(float(value)) for value in values):
            raise ValueError(
                f"Run {run.id} has an incomplete/non-finite macro-F1 row: {values}"
            )
        global_step = history_row.get("train/global_step")
        if global_step is None:
            raise ValueError(f"Run {run.id} metric row has no train/global_step.")
        language_metrics = {
            language: float(value)
            for language, value in zip(languages, values, strict=True)
        }
        rows.append(
            {
                "wandb_step": int(history_row["_step"]),
                "checkpoint_global_step": int(global_step),
                "language_macro_f1": language_metrics,
                "mean_language_macro_f1": sum(language_metrics.values())
                / len(language_metrics),
                "logged_support_weighted_all_f1": history_row.get(
                    "classification/all_f1"
                ),
            }
        )
    if not rows:
        raise ValueError(f"Run {run.id} has no complete validation macro-F1 rows.")
    return sorted(rows, key=lambda row: row["checkpoint_global_step"])


def select_checkpoint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        rows,
        key=lambda row: (
            -row["mean_language_macro_f1"],
            row["checkpoint_global_step"],
        ),
    )[0]


def reconcile() -> dict[str, Any]:
    api = wandb.Api()
    families: dict[str, Any] = {}
    for family, specification in FAMILIES.items():
        runs: list[dict[str, Any]] = []
        for run_spec in specification["runs"]:
            run = api.run(f"{ENTITY}/{PROJECT}/{run_spec['run_id']}")
            rows = metric_rows(run, specification["languages"])
            best = select_checkpoint(rows)
            runs.append(
                {
                    **run_spec,
                    "wandb_name": run.name,
                    "wandb_state": run.state,
                    "wandb_url": run.url,
                    "validation_rows": rows,
                    "selected_checkpoint": best,
                }
            )
        winner = sorted(
            runs,
            key=lambda item: (
                -item["selected_checkpoint"]["mean_language_macro_f1"],
                item["lr"],
                item["selected_checkpoint"]["checkpoint_global_step"],
            ),
        )[0]
        families[family] = {
            "languages": specification["languages"],
            "runs": runs,
            "excluded_trials": specification["excluded_trials"],
            "winner": {
                "run_id": winner["run_id"],
                "job_id": winner["job_id"],
                "lr": winner["lr"],
                "checkpoint": winner["selected_checkpoint"][
                    "checkpoint_global_step"
                ],
                "macro_f1": winner["selected_checkpoint"][
                    "mean_language_macro_f1"
                ],
            },
        }

    for family, expected in EXPECTED_WINNERS.items():
        actual = families[family]["winner"]
        for key in ("run_id", "job_id", "checkpoint"):
            if actual[key] != expected[key]:
                raise ValueError(
                    f"{family} winner drift for {key}: "
                    f"actual={actual[key]}, expected={expected[key]}"
                )
        if not math.isclose(
            actual["macro_f1"],
            expected["macro_f1"],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"{family} winner metric drift: "
                f"actual={actual['macro_f1']}, expected={expected['macro_f1']}"
            )

    return {
        "schema": "pure_gdn_classification_validation_reconciliation/v1",
        "protocol_date": "2026-08-09",
        "data_boundary": "validation-only W&B history; no held-out metric accessed",
        "aggregation_rule": (
            "label-macro F1 within each language, then arithmetic mean across "
            "languages; maximize; exact LR ties choose lower LR and exact "
            "checkpoint ties choose earlier global step"
        ),
        "early_stopping_rule": {
            "patience": 2,
            "absolute_threshold": 0.001,
            "note": "retained preregistered stopping rule; not re-optimized",
        },
        "source": {"entity": ENTITY, "project": PROJECT},
        "families": families,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = reconcile()
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output} ({digest})")
    for family, details in payload["families"].items():
        print(f"{family}: {details['winner']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
