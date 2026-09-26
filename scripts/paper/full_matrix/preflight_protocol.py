#!/usr/bin/env python3
"""Verify the frozen execution protocol without loading evaluation rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from lm_eval.tasks import TaskManager
from sallm.evaluation.lm_eval_runner import _prepare_include_paths


NER_CODES = {"tsn": "tn", "xho": "xh", "zul": "zu"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manager(source: Path, includes: list[str]) -> TaskManager:
    include_path = (
        _prepare_include_paths([str(source / value) for value in includes])
        if includes
        else None
    )
    return TaskManager(include_path=include_path, include_defaults=True)


def require_tasks(task_manager: TaskManager, tasks: set[str], label: str) -> None:
    missing = sorted(task for task in tasks if task not in task_manager.task_index)
    if missing:
        raise ValueError(f"Missing {label} tasks: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    units_path = args.inventory / "official_units.json"
    arms_path = args.inventory / "validation_arms.json"
    units = json.loads(units_path.read_text())
    arms = json.loads(arms_path.read_text())
    if len(units) != 97 or sum(int(unit["cells"]) for unit in units) != 293:
        raise ValueError("Frozen official scope differs from 97 units / 293 cells")
    if len(arms) != 25:
        raise ValueError("Frozen ambiguity scope differs from 25 validation arms")
    indexes = [int(unit["array_index"]) for unit in units]
    if indexes != list(range(97)):
        raise ValueError("Official array indexes are not exact and contiguous")
    cell_ids = [cell for unit in units for cell in unit["cell_ids"]]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("Official cell identifiers are not unique")
    required = {"mzansilm:belebele:tso:base", "mamba2:belebele:tso:base"}
    if not required.issubset(cell_ids):
        raise ValueError("Required Base Belebele Tso cells are absent")

    ner_validation: set[str] = set()
    sib_validation: set[str] = set()
    for arm in arms:
        target = ner_validation if arm["task_group"] == "ner" else sib_validation
        for language in arm["languages"]:
            for prompt in range(1, 6):
                if arm["task_group"] == "ner":
                    target.add(
                        f"sallm_masakhaner_{NER_CODES[language]}_prompt_{prompt}_val"
                    )
                elif arm["task_group"] == "sib":
                    target.add(f"sallm_sib_{language}_val_prompt_{prompt}")
                else:
                    raise ValueError(f"Unsupported validation group: {arm['task_group']}")

    require_tasks(
        manager(
            args.source,
            ["src/conf/eval/lm_eval_tasks/masakhaner_validation"],
        ),
        ner_validation,
        "MasakhaNER validation",
    )
    require_tasks(
        manager(args.source, ["src/conf/eval/lm_eval_tasks/sib_validation"]),
        sib_validation,
        "SIB validation",
    )

    news_tasks: set[str] = set()
    default_tasks: set[str] = set()
    for unit in units:
        if unit["task_group"] in {"ner", "pos", "generation"}:
            continue
        for task in unit["task_names"]:
            (news_tasks if task.startswith("sallm_masakhanews_") else default_tasks).add(task)
    require_tasks(
        manager(args.source, ["src/conf/eval/lm_eval_tasks/masakhanews_test"]),
        news_tasks,
        "MasakhaNews official",
    )
    require_tasks(manager(args.source, []), default_tasks, "default official")

    generation_template = yaml.safe_load(
        (args.source / "src/conf/eval/run_llama_general_full_matrix_r1.yaml").read_text()
    )
    generation_ids = {
        task["id"] for task in generation_template["evaluation"]["generation_tasks"]
    }
    if generation_ids != {"t2x_xho", "afrihg_xho", "afrihg_zul"}:
        raise ValueError(f"Frozen generation task set differs: {generation_ids}")

    payload = {
        "schema": "sallm.full_matrix_protocol_preflight/v1",
        "status": "PROTOCOL_PREFLIGHT_PASSED",
        "data_boundary": "configuration_and_task_registry_only",
        "held_out_rows_loaded": False,
        "official_units": len(units),
        "official_cells": len(cell_ids),
        "validation_arms": len(arms),
        "official_task_names": len(news_tasks | default_tasks),
        "validation_task_names": len(ner_validation | sib_validation),
        "inventory_sha256": sha256(units_path),
        "validation_arms_sha256": sha256(arms_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
