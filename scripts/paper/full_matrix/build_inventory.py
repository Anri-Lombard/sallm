#!/usr/bin/env python3
"""Freeze the remaining Base/Mono/Multi full-matrix execution inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_CELLS = 293
EXPECTED_UNITS = 97
MODEL_SLUG = {
    "MzansiLM": "mzansilm",
    "Mamba": "mamba2",
    "xLSTM": "xlstm",
    "GDN": "gdn",
}
LANG_SLUG = {
    "Afr": "afr",
    "Eng": "eng",
    "Nso": "nso",
    "Sot": "sot",
    "Ssw": "ssw",
    "Tsn": "tsn",
    "Tso": "tso",
    "Xho": "xho",
    "Zul": "zul",
}
TASK_SLUG = {
    "MasakhaNews": "news",
    "MasakhaNER": "ner",
    "MasakhaPOS": "pos",
    "SIB-200": "sib",
    "INJOngo Intent": "intent",
    "T2X": "t2x",
    "AfriHG": "afrihg",
    "Belebele": "belebele",
    "AfriXNLI": "afrixnli",
    "AfriMMLU": "afrimmlu",
    "AfriMGSM": "afrimgsm",
}
NEWS_PROMPTS = {"eng": 2, "xho": 4}
SIB_PROMPTS = {"afr": 4, "eng": 3, "nso": 4, "sot": 3, "xho": 5, "zul": 5}
INTENT_PROMPTS = {"eng": 4, "sot": 1, "xho": 2, "zul": 2}
XNLI_PROMPTS = {"eng": 1, "sot": 5, "xho": 3, "zul": 4}
MMLU_PROMPTS = {"eng": 2, "sot": 5, "xho": 5, "zul": 4}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded(row: dict[str, str]) -> str | None:
    if row["regime"] == "General" and row["task"] in {"MasakhaNER", "MasakhaPOS"}:
        return "general_sequence_jobs_1343060_1343062"
    if row["model"] == "MzansiLM" and row["task"] == "SIB-200" and row["regime"] == "Mono":
        return "mzansilm_sib_mono_jobs_1343003_1343004"
    if (
        row["model"] == "MzansiLM"
        and row["task"] == "MasakhaNews"
        and row["regime"] in {"Mono", "Multi"}
    ):
        return "historical_selected_job_1342734"
    if row["model"] == "MzansiLM" and row["task"] == "MasakhaNER" and row["regime"] == "Mono":
        return "historical_selected_job_1342734"
    if row["model"] == "MzansiLM" and row["task"] == "SIB-200" and row["regime"] == "Multi":
        return "historical_selected_job_1342734"
    if row["model"] == "xLSTM" and row["task"] == "MasakhaNER" and row["regime"] == "Mono":
        return "historical_selected_job_1342734"
    return None


def strip_digest_suffix(value: str) -> tuple[str, dict[str, str]]:
    metadata: dict[str, str] = {}
    match = re.search(r"@(tree|file):([0-9a-f]{64})$", value)
    if match:
        metadata[f"expected_{match.group(1)}_sha256"] = match.group(2)
        value = value[: match.start()]
    return value, metadata


def parse_location(value: str, *, kind: str) -> dict[str, Any]:
    value = value.strip()
    if value == "none":
        return {"kind": "none"}
    if value.startswith("path="):
        value = value.removeprefix("path=")
    if value.startswith("repo_id="):
        fields: dict[str, str] = {}
        for part in value.split("; "):
            if "=" in part:
                key, item = part.split("=", 1)
                fields[key] = item
        revision = fields.get("revision")
        result: dict[str, Any] = {
            "kind": "hub",
            "repo_id": fields["repo_id"],
            "revision": None if revision == "unresolved_private_head" else revision,
            "revision_status": "resolve_head_metadata_only" if revision == "unresolved_private_head" else "pinned",
        }
        if fields.get("adapter_model_sha256"):
            result["expected_adapter_model_sha256"] = fields["adapter_model_sha256"]
        return result
    if value.startswith("/"):
        location, metadata = strip_digest_suffix(value)
        return {"kind": "path", "path": location, **metadata}
    if "@" in value:
        repo_id, revision = value.rsplit("@", 1)
        return {"kind": "hub", "repo_id": repo_id, "revision": revision, "revision_status": "pinned"}
    if kind == "base" and value.count("/") == 1:
        return {"kind": "hub", "repo_id": value, "revision": None, "revision_status": "resolve_head_metadata_only"}
    raise ValueError(f"Unsupported {kind} binding: {value}")


def parse_binding(raw: str) -> dict[str, Any]:
    if not raw.startswith("base=") or "; adapter=" not in raw:
        raise ValueError(f"Malformed binding: {raw}")
    base_raw, adapter_raw = raw.removeprefix("base=").split("; adapter=", 1)
    candidates = [parse_location(item, kind="adapter") for item in adapter_raw.split(" | ")]
    return {"base": parse_location(base_raw, kind="base"), "adapter_candidates": candidates}


def selected_task(task: str, language: str) -> str:
    lang = LANG_SLUG[language]
    if task == "MasakhaNews":
        return f"sallm_masakhanews_{lang}_test_prompt_{NEWS_PROMPTS[lang]}"
    if task == "SIB-200":
        return f"sib_{lang}_prompt_{SIB_PROMPTS[lang]}"
    if task == "INJOngo Intent":
        return f"injongointent_{lang}_prompt_{INTENT_PROMPTS[lang]}"
    if task == "AfriXNLI":
        return f"afrixnli_{lang}_prompt_{XNLI_PROMPTS[lang]}"
    if task == "AfriMMLU":
        return f"afrimmlu_direct_{lang}_prompt_{MMLU_PROMPTS[lang]}"
    if task == "AfriMGSM":
        return f"afrimgsm_{lang}_prompt_1"
    if task == "Belebele":
        return f"belebele_{lang}_prompt_1"
    raise ValueError(task)


def cell_id(row: dict[str, str]) -> str:
    return ":".join(
        (
            MODEL_SLUG[row["model"]],
            TASK_SLUG[row["task"]],
            LANG_SLUG[row["language"]],
            row["regime"].lower(),
        )
    )


def unit_key(row: dict[str, str]) -> tuple[str, ...]:
    model = MODEL_SLUG[row["model"]]
    regime = row["regime"].lower()
    task = TASK_SLUG[row["task"]]
    if regime == "base":
        if row["task"] in {"MasakhaNER", "MasakhaPOS"}:
            return regime, model, task
        if row["task"] in {"T2X", "AfriHG"}:
            return regime, model, "generation"
        return regime, model, "prompt"
    return regime, model, task, row["checkpoint/adapter"]


def build(ledger: Path) -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    with ledger.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    actionable = [row for row in rows if row["action"] in {"rerun", "missing"}]
    excluded_rows = [(row, excluded(row)) for row in actionable if excluded(row)]
    retained = [row for row in actionable if excluded(row) is None]
    if len(actionable) != 339 or len(excluded_rows) != 46 or len(retained) != EXPECTED_CELLS:
        raise ValueError(
            f"Unexpected scope: actionable={len(actionable)} excluded={len(excluded_rows)} retained={len(retained)}"
        )

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in retained:
        grouped[unit_key(row)].append(row)

    units: list[dict[str, Any]] = []
    validation_arms: list[dict[str, Any]] = []
    for index, (_, unit_rows) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
        first = unit_rows[0]
        binding = parse_binding(first["checkpoint/adapter"])
        for row in unit_rows[1:]:
            if row["checkpoint/adapter"] != first["checkpoint/adapter"]:
                raise ValueError("A unit contains multiple bindings")
        task_names = []
        for row in unit_rows:
            if row["task"] not in {"MasakhaNER", "MasakhaPOS", "T2X", "AfriHG"}:
                task_names.append(selected_task(row["task"], row["language"]))
        unit_id = f"u{index:03d}-" + "-".join(
            part for part in unit_key(first)[:3] if not part.startswith("base=")
        )
        if first["regime"] != "Base" and len(unit_rows) == 1:
            unit_id += f"-{LANG_SLUG[first['language']]}"
        unit = {
            "array_index": index,
            "unit_id": unit_id,
            "model": first["model"],
            "architecture": MODEL_SLUG[first["model"]],
            "regime": first["regime"],
            "task_group": unit_key(first)[2],
            "tasks": sorted({row["task"] for row in unit_rows}),
            "languages": sorted({LANG_SLUG[row["language"]] for row in unit_rows}),
            "cell_ids": sorted(cell_id(row) for row in unit_rows),
            "cells": len(unit_rows),
            "task_names": sorted(task_names),
            "binding_evidence_status": first["binding_evidence_status"],
            "binding_evidence_source": first["binding_evidence_source"],
            "binding": binding,
        }
        units.append(unit)
        if len(binding["adapter_candidates"]) > 1:
            for candidate_index, candidate in enumerate(binding["adapter_candidates"]):
                validation_arms.append(
                    {
                        "array_index": len(validation_arms),
                        "unit_id": unit_id,
                        "candidate_index": candidate_index,
                        "architecture": unit["architecture"],
                        "task_group": unit["task_group"],
                        "languages": unit["languages"],
                        "base": binding["base"],
                        "adapter": candidate,
                        "data_boundary": "validation_only",
                    }
                )

    if len(units) != EXPECTED_UNITS:
        raise ValueError(f"Expected {EXPECTED_UNITS} units, found {len(units)}")
    if sum(unit["cells"] for unit in units) != EXPECTED_CELLS:
        raise ValueError("Unit coverage differs from retained cell count")
    if len(validation_arms) != 25:
        raise ValueError(f"Expected 25 ambiguity validation arms, found {len(validation_arms)}")
    if not any(
        row["model"] in {"MzansiLM", "Mamba"}
        and row["task"] == "Belebele"
        and row["language"] == "Tso"
        and row["regime"] == "Base"
        for row in retained
    ):
        raise ValueError("Required MzansiLM/Mamba Base Belebele Tso cells are absent")

    summary = {
        "schema": "sallm.full_matrix_remaining_execution/v1",
        "ledger": str(ledger.resolve()),
        "ledger_sha256": sha256(ledger),
        "actionable_at_freeze": len(actionable),
        "excluded_cells": len(excluded_rows),
        "covered_cells": len(retained),
        "official_units": len(units),
        "ambiguity_validation_arms": len(validation_arms),
        "held_out_accessed": False,
        "selection_policy": "adapter ambiguity resolved from validation only; frozen prompt maps reused; no test-based selection or retry",
        "required_base_belebele_tso": ["mzansilm:belebele:tso:base", "mamba2:belebele:tso:base"],
        "exclusion_counts": dict(
            sorted(
                (reason, sum(1 for _, item in excluded_rows if item == reason))
                for reason in {item for _, item in excluded_rows}
            )
        ),
    }
    return retained, units, validation_arms, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    retained, units, validation_arms, summary = build(args.ledger)
    with (args.output / "cell_inventory.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=retained[0].keys())
        writer.writeheader()
        writer.writerows(retained)
    for name, value in (
        ("official_units.json", units),
        ("validation_arms.json", validation_arms),
        ("summary.json", summary),
    ):
        (args.output / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest = []
    for path in sorted(args.output.iterdir()):
        if path.name != "ARTIFACTS.sha256":
            manifest.append(f"{sha256(path)}  {path.name}")
    (args.output / "ARTIFACTS.sha256").write_text("\n".join(manifest) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
