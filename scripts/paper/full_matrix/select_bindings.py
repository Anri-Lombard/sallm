#!/usr/bin/env python3
"""Select only ambiguous adapters from frozen validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from statistics import fmean
from typing import Any


NER_CODES = {"tsn": "tn", "xho": "xh", "zul": "zu"}
SIB_PROMPTS = {"afr": 4, "eng": 3, "nso": 4, "sot": 3, "xho": 5, "zul": 5}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def ref_id(reference: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(reference).encode()).hexdigest()


def metric(values: dict[str, Any], task: str) -> float:
    names = (
        ("f1,flexible-extract", "f1")
        if task == "ner"
        else ("f1,none", "f1", "acc,none", "acc")
    )
    for name in names:
        value = values.get(name)
        if isinstance(value, int | float):
            return float(value)
    raise ValueError(f"No frozen selection metric in {values}")


def score(payload: dict[str, Any], arm: dict[str, Any], prompts: dict[str, dict[str, int]]) -> float:
    results = payload["calls"]["validation"]["results"]
    values = []
    for language in arm["languages"]:
        prompt = prompts[arm["task_group"]][language]
        if arm["task_group"] == "ner":
            name = f"sallm_masakhaner_{NER_CODES[language]}_prompt_{prompt}_val"
        else:
            name = f"sallm_sib_{language}_val_prompt_{prompt}"
        if name not in results:
            raise ValueError(f"Missing validation task {name}")
        values.append(metric(results[name], arm["task_group"]))
    return fmean(values)


def protocol_for(
    unit: dict[str, Any],
    candidate_index: int,
    entries: dict[str, Any],
    template: dict[str, Any],
) -> dict[str, Any]:
    base = entries[ref_id(unit["binding"]["base"])]
    adapter_ref = unit["binding"]["adapter_candidates"][candidate_index]
    adapter = entries[ref_id(adapter_ref)]
    architecture = unit["architecture"]
    interface = {
        "mzansilm": {"dtype": "bfloat16", "merge_lora": False, "tie_word_embeddings": None},
        "mamba2": {"dtype": "float32", "merge_lora": False, "tie_word_embeddings": False},
        "xlstm": {"dtype": "float32", "merge_lora": True, "tie_word_embeddings": False},
        "gdn": {"dtype": "bfloat16", "merge_lora": False, "tie_word_embeddings": None},
    }[architecture]
    output = deepcopy(template)
    output["models"] = {
        architecture: {
            **interface,
            "execution_status": "runnable",
            "base_path": base["path"],
            "adapter_path": adapter["path"],
            "base_tree_sha256": base["tree_sha256"],
            "adapter_tree_sha256": adapter["tree_sha256"],
            "base_files": base["files"],
            "adapter_files": adapter["files"],
        }
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--sequence-selection", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--protocol-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    arms = json.loads((args.inventory / "validation_arms.json").read_text())
    units = json.loads((args.inventory / "official_units.json").read_text())
    sequence_selection = json.loads(args.sequence_selection.read_text())
    prompts = {
        "ner": {key: int(value) for key, value in sequence_selection["selected_prompts"]["ner"].items()},
        "sib": SIB_PROMPTS,
    }
    by_unit: dict[str, list[tuple[int, float, str]]] = {}
    for arm in arms:
        path = args.validation_root / f"{arm['array_index']}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        if payload.get("mode") != "validation" or payload["item"] != arm:
            raise ValueError(f"Validation binding mismatch: {path}")
        value = score(payload, arm, prompts)
        by_unit.setdefault(arm["unit_id"], []).append(
            (int(arm["candidate_index"]), value, sha256(path))
        )
    selected: dict[str, int] = {}
    evidence: dict[str, Any] = {}
    for unit_id, candidates in sorted(by_unit.items()):
        ordered = sorted(candidates, key=lambda value: (-value[1], value[0]))
        selected[unit_id] = ordered[0][0]
        evidence[unit_id] = {
            "selected_candidate_index": ordered[0][0],
            "selection_metric": "validation_only_fixed_official_prompt",
            "candidates": [
                {"candidate_index": index, "validation_metric": value, "artifact_sha256": digest}
                for index, value, digest in sorted(candidates)
            ],
        }
    if len(selected) != 9:
        raise ValueError(f"Expected nine ambiguous execution units, found {len(selected)}")
    args.output.mkdir(parents=True, exist_ok=False)
    selection = {
        "schema": "sallm.full_matrix_binding_selection/v1",
        "data_boundary": "validation_only",
        "test_accessed": False,
        "no_test_based_selection": True,
        "sequence_prompt_selection_sha256": sha256(args.sequence_selection),
        "selected_bindings": selected,
        "evidence": evidence,
    }
    selection_path = args.output / "SELECTION.json"
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    entries = json.loads(args.bindings.read_text())["entries"]
    template = json.loads(args.protocol_template.read_text())
    protocols = args.output / "sequence_protocols"
    protocols.mkdir()
    for unit in units:
        if unit["task_group"] not in {"ner", "pos"}:
            continue
        candidate_index = selected.get(unit["unit_id"], 0)
        protocol = protocol_for(unit, candidate_index, entries, template)
        (protocols / f"{unit['unit_id']}.json").write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n"
        )
    marker = {
        "schema": "sallm.full_matrix_validation_verified/v1",
        "status": "VALIDATION_VERIFIED",
        "selection_sha256": sha256(selection_path),
        "validation_arm_count": len(arms),
        "selected_unit_count": len(selected),
        "sequence_protocol_count": len(list(protocols.glob("*.json"))),
        "test_accessed": False,
    }
    (args.output / "VALIDATION_VERIFIED.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(marker, sort_keys=True))


if __name__ == "__main__":
    main()
