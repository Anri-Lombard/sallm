#!/usr/bin/env python3
"""Independently verify and integrate all official General NER/POS test outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from finalize_general_sequence_replacement_20260914 import (
    ARCHITECTURES,
    LANGUAGES,
    atomic_json,
    canonical_sha256,
    load_payloads,
    ner_task_name,
    sha256,
    verify_common,
    verify_ner,
    verify_pos,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ner-dir", required=True, type=Path)
    parser.add_argument("--pos-dir", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--ready", required=True, type=Path)
    parser.add_argument("--ner-release", required=True, type=Path)
    parser.add_argument("--pos-release", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def expected_json_files(directory: Path) -> None:
    expected = {f"{architecture}.json" for architecture in ARCHITECTURES}
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    if observed != expected:
        raise ValueError(f"Unexpected official-test files in {directory}: {observed}")
    if any(directory.glob("*.tmp")):
        raise ValueError(f"Temporary official-test output remains in {directory}")


def verify_release(
    release_path: Path,
    task: str,
    selection_sha: str,
) -> None:
    release = json.loads(release_path.read_text())
    expected = {
        "schema": "sallm.general_sequence_test_release/v1",
        "task": task,
        "selection_sha256": selection_sha,
        "authorized": True,
        "one_time_test": True,
        "no_score_based_retry": True,
    }
    if any(release.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Invalid {task} test release")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    expected_json_files(args.ner_dir)
    expected_json_files(args.pos_dir)
    protocol = json.loads(args.protocol.read_text())
    protocol_sha = sha256(args.protocol)
    selection = json.loads(args.selection.read_text())
    selection_sha = sha256(args.selection)
    ready = json.loads(args.ready.read_text())
    expected_ready = {
        "schema": "sallm.general_sequence_test_ready/v1",
        "status": "VALIDATION_SELECTION_FROZEN_HELDOUT_LOCKED",
        "selection_sha256": selection_sha,
        "protocol_sha256": protocol_sha,
        "one_time_test": True,
        "no_score_based_retry": True,
    }
    if any(ready.get(key) != value for key, value in expected_ready.items()):
        raise ValueError("Official-test ready marker mismatch")
    verify_release(args.ner_release, "ner", selection_sha)
    verify_release(args.pos_release, "pos", selection_sha)

    payloads = {
        "ner": load_payloads(args.ner_dir),
        "pos": load_payloads(args.pos_dir),
    }
    all_scores: dict[str, dict[str, dict[str, float]]] = {"ner": {}, "pos": {}}
    all_identities: dict[str, dict[str, dict[str, set[Any]]]] = {
        "ner": {},
        "pos": {},
    }
    expected_keys: dict[str, set[str]] = {"ner": set(), "pos": set()}
    for language, prompt in selection["selected_prompts"]["ner"].items():
        expected_keys["ner"].add(ner_task_name(language, int(prompt), "test"))
    for language, prompt in selection["selected_prompts"]["pos"].items():
        expected_keys["pos"].add(f"{language}/P{int(prompt)}")

    rows_by_artifact: dict[str, dict[str, int]] = {"ner": {}, "pos": {}}
    rows_by_cell: dict[str, dict[str, dict[str, int]]] = {"ner": {}, "pos": {}}
    input_sha256: dict[str, dict[str, str]] = {"ner": {}, "pos": {}}
    prompt_identities: dict[str, dict[str, dict[str, set[tuple[str, str]]]]] = {
        "ner": {},
        "pos": {},
    }
    for task in ("ner", "pos"):
        verifier = verify_ner if task == "ner" else verify_pos
        directory = args.ner_dir if task == "ner" else args.pos_dir
        for architecture in ARCHITECTURES:
            payload = payloads[task][architecture]
            verify_common(
                payload,
                protocol,
                protocol_sha,
                architecture,
                task,
                "test",
                selection_sha,
            )
            scores, identities = verifier(payload, protocol, "test")
            if set(scores) != expected_keys[task] or set(identities) != expected_keys[task]:
                raise ValueError(f"Selected {task} cell mismatch for {architecture}")
            all_scores[task][architecture] = scores
            all_identities[task][architecture] = identities
            rows_by_artifact[task][architecture] = len(payload["rows"])
            rows_by_cell[task][architecture] = {
                cell: len(cell_identities)
                for cell, cell_identities in identities.items()
            }
            cell_prompts: dict[str, set[tuple[str, str]]] = defaultdict(set)
            for row in payload["rows"]:
                cell = (
                    str(row["task"])
                    if task == "ner"
                    else f"{row['language']}/P{int(row['prompt'])}"
                )
                prompt_hash = str(
                    row["prompt_hash" if task == "ner" else "prompt_sha256"]
                )
                if len(prompt_hash) != 64:
                    raise ValueError(f"Invalid prompt hash for {architecture}/{cell}")
                int(prompt_hash, 16)
                source_id = str(row["doc_id"] if task == "ner" else row["id"])
                cell_prompts[cell].add((source_id, prompt_hash))
            if {
                cell: len(values) for cell, values in cell_prompts.items()
            } != rows_by_cell[task][architecture]:
                raise ValueError(f"Prompt-hash row mismatch for {architecture}/{task}")
            prompt_identities[task][architecture] = dict(cell_prompts)
            if len(payload["rows"]) != sum(rows_by_cell[task][architecture].values()):
                raise ValueError(f"Row-accounting mismatch for {architecture}/{task}")
            if len(payload["rows"]) <= 0:
                raise ValueError(f"Empty official-test artifact for {architecture}/{task}")
            input_sha256[task][architecture] = sha256(
                directory / f"{architecture}.json"
            )
        reference = all_identities[task][ARCHITECTURES[0]]
        reference_counts = rows_by_cell[task][ARCHITECTURES[0]]
        reference_prompts = prompt_identities[task][ARCHITECTURES[0]]
        reference_evidence = payloads[task][ARCHITECTURES[0]]["task_evidence"]
        for architecture in ARCHITECTURES[1:]:
            if all_identities[task][architecture] != reference:
                raise ValueError(f"Unpaired official {task} rows for {architecture}")
            if rows_by_cell[task][architecture] != reference_counts:
                raise ValueError(f"Official {task} row-count mismatch for {architecture}")
            if prompt_identities[task][architecture] != reference_prompts:
                raise ValueError(f"Official {task} prompt hashes differ for {architecture}")
            if payloads[task][architecture]["task_evidence"] != reference_evidence:
                raise ValueError(f"Official {task} evidence differs for {architecture}")

    score_rows: list[dict[str, Any]] = []
    code_to_language = {"tn": "tsn", "xh": "xho", "zu": "zul"}
    for task in ("ner", "pos"):
        metric = "micro_entity_span_f1" if task == "ner" else "micro_token_accuracy"
        for architecture in ARCHITECTURES:
            for cell, score in sorted(all_scores[task][architecture].items()):
                if task == "ner":
                    parts = cell.split("_")
                    language = code_to_language[parts[2]]
                    prompt = int(parts[4])
                else:
                    language, prompt_text = cell.split("/P", 1)
                    prompt = int(prompt_text)
                score_rows.append(
                    {
                        "architecture": architecture,
                        "task": task,
                        "language": language,
                        "prompt": prompt,
                        "metric": metric,
                        "score": score,
                        "rows": rows_by_cell[task][architecture][cell],
                        "source_identities_sha256": canonical_sha256(
                            sorted(all_identities[task][architecture][cell])
                        ),
                        "prompt_identities_sha256": canonical_sha256(
                            sorted(prompt_identities[task][architecture][cell])
                        ),
                        "artifact_sha256": input_sha256[task][architecture],
                    }
                )
    if len(score_rows) != 24:
        raise ValueError(f"Expected 24 official scores, found {len(score_rows)}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema": "sallm.general_sequence_official_test_integration/v1",
        "status": "VERIFIED",
        "data_boundary": "official_heldout_test",
        "test_accessed": True,
        "one_time_test": True,
        "no_score_based_retry": True,
        "protocol_sha256": protocol_sha,
        "selection_sha256": selection_sha,
        "ready_sha256": sha256(args.ready),
        "release_sha256": {
            "ner": sha256(args.ner_release),
            "pos": sha256(args.pos_release),
        },
        "input_sha256": input_sha256,
        "rows_by_artifact": rows_by_artifact,
        "rows_by_cell": rows_by_cell,
        "score_count": len(score_rows),
        "expected_score_count": 24,
        "scores": score_rows,
    }
    output = args.output_dir / "GENERAL_TEST_INTEGRATION.json"
    atomic_json(output, payload)
    manifest = args.output_dir / "ARTIFACTS.sha256"
    manifest.write_text(f"{sha256(output)}  {output.name}\n")
    print(
        "GENERAL_OFFICIAL_TEST_VERIFIED "
        f"scores={len(score_rows)} rows={sum(sum(v.values()) for v in rows_by_artifact.values())}"
    )


if __name__ == "__main__":
    main()
