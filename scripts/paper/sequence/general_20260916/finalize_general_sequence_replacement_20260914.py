#!/usr/bin/env python3
"""Verify General NER/POS rows, freeze shared prompts, and verify selected tests."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from statistics import fmean
from typing import Any

ARCHITECTURES = ("mzansilm", "mamba2", "xlstm", "gdn")
LANGUAGES = ("tsn", "xho", "zul")
NER_CODES = {"tsn": "tn", "xho": "xh", "zul": "zu"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--ner-dir", type=Path, required=True)
    select.add_argument("--pos-dir", type=Path, required=True)
    select.add_argument("--protocol", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify-test")
    verify.add_argument("--task", choices=("ner", "pos"), required=True)
    verify.add_argument("--input-dir", type=Path, required=True)
    verify.add_argument("--protocol", type=Path, required=True)
    verify.add_argument("--selection", type=Path, required=True)
    verify.add_argument("--ready", type=Path, required=True)
    verify.add_argument("--release", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    subparsers.add_parser("self-check")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def assert_hash(value: Any, context: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Invalid hash for {context}: {value!r}")
    int(value, 16)


def assert_close(left: float, right: float, context: str) -> None:
    if not (math.isfinite(left) and math.isfinite(right)):
        raise ValueError(f"Non-finite metric for {context}")
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Metric mismatch for {context}: {left} != {right}")


def normalize_span_text(value: Any) -> str:
    # This deliberately preserves the historical task-pack character class,
    # including its unescaped hyphens.  It is the frozen metric contract used
    # by lm-eval for these immutable validation rows.
    punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@.""-,`'
    # NFD model outputs vs NFC references: compare in NFC.
    text = unicodedata.normalize("NFC", str(value))
    text = re.sub("[" + punctuation + "]+", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s{3,}|\t", "", text)
    return re.sub(r"\s+", " ", text).lower()


def spans(value: Any) -> list[tuple[str, str]]:
    if isinstance(value, list):
        value = " ".join(str(item).strip() for item in value)
    pieces = [
        item.strip()
        for sub in str(value).strip().split("$$")
        for item in sub.split("$")
        if item
    ]
    pieces = [
        item.strip()
        for value_part in pieces
        for sub in value_part.split(". ")
        for item in sub.split(", ")
    ]
    output = []
    for item in pieces:
        parts = item.split(": ")
        if len(parts) == 2:
            output.append(
                (normalize_span_text(parts[0]), normalize_span_text(parts[1]))
            )
    return output


def span_f1(rows: list[dict[str, Any]]) -> float:
    true_positive: collections.Counter[str] = collections.Counter()
    false_positive: collections.Counter[str] = collections.Counter()
    false_negative: collections.Counter[str] = collections.Counter()
    for row in rows:
        gold = spans(row["target"])
        predicted = spans(row["prediction"])
        for candidate in predicted:
            if candidate in gold:
                true_positive[candidate[0]] += 1
                gold.remove(candidate)
            else:
                false_positive[candidate[0]] += 1
        for missed in gold:
            false_negative[missed[0]] += 1
    tp = sum(true_positive.values())
    fp = sum(false_positive.values())
    fn = sum(false_negative.values())
    precision = tp / (tp + fp + 1e-13)
    recall = tp / (tp + fn + 1e-13)
    return 2.0 * precision * recall / (precision + recall + 1e-13)


def ner_task_name(language: str, prompt: int, phase: str) -> str:
    suffix = "val" if phase == "validation" else "test"
    return f"sallm_masakhaner_{NER_CODES[language]}_prompt_{prompt}_{suffix}"


def load_payloads(directory: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for architecture in ARCHITECTURES:
        path = directory / f"{architecture}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payloads[architecture] = json.loads(path.read_text())
    return payloads


def verify_binding(
    payload: dict[str, Any], protocol: dict[str, Any], architecture: str
) -> None:
    expected = protocol["models"][architecture]
    binding = payload["binding"]
    if binding.get("base_path") != expected["base_path"]:
        raise ValueError(f"Base path mismatch for {architecture}")
    if binding.get("adapter_path") != expected["adapter_path"]:
        raise ValueError(f"Adapter path mismatch for {architecture}")
    for field in ("base_tree_sha256", "adapter_tree_sha256"):
        if binding.get(field) != expected[field]:
            raise ValueError(f"{field} mismatch for {architecture}")
    for group, field in (("base", "base_files"), ("adapter", "adapter_files")):
        if binding["files"].get(group) != expected[field]:
            raise ValueError(f"{group} file binding mismatch for {architecture}")


def verify_common(
    payload: dict[str, Any],
    protocol: dict[str, Any],
    protocol_hash: str,
    architecture: str,
    task: str,
    phase: str,
    selection_hash: str | None,
) -> None:
    expected = {
        "schema": "sallm.general_sequence_eval/v1",
        "task": task,
        "phase": phase,
        "architecture": architecture,
        "test_accessed": phase == "test",
        "selection_sha256": selection_hash,
        "protocol_sha256": protocol_hash,
        "maximum_input_tokens": 1024,
        "deterministic": True,
        "seed": 42,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Common contract mismatch for {architecture}/{task}")
    if phase == "validation" and payload.get("limit_per_cell") is not None:
        raise ValueError(f"Partial validation result for {architecture}/{task}")
    if not math.isfinite(float(payload.get("elapsed_seconds", math.nan))):
        raise ValueError(f"Invalid elapsed time for {architecture}/{task}")
    verify_binding(payload, protocol, architecture)


def verify_ner(
    payload: dict[str, Any],
    protocol: dict[str, Any],
    phase: str,
) -> tuple[dict[str, float], dict[str, set[tuple[str, str, str]]]]:
    if phase == "validation":
        expected_tasks = {
            ner_task_name(language, prompt, phase)
            for language in LANGUAGES
            for prompt in range(1, 6)
        }
    else:
        expected_tasks = set(payload["reported_metrics"])
        if len(expected_tasks) != len(LANGUAGES):
            raise ValueError("NER test must contain one selected prompt per language")
    if set(payload["reported_metrics"]) != expected_tasks:
        raise ValueError("NER task coverage mismatch")
    if set(payload["task_evidence"]) != expected_tasks:
        raise ValueError("NER evidence coverage mismatch")

    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    identities: dict[str, set[tuple[str, str, str]]] = {}
    for row in payload["rows"]:
        task = str(row["task"])
        if task not in expected_tasks:
            raise ValueError(f"Unexpected NER task {task}")
        for field in ("doc_hash", "prompt_hash", "target_hash"):
            assert_hash(row.get(field), f"{task}/{field}")
        if int(row["input_token_count"]) > 1024:
            raise ValueError(f"NER input cap exceeded for {task}")
        grouped[task].append(row)
    if set(grouped) != expected_tasks:
        raise ValueError("NER row coverage mismatch")

    scores: dict[str, float] = {}
    for task, rows in grouped.items():
        language = payload["task_evidence"][task]["language"]
        evidence = payload["task_evidence"][task]
        if evidence.get("split") != phase:
            raise ValueError(f"NER split mismatch for {task}")
        if evidence.get("dataset_revision") != protocol["tasks"]["ner"]["revision"]:
            raise ValueError(f"NER revision mismatch for {task}")
        expected_rows = int(evidence["rows"])
        if phase == "validation":
            expected_rows = int(protocol["tasks"]["ner"]["validation_rows"][language])
        if len(rows) != expected_rows:
            raise ValueError(f"NER row count mismatch for {task}")
        ids = {
            (str(row["doc_id"]), row["doc_hash"], row["target_hash"])
            for row in rows
        }
        if len(ids) != expected_rows:
            raise ValueError(f"Duplicate or unpaired NER rows for {task}")
        score = span_f1(rows)
        assert_close(score, float(payload["reported_metrics"][task]), task)
        scores[task] = score
        identities[task] = ids
    return scores, identities


def verify_pos(
    payload: dict[str, Any],
    protocol: dict[str, Any],
    phase: str,
) -> tuple[dict[str, float], dict[str, set[tuple[str, str]]]]:
    task_protocol = protocol["tasks"]["pos"]
    if payload.get("interface") != "closed_label_tuple_mean_logprob_v1":
        raise ValueError("POS interface mismatch")
    if payload.get("score_mode") != "mean" or payload.get("pad_to_multiple_of") != 64:
        raise ValueError("POS scoring contract mismatch")
    if payload.get("label_set") != task_protocol["labels"]:
        raise ValueError("POS label set mismatch")
    allowed_labels = set(task_protocol["labels"])
    if set(payload.get("task_evidence", {})) != set(LANGUAGES):
        raise ValueError("POS evidence language coverage mismatch")
    for language, evidence in payload["task_evidence"].items():
        if evidence.get("split") != phase:
            raise ValueError(f"POS split mismatch for {language}")
        if evidence.get("dataset_revision") != task_protocol["revision"]:
            raise ValueError(f"POS revision mismatch for {language}")
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in payload["rows"]:
        cell = f"{row['language']}/P{int(row['prompt'])}"
        grouped[cell].append(row)
    if set(grouped) != set(payload["reported_metrics"]):
        raise ValueError("POS cell coverage mismatch")
    if phase == "validation":
        expected_cells = {
            f"{language}/P{prompt}"
            for language in LANGUAGES
            for prompt in range(1, 5)
        }
        if set(grouped) != expected_cells:
            raise ValueError("POS validation prompt coverage mismatch")

    scores: dict[str, float] = {}
    identities: dict[str, set[tuple[str, str]]] = {}
    for cell, rows in grouped.items():
        language = cell.split("/", 1)[0]
        expected_rows = int(payload["task_evidence"][language]["rows"])
        if phase == "validation":
            expected_rows = int(task_protocol["validation_rows"][language])
        if len(rows) != expected_rows:
            raise ValueError(f"POS row count mismatch for {cell}")
        correct = 0
        total = 0
        ids: set[tuple[str, str]] = set()
        for row in rows:
            tokens = [str(value) for value in row["tokens"]]
            gold = [str(value) for value in row["gold"]]
            prediction = [str(value) for value in row["prediction"]]
            if not tokens or len(tokens) != len(gold) or len(gold) != len(prediction):
                raise ValueError(f"POS sequence length mismatch for {cell}")
            if any(label not in allowed_labels for label in gold + prediction):
                raise ValueError(f"Illegal POS label for {cell}")
            expected_mask = [
                predicted == target
                for predicted, target in zip(prediction, gold, strict=True)
            ]
            if row["correct"] != expected_mask:
                raise ValueError(f"POS correctness mask mismatch for {cell}")
            if int(row["maximum_forward_tokens"]) > 1024:
                raise ValueError(f"POS input cap exceeded for {cell}")
            if len(row["selected_mean_logprobs"]) != len(tokens) or any(
                not math.isfinite(float(value))
                for value in row["selected_mean_logprobs"]
            ):
                raise ValueError(f"Invalid POS scores for {cell}")
            source = {
                "id": str(row["id"]),
                "tokens": tokens,
                "gold": gold,
            }
            if canonical_sha256(source) != row["source_sha256"]:
                raise ValueError(f"POS source hash mismatch for {cell}")
            ids.add((str(row["id"]), row["source_sha256"]))
            correct += sum(expected_mask)
            total += len(tokens)
        if len(ids) != expected_rows or total <= 0:
            raise ValueError(f"Duplicate or empty POS rows for {cell}")
        if phase == "validation":
            expected_tokens = int(task_protocol["validation_tokens"][language])
            if total != expected_tokens:
                raise ValueError(f"POS token count mismatch for {cell}")
        reported = payload["reported_metrics"][cell]
        if int(reported["correct"]) != correct or int(reported["total"]) != total:
            raise ValueError(f"POS reported counts mismatch for {cell}")
        accuracy = correct / total
        assert_close(accuracy, float(reported["token_accuracy"]), cell)
        scores[cell] = accuracy
        identities[cell] = ids
    return scores, identities


def select_prompts(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    protocol = json.loads(args.protocol.read_text())
    protocol_hash = sha256(args.protocol)
    payloads = {
        "ner": load_payloads(args.ner_dir),
        "pos": load_payloads(args.pos_dir),
    }
    scores: dict[str, dict[str, dict[str, float]]] = {"ner": {}, "pos": {}}
    identities: dict[str, dict[str, dict[str, set[Any]]]] = {"ner": {}, "pos": {}}
    for task in ("ner", "pos"):
        for architecture in ARCHITECTURES:
            payload = payloads[task][architecture]
            verify_common(
                payload,
                protocol,
                protocol_hash,
                architecture,
                task,
                "validation",
                None,
            )
            verifier = verify_ner if task == "ner" else verify_pos
            task_scores, task_identities = verifier(payload, protocol, "validation")
            scores[task][architecture] = task_scores
            identities[task][architecture] = task_identities
        reference = identities[task][ARCHITECTURES[0]]
        for architecture in ARCHITECTURES[1:]:
            if identities[task][architecture] != reference:
                raise ValueError(f"Unpaired {task} validation rows for {architecture}")

    selected: dict[str, dict[str, int]] = {"ner": {}, "pos": {}}
    evidence: dict[str, Any] = {"ner": {}, "pos": {}}
    for task in ("ner", "pos"):
        prompt_count = 5 if task == "ner" else 4
        for language in LANGUAGES:
            candidates: dict[str, Any] = {}
            for prompt in range(1, prompt_count + 1):
                key = (
                    ner_task_name(language, prompt, "validation")
                    if task == "ner"
                    else f"{language}/P{prompt}"
                )
                model_scores = {
                    architecture: scores[task][architecture][key]
                    for architecture in ARCHITECTURES
                }
                candidates[str(prompt)] = {
                    "models": model_scores,
                    "mean": fmean(model_scores.values()),
                }
            winner = min(
                range(1, prompt_count + 1),
                key=lambda prompt: (-candidates[str(prompt)]["mean"], prompt),
            )
            selected[task][language] = winner
            evidence[task][language] = candidates

    args.output_dir.mkdir(parents=True, exist_ok=False)
    input_hashes = {
        task: {
            architecture: sha256(
                (args.ner_dir if task == "ner" else args.pos_dir)
                / f"{architecture}.json"
            )
            for architecture in ARCHITECTURES
        }
        for task in ("ner", "pos")
    }
    verified = {
        "schema": "sallm.general_sequence_validation_verified/v1",
        "status": "VERIFIED",
        "data_boundary": "validation_only",
        "test_accessed": False,
        "protocol_sha256": protocol_hash,
        "input_sha256": input_hashes,
        "rows": {"ner": 4 * 10760, "pos": 4 * 1800},
    }
    verified_path = args.output_dir / "VALIDATION_VERIFIED.json"
    atomic_json(verified_path, verified)
    selection = {
        "schema": "sallm.general_sequence_selection/v1",
        "selection_data": "validation_only",
        "rule": protocol["selection"]["rule"],
        "tie_break": "lowest prompt id",
        "architectures": list(ARCHITECTURES),
        "maximum_input_tokens": 1024,
        "selected_prompts": selected,
        "validation_scores": evidence,
        "validation_verified_sha256": sha256(verified_path),
        "protocol_sha256": protocol_hash,
    }
    selection_path = args.output_dir / "SELECTION.json"
    atomic_json(selection_path, selection)
    ready = {
        "schema": "sallm.general_sequence_test_ready/v1",
        "status": "VALIDATION_SELECTION_FROZEN_HELDOUT_LOCKED",
        "selection_sha256": sha256(selection_path),
        "validation_verified_sha256": sha256(verified_path),
        "protocol_sha256": protocol_hash,
        "required_release_files": {
            "ner": "NER_TEST_ACCESS_RELEASED_V1.json",
            "pos": "POS_TEST_ACCESS_RELEASED_V1.json",
        },
        "no_score_based_retry": True,
    }
    ready_path = args.output_dir / "READY_FOR_TEST.json"
    atomic_json(ready_path, ready)
    manifest = args.output_dir / "ARTIFACTS.sha256"
    entries = [
        f"{sha256(path)}  {path.name}"
        for path in (verified_path, selection_path, ready_path)
    ]
    manifest.write_text("\n".join(entries) + "\n")
    print(json.dumps(selected, indent=2, sort_keys=True))


def verify_test(args: argparse.Namespace) -> None:
    protocol = json.loads(args.protocol.read_text())
    protocol_hash = sha256(args.protocol)
    selection_hash = sha256(args.selection)
    selection = json.loads(args.selection.read_text())
    ready = json.loads(args.ready.read_text())
    release = json.loads(args.release.read_text())
    if ready.get("status") != "VALIDATION_SELECTION_FROZEN_HELDOUT_LOCKED":
        raise ValueError("Validation-ready marker is invalid")
    if ready.get("selection_sha256") != selection_hash:
        raise ValueError("Selection changed after validation freeze")
    expected_release = {
        "schema": "sallm.general_sequence_test_release/v1",
        "task": args.task,
        "selection_sha256": selection_hash,
        "authorized": True,
        "no_score_based_retry": True,
    }
    if any(release.get(key) != value for key, value in expected_release.items()):
        raise ValueError("Task-specific held-out release is invalid")
    payloads = load_payloads(args.input_dir)
    scores: dict[str, dict[str, float]] = {}
    identities: dict[str, dict[str, set[Any]]] = {}
    verifier = verify_ner if args.task == "ner" else verify_pos
    for architecture, payload in payloads.items():
        verify_common(
            payload,
            protocol,
            protocol_hash,
            architecture,
            args.task,
            "test",
            selection_hash,
        )
        scores[architecture], identities[architecture] = verifier(
            payload, protocol, "test"
        )
    reference = identities[ARCHITECTURES[0]]
    for architecture in ARCHITECTURES[1:]:
        if identities[architecture] != reference:
            raise ValueError(f"Unpaired {args.task} held-out rows for {architecture}")
    expected_prompts = selection["selected_prompts"][args.task]
    expected_keys = {
        (
            ner_task_name(language, int(prompt), "test")
            if args.task == "ner"
            else f"{language}/P{int(prompt)}"
        )
        for language, prompt in expected_prompts.items()
    }
    for architecture, architecture_scores in scores.items():
        if set(architecture_scores) != expected_keys:
            raise ValueError(
                f"Selected {args.task} coverage mismatch for {architecture}"
            )
        for language, prompt in expected_prompts.items():
            expected_key = (
                ner_task_name(language, int(prompt), "test")
                if args.task == "ner"
                else f"{language}/P{int(prompt)}"
            )
            if expected_key not in architecture_scores:
                raise ValueError(
                    f"Missing selected {args.task} result for {architecture}/{language}"
                )
    verification = {
        "schema": "sallm.general_sequence_selected_test_verified/v1",
        "passed": True,
        "task": args.task,
        "selection_sha256": selection_hash,
        "protocol_sha256": protocol_hash,
        "no_score_based_retry": True,
        "scores": scores,
        "input_sha256": {
            architecture: sha256(args.input_dir / f"{architecture}.json")
            for architecture in ARCHITECTURES
        },
    }
    atomic_json(args.output, verification)


def self_check() -> None:
    rows = [
        {"target": "PER: Alice $$ LOC: Cape Town", "prediction": "PER: Alice"},
        {"target": "", "prediction": "LOC: Durban"},
    ]
    assert math.isclose(span_f1(rows), 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert spans("PERSON: Alice $ LOCATION: Cape Town") == [
        ("person", "alice"),
        ("location", "cape town"),
    ]
    assert ner_task_name("xho", 3, "test") == (
        "sallm_masakhaner_xh_prompt_3_test"
    )
    print("SELF_CHECK_OK")


def main() -> None:
    args = parse_args()
    if args.command == "self-check":
        self_check()
    elif args.command == "select":
        select_prompts(args)
    else:
        verify_test(args)


if __name__ == "__main__":
    main()
