#!/usr/bin/env python3
"""Freeze validation-selected prompts and verify the one-time General tests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from injongointent_trainonly_validation import MANIFEST_SCHEMA
from sklearn.metrics import f1_score

ARCHITECTURES = ("mzansilm", "mamba2", "xlstm", "gdn")
LANGUAGES = ("eng", "sot", "xho", "zul")
BELEBELE_LANGUAGES = (
    "afr",
    "eng",
    "sot",
    "ssw",
    "tsn",
    "tso",
    "xho",
    "zul",
)
INTENT_TEST_ROWS = {"eng": 622, "sot": 640, "xho": 640, "zul": 640}
VALIDATION_ROWS = {"afrixnli": 450, "afrimmlu": 83}
TEST_ROWS = {"afrixnli": 600, "afrimmlu": 500, "afrimgsm": 250}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--lm-dir", type=Path, required=True)
    select.add_argument("--intent-dir", type=Path, required=True)
    select.add_argument("--intent-validation-manifest", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify-test")
    verify.add_argument("--lm-dir", type=Path, required=True)
    verify.add_argument("--intent-dir", type=Path, required=True)
    verify.add_argument("--selection", type=Path, required=True)
    verify.add_argument("--ready", type=Path, required=True)
    verify.add_argument("--output-dir", type=Path, required=True)
    subparsers.add_parser("self-check")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_architectures(directory: Path) -> dict[str, dict[str, Any]]:
    payloads = {}
    for architecture in ARCHITECTURES:
        path = directory / f"{architecture}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payloads[architecture] = json.loads(path.read_text())
    return payloads


def assert_hash(value: Any, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Invalid {field}: {value!r}")
    int(value, 16)


def assert_close(left: float, right: float, context: str) -> None:
    if not (math.isfinite(left) and math.isfinite(right)):
        raise ValueError(f"Non-finite metric for {context}")
    if abs(left - right) > 1e-12:
        raise ValueError(f"Metric mismatch for {context}: {left} != {right}")


def validate_samples(
    samples: list[dict[str, Any]],
    expected_rows: int,
    *,
    filter_name: str | None = None,
) -> list[dict[str, Any]]:
    if filter_name is not None:
        samples = [row for row in samples if row.get("filter") == filter_name]
    if len(samples) != expected_rows:
        raise ValueError(f"Expected {expected_rows} samples, found {len(samples)}")
    doc_ids = set()
    for row in samples:
        doc_ids.add(str(row["doc_id"]))
        for field in ("doc_hash", "prompt_hash", "target_hash"):
            assert_hash(row.get(field), field)
    if len(doc_ids) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} unique document ids, found {len(doc_ids)}"
        )
    return samples


def validation_task_name(family: str, language: str, prompt: int) -> str:
    stem = "afrixnli" if family == "afrixnli" else "afrimmlu_direct"
    return f"sallm_val_{stem}_{language}_prompt_{prompt}"


def intent_prompt(prompt: int) -> str:
    return f"injongointent_intent_classification/lm_eval_p{prompt}"


def verify_lm_validation(payload: dict[str, Any], architecture: str) -> None:
    if payload.get("phase") != "validation" or payload.get("limit") is not None:
        raise ValueError(f"{architecture} is not a full validation result")
    if payload.get("architecture") != architecture:
        raise ValueError(f"Architecture mismatch for {architecture}")
    if payload.get("maximum_input_tokens") != 1024:
        raise ValueError(f"{architecture} did not use max_length=1024")
    for field in ("checkpoint_config_sha256", "adapter_config_sha256"):
        assert_hash(payload.get(field), f"{architecture}/{field}")
    call = payload["calls"]["validation"]
    expected_tasks = {
        validation_task_name(family, language, prompt)
        for family in VALIDATION_ROWS
        for language in LANGUAGES
        for prompt in range(1, 6)
    }
    if set(call["results"]) != expected_tasks or set(call["samples"]) != expected_tasks:
        raise ValueError(f"Validation task coverage mismatch for {architecture}")
    if set(payload.get("task_evidence", {})) != expected_tasks:
        raise ValueError(f"Validation evidence coverage mismatch for {architecture}")
    for family, expected_rows in VALIDATION_ROWS.items():
        for language in LANGUAGES:
            for prompt in range(1, 6):
                task = validation_task_name(family, language, prompt)
                rows = validate_samples(call["samples"][task], expected_rows)
                evidence = payload["task_evidence"][task]
                if evidence.get("split") != "validation":
                    raise ValueError(f"Non-validation task evidence for {task}")
                if int(evidence.get("rows", -1)) != expected_rows:
                    raise ValueError(f"Task evidence row mismatch for {task}")
                assert_hash(evidence.get("yaml_sha256"), f"{task}/yaml_sha256")
                if not isinstance(evidence.get("fingerprint"), str):
                    raise ValueError(f"Missing dataset fingerprint for {task}")
                recomputed = fmean(float(row["acc"]) for row in rows)
                reported = float(call["results"][task]["acc,none"])
                assert_close(recomputed, reported, f"{architecture}/{task}")


def verify_intent_validation(
    payload: dict[str, Any],
    architecture: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
) -> None:
    if payload.get("schema") != "sallm.injongointent_mean_eval/v2":
        raise ValueError(f"Unexpected Intent result schema for {architecture}")
    if payload.get("architecture") != architecture:
        raise ValueError(f"Intent architecture mismatch for {architecture}")
    if payload.get("split") != "validation":
        raise ValueError(f"Intent result is not validation-only for {architecture}")
    if payload.get("maximum_input_tokens") != 1024:
        raise ValueError(f"Intent max length mismatch for {architecture}")
    if payload.get("score_mode") != "mean_token_logprob":
        raise ValueError(f"Intent score mode mismatch for {architecture}")
    if payload.get("data_boundary") != "pinned_train_only_validation":
        raise ValueError(f"Intent validation is not train-only for {architecture}")
    if payload.get("held_out_data_accessed") is not False:
        raise ValueError(f"Intent validation accessed held-out data for {architecture}")
    if payload.get("validation_manifest_sha256") != manifest_sha256:
        raise ValueError(f"Intent validation manifest mismatch for {architecture}")
    validation_data = payload.get("validation_data", {})
    if validation_data.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"Intent validation evidence mismatch for {architecture}")
    if validation_data.get("source_split") != "train":
        raise ValueError(f"Intent validation source is not train for {architecture}")
    if validation_data.get("held_out_data_accessed") is not False:
        raise ValueError(f"Intent validation evidence accessed test for {architecture}")
    if validation_data.get("architecture_blind") is not True:
        raise ValueError(f"Intent split is not architecture-blind for {architecture}")
    if payload.get("template_selection") is not None:
        raise ValueError(
            f"Intent validation used a frozen test selection for {architecture}"
        )
    rows = payload["rows"]
    expected_counts = {
        language: int(values["validation_row_count"])
        for language, values in manifest["languages"].items()
    }
    expected_records = {
        language: {
            (int(record["source_index"]), str(record["row_sha256"]))
            for record in values["validation_records"]
        }
        for language, values in manifest["languages"].items()
    }
    expected = sum(expected_counts.values()) * 5
    if len(rows) != expected:
        raise ValueError(
            f"Intent validation expected {expected} rows, found {len(rows)}"
        )
    seen = set()
    counts = Counter()
    for row in rows:
        language = str(row["lang"])
        template = str(row["template_id"])
        key = (language, template, str(row["example_id"]))
        if key in seen:
            raise ValueError(f"Duplicate Intent validation row {key}")
        seen.add(key)
        assert_hash(row.get("rendered_prompt_sha256"), "rendered_prompt_sha256")
        assert_hash(row.get("validation_row_sha256"), "validation_row_sha256")
        if not isinstance(row.get("validation_source_index"), int):
            raise ValueError("Intent row is missing its frozen training index")
        counts[(language, template)] += 1
    for language, expected_rows in expected_counts.items():
        for prompt in range(1, 6):
            template = intent_prompt(prompt)
            if counts[(language, template)] != expected_rows:
                raise ValueError(
                    f"Intent {language}/{template} expected {expected_rows} rows"
                )
            prompt_rows = [
                row
                for row in rows
                if str(row["lang"]) == language and str(row["template_id"]) == template
            ]
            observed_records = {
                (
                    int(row["validation_source_index"]),
                    str(row["validation_row_sha256"]),
                )
                for row in prompt_rows
            }
            if observed_records != expected_records[language]:
                raise ValueError(
                    "Intent frozen validation identities changed for "
                    f"{language}/{template}"
                )
            recomputed = float(
                f1_score(
                    [str(row["gold"]) for row in prompt_rows],
                    [str(row["prediction"]) for row in prompt_rows],
                    average="weighted",
                )
            )
            reported = float(
                payload["summary"]["languages"][language]["prompts"][template]["f1"]
            )
            assert_close(
                recomputed, reported, f"{architecture}/intent/{language}/{template}"
            )


def assert_paired_validation_inputs(
    lm_payloads: dict[str, dict[str, Any]],
    intent_payloads: dict[str, dict[str, Any]],
) -> None:
    reference = ARCHITECTURES[0]
    reference_lm = lm_payloads[reference]["calls"]["validation"]["samples"]
    for architecture in ARCHITECTURES[1:]:
        samples = lm_payloads[architecture]["calls"]["validation"]["samples"]
        for task, reference_rows in reference_lm.items():
            reference_ids = {
                (str(row["doc_id"]), str(row["doc_hash"]), str(row["target_hash"]))
                for row in reference_rows
            }
            observed_ids = {
                (str(row["doc_id"]), str(row["doc_hash"]), str(row["target_hash"]))
                for row in samples[task]
            }
            if observed_ids != reference_ids:
                raise ValueError(
                    f"Unpaired validation documents for {architecture}/{task}"
                )
    reference_intent = {
        (
            str(row["lang"]),
            str(row["template_id"]),
            str(row["example_id"]),
            str(row["gold"]),
            int(row["validation_source_index"]),
            str(row["validation_row_sha256"]),
        )
        for row in intent_payloads[reference]["rows"]
    }
    for architecture in ARCHITECTURES[1:]:
        observed = {
            (
                str(row["lang"]),
                str(row["template_id"]),
                str(row["example_id"]),
                str(row["gold"]),
                int(row["validation_source_index"]),
                str(row["validation_row_sha256"]),
            )
            for row in intent_payloads[architecture]["rows"]
        }
        if observed != reference_intent:
            raise ValueError(f"Unpaired Intent validation rows for {architecture}")


def select_prompts(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    lm_payloads = load_architectures(args.lm_dir)
    intent_payloads = load_architectures(args.intent_dir)
    intent_manifest = json.loads(args.intent_validation_manifest.read_text())
    if intent_manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Unexpected Intent train-only validation manifest")
    if intent_manifest.get("source_split") != "train":
        raise ValueError("Intent prompt selection did not bind the train split")
    if intent_manifest.get("held_out_data_accessed") is not False:
        raise ValueError("Intent validation manifest accessed held-out data")
    if intent_manifest.get("architecture_blind") is not True:
        raise ValueError("Intent validation split is not architecture-blind")
    intent_manifest_sha256 = file_sha256(args.intent_validation_manifest)
    for architecture in ARCHITECTURES:
        verify_lm_validation(lm_payloads[architecture], architecture)
        verify_intent_validation(
            intent_payloads[architecture],
            architecture,
            intent_manifest,
            intent_manifest_sha256,
        )
    assert_paired_validation_inputs(lm_payloads, intent_payloads)

    score_evidence: dict[str, Any] = {}
    selected: dict[str, dict[str, Any]] = {}
    for family in ("afrixnli", "afrimmlu"):
        score_evidence[family] = {}
        selected[family] = {}
        for language in LANGUAGES:
            prompt_scores = {}
            for prompt in range(1, 6):
                task = validation_task_name(family, language, prompt)
                by_architecture = {
                    architecture: float(
                        lm_payloads[architecture]["calls"]["validation"]["results"][
                            task
                        ]["acc,none"]
                    )
                    for architecture in ARCHITECTURES
                }
                prompt_scores[str(prompt)] = {
                    "models": by_architecture,
                    "mean": fmean(by_architecture.values()),
                }
            winner = min(
                range(1, 6),
                key=lambda prompt: (-prompt_scores[str(prompt)]["mean"], prompt),
            )
            selected[family][language] = winner
            score_evidence[family][language] = prompt_scores

    score_evidence["intent"] = {}
    selected["intent"] = {}
    for language in LANGUAGES:
        prompt_scores = {}
        for prompt in range(1, 6):
            template = intent_prompt(prompt)
            by_architecture = {
                architecture: float(
                    intent_payloads[architecture]["summary"]["languages"][language][
                        "prompts"
                    ][template]["f1"]
                )
                for architecture in ARCHITECTURES
            }
            prompt_scores[str(prompt)] = {
                "models": by_architecture,
                "mean": fmean(by_architecture.values()),
            }
        winner = min(
            range(1, 6),
            key=lambda prompt: (-prompt_scores[str(prompt)]["mean"], prompt),
        )
        selected["intent"][language] = intent_prompt(winner)
        score_evidence["intent"][language] = prompt_scores

    input_hashes = {
        family: {
            architecture: file_sha256(directory / f"{architecture}.json")
            for architecture in ARCHITECTURES
        }
        for family, directory in (
            ("lm_eval", args.lm_dir),
            ("intent", args.intent_dir),
        )
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    validation_verified = {
        "schema": "sallm.general_prompt_validation_verified/v2",
        "status": "VERIFIED",
        "data_boundary": "validation_selection_only",
        "test_predictions_accessed": False,
        "intent_held_out_data_accessed": False,
        "intent_validation_source": "pinned_train_only",
        "intent_validation_manifest_sha256": intent_manifest_sha256,
        "intent_split_architecture_blind": True,
        "maximum_input_tokens": 1024,
        "architectures": list(ARCHITECTURES),
        "input_sha256": input_hashes,
    }
    validation_path = args.output_dir / "VALIDATION_VERIFIED.json"
    validation_path.write_text(json.dumps(validation_verified, indent=2) + "\n")
    selection = {
        "schema": "sallm.general_prompt_selection/v2",
        "selected_at_utc": datetime.now(UTC).isoformat(),
        "selection_data": "validation_only",
        "intent_validation_manifest_sha256": intent_manifest_sha256,
        "intent_split_architecture_blind": True,
        "architecture_symmetric_prompt_rule": True,
        "rule": (
            "per task-language maximize the unweighted mean validation metric "
            "across MzansiLM, Mamba, xLSTM and GDN; lowest prompt id breaks ties"
        ),
        "maximum_input_tokens": 1024,
        "selected_prompts": selected,
        "validation_scores": score_evidence,
        "validation_verified_sha256": file_sha256(validation_path),
    }
    selection_path = args.output_dir / "SELECTION.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    intent_path = args.output_dir / "intent_selection.json"
    intent_path.write_text(json.dumps(selected["intent"], indent=2) + "\n")
    ready = {
        "schema": "sallm.general_prompt_test_ready/v2",
        "status": "READY_FOR_ONE_TIME_TEST",
        "selection_sha256": file_sha256(selection_path),
        "intent_selection_sha256": file_sha256(intent_path),
        "validation_verified_sha256": file_sha256(validation_path),
        "intent_validation_manifest_sha256": intent_manifest_sha256,
        "retry_policy": "no model, prompt, metric, or score-based retry",
    }
    ready_path = args.output_dir / "READY_FOR_TEST.json"
    ready_path.write_text(json.dumps(ready, indent=2) + "\n")
    write_artifact_manifest(args.output_dir)
    print(json.dumps(selected, indent=2))


def task_name(family: str, language: str, prompt: int) -> str:
    if family == "afrixnli":
        return f"afrixnli_{language}_prompt_{prompt}"
    if family == "afrimmlu":
        return f"afrimmlu_direct_{language}_prompt_{prompt}"
    if family == "afrimgsm":
        return f"afrimgsm_{language}_prompt_1"
    if family == "belebele":
        return f"belebele_{language}_prompt_1"
    raise ValueError(family)


def verify_lm_test(
    payload: dict[str, Any],
    architecture: str,
    selection: dict[str, Any],
) -> list[dict[str, Any]]:
    if payload.get("phase") != "test" or payload.get("limit") is not None:
        raise ValueError(f"{architecture} is not a full test result")
    if payload.get("architecture") != architecture:
        raise ValueError(f"Test architecture mismatch for {architecture}")
    if payload.get("maximum_input_tokens") != 1024:
        raise ValueError(f"Test max length mismatch for {architecture}")
    for field in ("checkpoint_config_sha256", "adapter_config_sha256"):
        assert_hash(payload.get(field), f"{architecture}/{field}")
    rows_out = []
    selected = selection["selected_prompts"]
    closed = payload["calls"]["closed_and_generation"]
    belebele = payload["calls"]["belebele"]
    expected_closed = {
        task_name(family, language, int(selected[family][language]))
        for family in ("afrixnli", "afrimmlu")
        for language in LANGUAGES
    } | {task_name("afrimgsm", language, 1) for language in LANGUAGES}
    expected_belebele = {
        task_name("belebele", language, 1) for language in BELEBELE_LANGUAGES
    }
    if (
        set(closed["results"]) != expected_closed
        or set(closed["samples"]) != expected_closed
    ):
        raise ValueError(f"Closed/generation test coverage mismatch for {architecture}")
    if (
        set(belebele["results"]) != expected_belebele
        or set(belebele["samples"]) != expected_belebele
    ):
        raise ValueError(f"Belebele test coverage mismatch for {architecture}")
    if set(payload.get("task_evidence", {})) != expected_closed | expected_belebele:
        raise ValueError(f"Test evidence coverage mismatch for {architecture}")
    for task, evidence in payload["task_evidence"].items():
        assert_hash(evidence.get("yaml_sha256"), f"{task}/yaml_sha256")
    for family in ("afrixnli", "afrimmlu"):
        for language in LANGUAGES:
            prompt = int(selected[family][language])
            task = task_name(family, language, prompt)
            samples = validate_samples(closed["samples"][task], TEST_ROWS[family])
            recomputed = fmean(float(row["acc"]) for row in samples)
            reported = float(closed["results"][task]["acc,none"])
            assert_close(recomputed, reported, f"{architecture}/{task}")
            rows_out.append(
                {
                    "architecture": architecture,
                    "task": family,
                    "language": language,
                    "prompt": prompt,
                    "metric": "accuracy",
                    "score": reported,
                    "rows": len(samples),
                }
            )
    for language in LANGUAGES:
        task = task_name("afrimgsm", language, 1)
        samples = validate_samples(
            closed["samples"][task],
            TEST_ROWS["afrimgsm"],
            filter_name="flexible-extract",
        )
        recomputed = fmean(float(row["exact_match"]) for row in samples)
        reported = float(closed["results"][task]["exact_match,flexible-extract"])
        assert_close(recomputed, reported, f"{architecture}/{task}")
        rows_out.append(
            {
                "architecture": architecture,
                "task": "afrimgsm",
                "language": language,
                "prompt": 1,
                "metric": "flexible_exact_match",
                "score": reported,
                "rows": len(samples),
            }
        )
    for language in BELEBELE_LANGUAGES:
        task = task_name("belebele", language, 1)
        samples = validate_samples(belebele["samples"][task], 900)
        recomputed = fmean(float(row["acc_norm"]) for row in samples)
        reported = float(belebele["results"][task]["acc_norm,none"])
        assert_close(recomputed, reported, f"{architecture}/{task}")
        rows_out.append(
            {
                "architecture": architecture,
                "task": "belebele",
                "language": language,
                "prompt": 1,
                "metric": "normalized_accuracy",
                "score": reported,
                "rows": len(samples),
            }
        )
    return rows_out


def verify_intent_test(
    payload: dict[str, Any],
    architecture: str,
    selection: dict[str, Any],
) -> list[dict[str, Any]]:
    if payload.get("architecture") != architecture or payload.get("split") != "test":
        raise ValueError(f"Invalid Intent test result for {architecture}")
    if payload.get("maximum_input_tokens") != 1024:
        raise ValueError(f"Intent test max length mismatch for {architecture}")
    if payload.get("score_mode") != "mean_token_logprob":
        raise ValueError(f"Intent test score mode mismatch for {architecture}")
    expected_selection = selection["selected_prompts"]["intent"]
    if payload.get("template_selection") != expected_selection:
        raise ValueError(f"Intent prompt selection mismatch for {architecture}")
    expected_total = sum(INTENT_TEST_ROWS.values())
    if len(payload["rows"]) != expected_total:
        raise ValueError(f"Intent test row count mismatch for {architecture}")
    output = []
    for language, expected_rows in INTENT_TEST_ROWS.items():
        template = expected_selection[language]
        rows = [row for row in payload["rows"] if str(row["lang"]) == language]
        if len(rows) != expected_rows:
            raise ValueError(f"Intent test row count mismatch for {language}")
        if {str(row["template_id"]) for row in rows} != {template}:
            raise ValueError(f"Intent template mismatch for {language}")
        ids = set()
        for row in rows:
            ids.add(str(row["example_id"]))
            assert_hash(row.get("rendered_prompt_sha256"), "rendered_prompt_sha256")
        if len(ids) != expected_rows:
            raise ValueError(f"Duplicate Intent ids for {language}")
        gold = [str(row["gold"]) for row in rows]
        predictions = [str(row["prediction"]) for row in rows]
        score = float(f1_score(gold, predictions, average="weighted"))
        reported = float(
            payload["summary"]["languages"][language]["prompts"][template]["f1"]
        )
        assert_close(score, reported, f"{architecture}/intent/{language}")
        output.append(
            {
                "architecture": architecture,
                "task": "intent",
                "language": language,
                "prompt": template.rsplit("p", 1)[-1],
                "metric": "support_weighted_f1",
                "score": score,
                "rows": len(rows),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_artifact_manifest(root: Path) -> None:
    path = root / "ARTIFACTS.sha256"
    entries = []
    for artifact in sorted(item for item in root.rglob("*") if item.is_file()):
        if artifact == path:
            continue
        entries.append(f"{file_sha256(artifact)}  {artifact.relative_to(root)}")
    path.write_text("\n".join(entries) + "\n")


def verify_test(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    selection = json.loads(args.selection.read_text())
    ready = json.loads(args.ready.read_text())
    if ready.get("status") != "READY_FOR_ONE_TIME_TEST":
        raise ValueError("Missing frozen test authorization")
    if ready.get("selection_sha256") != file_sha256(args.selection):
        raise ValueError("Selection changed after validation freeze")
    lm_payloads = load_architectures(args.lm_dir)
    intent_payloads = load_architectures(args.intent_dir)
    language_rows = []
    for architecture in ARCHITECTURES:
        if lm_payloads[architecture].get("selection_sha256") != file_sha256(
            args.selection
        ):
            raise ValueError(f"LM test selection hash mismatch for {architecture}")
        language_rows.extend(
            verify_lm_test(lm_payloads[architecture], architecture, selection)
        )
        language_rows.extend(
            verify_intent_test(intent_payloads[architecture], architecture, selection)
        )
    task_rows = []
    for architecture in ARCHITECTURES:
        for task in ("intent", "afrixnli", "afrimmlu", "afrimgsm", "belebele"):
            scores = [
                float(row["score"])
                for row in language_rows
                if row["architecture"] == architecture and row["task"] == task
            ]
            task_rows.append(
                {
                    "architecture": architecture,
                    "task": task,
                    "score": fmean(scores),
                    "language_count": len(scores),
                }
            )
    input_hashes = {
        family: {
            architecture: file_sha256(directory / f"{architecture}.json")
            for architecture in ARCHITECTURES
        }
        for family, directory in (("lm_eval", args.lm_dir), ("intent", args.intent_dir))
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(
        args.output_dir / "language_scores.csv",
        language_rows,
        ["architecture", "task", "language", "prompt", "metric", "score", "rows"],
    )
    write_csv(
        args.output_dir / "task_means.csv",
        task_rows,
        ["architecture", "task", "score", "language_count"],
    )
    summary = {
        "schema": "sallm.general_prompt_corrected_summary/v1",
        "aggregation": "task-language metric, then unweighted mean across languages",
        "language_scores": language_rows,
        "task_means": task_rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    verified = {
        "schema": "sallm.general_prompt_corrected_verified/v1",
        "status": "VERIFIED",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "architectures": list(ARCHITECTURES),
        "tasks": ["intent", "afrixnli", "afrimmlu", "afrimgsm", "belebele"],
        "maximum_input_tokens": 1024,
        "selection_sha256": file_sha256(args.selection),
        "ready_sha256": file_sha256(args.ready),
        "input_sha256": input_hashes,
        "summary_sha256": file_sha256(summary_path),
        "retry_policy": "no score-based retry",
    }
    (args.output_dir / "VERIFIED.json").write_text(
        json.dumps(verified, indent=2) + "\n"
    )
    write_artifact_manifest(args.output_dir)
    print(json.dumps(task_rows, indent=2))


def self_check() -> None:
    assert len(BELEBELE_LANGUAGES) == 8
    assert "tso" in BELEBELE_LANGUAGES
    assert validation_task_name("afrixnli", "eng", 3) == (
        "sallm_val_afrixnli_eng_prompt_3"
    )
    assert validation_task_name("afrimmlu", "zul", 5) == (
        "sallm_val_afrimmlu_direct_zul_prompt_5"
    )
    values = {1: 0.2, 2: 0.3, 3: 0.3}
    winner = min(values, key=lambda prompt: (-values[prompt], prompt))
    assert winner == 2


def main() -> None:
    args = parse_args()
    if args.command == "self-check":
        self_check()
        print("SELF_CHECK_OK")
    elif args.command == "select":
        select_prompts(args)
    else:
        verify_test(args)


if __name__ == "__main__":
    main()
