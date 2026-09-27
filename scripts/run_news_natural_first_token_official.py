#!/usr/bin/env python3
"""Run the frozen corrected General MasakhaNEWS held-out replacement once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ARCHITECTURES = ("mzansilm", "mamba", "xlstm", "gdn")
DISPLAY_NAMES = {
    "mzansilm": "MzansiLM",
    "mamba": "Mamba",
    "xlstm": "xLSTM",
    "gdn": "GDN",
}
LANGUAGE_ROWS = {"eng": 948, "xho": 297}
LABELS = (
    "business",
    "entertainment",
    "health",
    "politics",
    "religion",
    "sports",
    "technology",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validated_ref(ref: dict[str, str], *, name: str) -> Path:
    require(set(ref) == {"path", "sha256"}, f"malformed file reference: {name}")
    path = Path(ref["path"]).resolve()
    require(path.is_file(), f"missing frozen file: {name}")
    require(sha256_file(path) == ref["sha256"], f"frozen file changed: {name}")
    return path


def _validate_snapshot_manifest(
    *, snapshot: Path, manifest_path: Path, expected_sha256: str
) -> None:
    require(
        sha256_file(manifest_path) == expected_sha256,
        "runtime snapshot manifest changed",
    )
    snapshot = snapshot.resolve()
    seen: set[str] = set()
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = raw_line.partition("  ")
        require(bool(separator) and len(digest) == 64, "malformed snapshot manifest")
        require(relative not in seen, "duplicate snapshot manifest entry")
        seen.add(relative)
        path = (snapshot / relative).resolve()
        require(path.is_relative_to(snapshot), "snapshot entry escapes its root")
        require(path.is_file(), f"snapshot file missing: {relative}")
        require(sha256_file(path) == digest, f"snapshot file changed: {relative}")
    require(bool(seen), "empty snapshot manifest")


def _validate_shared_selection(
    *, selection: dict[str, Any], summaries: dict[str, dict[str, Any]]
) -> dict[str, str]:
    require(
        selection.get("schema") == "sallm.news_general_shared_prompt_selection/v1",
        "unexpected shared prompt selection",
    )
    require(
        selection.get("data_boundary") == "validation_only", "selection split drift"
    )
    require(selection.get("test_accessed") is False, "selection accessed test")
    require(
        tuple(selection.get("architectures", ())) == ARCHITECTURES,
        "architecture order drift",
    )
    require(
        set(selection.get("selections", {})) == set(LANGUAGE_ROWS),
        "selection language drift",
    )

    recomputed_candidates: dict[str, dict[str, float]] = {}
    selected: dict[str, str] = {}
    for language in LANGUAGE_ROWS:
        recomputed_candidates[language] = {}
        for prompt_number in range(1, 6):
            prompt = f"p{prompt_number}"
            values = [
                float(
                    summaries[architecture]["subsets"][f"{language}/{prompt}"][
                        "first_token"
                    ]["weighted_f1"]
                )
                for architecture in ARCHITECTURES
            ]
            mean = sum(values) / len(values)
            recomputed_candidates[language][prompt] = mean
            recorded = float(
                selection["candidates"][language][prompt]["mean_weighted_f1"]
            )
            require(
                math.isclose(mean, recorded, rel_tol=0.0, abs_tol=1e-15),
                f"shared validation mean mismatch: {language}/{prompt}",
            )
        winner = max(
            recomputed_candidates[language],
            key=lambda prompt: (
                recomputed_candidates[language][prompt],
                -int(prompt[1:]),
            ),
        )
        require(
            selection["selections"][language]["prompt"] == winner,
            f"shared validation selection mismatch: {language}",
        )
        selected[language] = winner
    return selected


def load_and_validate_protocol(
    *, protocol_path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(
        protocol.get("schema") == "sallm.news_general_official_execution/v1",
        "unexpected News official execution protocol",
    )
    require(
        protocol.get("status") == "READY_FOR_SINGLE_CORRECTED_REPLACEMENT",
        "official execution is not authorized",
    )
    require(protocol.get("data_boundary") == "held_out_test", "wrong data boundary")
    require(protocol.get("split") == "test", "wrong evaluation split")
    require(protocol.get("regime") == "General", "wrong adaptation regime")
    require(protocol.get("no_score_based_retry") is True, "retry policy drift")
    require(
        set(protocol.get("models", {})) == set(ARCHITECTURES), "model coverage drift"
    )

    snapshot = Path(protocol["runtime"]["snapshot"]).resolve()
    require(repo.resolve() == snapshot, "runtime snapshot path drift")
    manifest_ref = protocol["runtime"]["snapshot_manifest"]
    snapshot_manifest = Path(manifest_ref["path"]).resolve()
    require(
        snapshot_manifest == snapshot / "SNAPSHOT_PAYLOAD.sha256",
        "unexpected snapshot manifest path",
    )
    _validate_snapshot_manifest(
        snapshot=snapshot,
        manifest_path=snapshot_manifest,
        expected_sha256=manifest_ref["sha256"],
    )
    require(protocol["runtime"].get("visible_gpu") == "1", "GPU assignment drift")
    require(protocol["runtime"].get("device") == "cuda:0", "device drift")
    require(protocol["runtime"].get("dtype") == "bfloat16", "dtype drift")
    require(
        protocol["runtime"].get("maximum_input_tokens") == 1024, "context cap drift"
    )
    require(
        protocol["runtime"].get("central_metric")
        == "singleton_natural_first_label_token_log_probability",
        "central score contract drift",
    )
    require(protocol["runtime"].get("labels") == list(LABELS), "label order drift")
    require(
        protocol["runtime"].get("exact_tie_break") == "first_label_in_frozen_order",
        "exact-tie policy drift",
    )

    validation_protocol_path = _validated_ref(
        protocol["validation_protocol"], name="validation protocol"
    )
    require(
        validation_protocol_path
        == snapshot / ".audit/mamba_news_common_validation_protocol_20260914.json",
        "validation protocol path drift",
    )
    chain_path = _validated_ref(
        protocol["validation_gate"]["chain_verification"],
        name="validation-chain verification",
    )
    selection_path = _validated_ref(
        protocol["validation_gate"]["prompt_selection"],
        name="shared prompt selection",
    )
    tie_sensitivity_path = _validated_ref(
        protocol["validation_gate"]["tie_sensitivity"],
        name="validation tie-sensitivity audit",
    )
    chain = json.loads(chain_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    tie_sensitivity = json.loads(tie_sensitivity_path.read_text(encoding="utf-8"))
    require(
        chain.get("status") == "VALIDATION_CHAIN_VERIFIED", "validation chain failed"
    )
    require(
        chain.get("data_boundary") == "validation_only", "validation boundary drift"
    )
    require(chain.get("test_accessed") is False, "validation chain accessed test")
    require(
        chain.get("shared_prompt_selection_sha256") == sha256_file(selection_path),
        "validation chain does not bind prompt selection",
    )
    require(
        tie_sensitivity.get("status") == "VALIDATION_CHAIN_INDEPENDENTLY_VERIFIED",
        "validation tie-sensitivity audit failed",
    )
    require(tie_sensitivity.get("test_accessed") is False, "tie audit accessed test")
    require(
        tuple(tie_sensitivity["tie_policy"]["frozen_order"]) == LABELS,
        "tie-audit label order drift",
    )
    require(
        tie_sensitivity["tie_policy"]["shared_prompt_changes_under_reverse"] is False,
        "reverse tie order changes prompt selection",
    )
    require(
        tie_sensitivity["tie_policy"]["architecture_ranking_changes_under_reverse"]
        is False,
        "reverse tie order changes architecture ranking",
    )

    summaries: dict[str, dict[str, Any]] = {}
    for architecture in ARCHITECTURES:
        path = _validated_ref(
            protocol["validation_summaries"][architecture],
            name=f"{architecture} validation summary",
        )
        require(
            selection["input_summary_sha256"][architecture] == sha256_file(path),
            f"selection does not bind {architecture} summary",
        )
        summary = json.loads(path.read_text(encoding="utf-8"))
        require(summary.get("arm") == "general", f"{architecture} regime drift")
        require(
            summary.get("records") == summary.get("expected_records") == 3095,
            f"{architecture} validation summary incomplete",
        )
        summaries[architecture] = summary
    selected_prompts = _validate_shared_selection(
        selection=selection,
        summaries=summaries,
    )
    require(
        selected_prompts == protocol.get("selected_prompts"),
        "execution prompts differ from validation selection",
    )
    recorded_tie_counts = protocol.get("validation_tie_counts")
    require(
        recorded_tie_counts
        == {
            architecture: {
                "all_prompts": tie_sensitivity["reports"][architecture][
                    "central_top_score_ties"
                ],
                "selected_prompt": {
                    language: tie_sensitivity["reports"][architecture][
                        "central_top_score_ties_by_subset"
                    ][f"{language}/{selected_prompts[language]}"]
                    for language in LANGUAGE_ROWS
                },
            }
            for architecture in ARCHITECTURES
        },
        "validation tie-count transcription drift",
    )

    access_protocol_path = _validated_ref(
        protocol["test_access"]["protocol"], name="test access protocol"
    )
    authorization_path = _validated_ref(
        protocol["test_access"]["authorization"],
        name="held-out authorization marker",
    )
    access_started_path = _validated_ref(
        protocol["test_access"]["access_started"], name="test access marker"
    )
    access_protocol = json.loads(access_protocol_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    access_started = json.loads(access_started_path.read_text(encoding="utf-8"))
    require(
        access_protocol.get("schema") == "sallm.news_official_test_access/v1",
        "unexpected test access protocol",
    )
    require(
        access_protocol.get("selected_prompts") == selected_prompts,
        "access prompt drift",
    )
    require(
        access_protocol.get("tie_policy")
        == {
            "labels": list(LABELS),
            "rule": "first_label_in_frozen_order",
            "exact_tolerance": 1e-12,
        },
        "access tie-policy drift",
    )
    require(access_protocol.get("no_score_based_retry") is True, "access retry drift")
    require(
        authorization.get("schema") == "sallm.news_official_test_authorization/v1",
        "unexpected authorization marker",
    )
    require(
        authorization.get("status") == "AUTHORIZED_AFTER_VALIDATION_REVIEW",
        "held-out access was not authorized",
    )
    require(
        authorization.get("access_protocol_sha256")
        == sha256_file(access_protocol_path),
        "authorization binding drift",
    )
    require(
        access_started.get("schema") == "sallm.news_official_test_access_started/v1",
        "unexpected test access marker",
    )
    require(
        access_started.get("protocol_sha256") == sha256_file(access_protocol_path),
        "test access marker binding drift",
    )
    require(
        access_started.get("authorization_sha256") == sha256_file(authorization_path),
        "test access authorization binding drift",
    )
    require(
        access_started.get("no_score_based_retry") is True, "access marker retry drift"
    )

    dataset_manifest_path = _validated_ref(
        protocol["dataset_manifest"], name="materialized test manifest"
    )
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    require(
        dataset_manifest.get("schema") == "sallm.news_official_test_dataset/v1",
        "unexpected test dataset manifest",
    )
    require(dataset_manifest.get("split") == "test", "dataset split drift")
    require(
        dataset_manifest.get("metrics_computed") is False,
        "test metrics already computed",
    )
    require(
        set(dataset_manifest.get("files", {})) == set(LANGUAGE_ROWS),
        "test language drift",
    )
    require(
        dataset_manifest.get("access_started_sha256")
        == sha256_file(access_started_path),
        "dataset manifest access marker drift",
    )
    for language, expected_rows in LANGUAGE_ROWS.items():
        spec = dataset_manifest["files"][language]
        require(spec.get("rows") == expected_rows, f"{language} row-count drift")
        require(
            spec.get("columns") == ["category", "headline", "text", "url"],
            f"{language} schema drift",
        )
        require(
            set(spec.get("observed_labels", ())).issubset(LABELS),
            f"{language} label drift",
        )
    return protocol, dataset_manifest, selection


def parse_test_tsv(
    *, payload: bytes, spec: dict[str, Any], language: str
) -> list[dict[str, str]]:
    require(
        hashlib.sha256(payload).hexdigest() == spec["sha256"],
        f"{language} test hash mismatch",
    )
    require(len(payload) == spec["bytes"], f"{language} test byte-count mismatch")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), delimiter="\t")
    require(reader.fieldnames == spec["columns"], f"{language} test schema mismatch")
    rows = [dict(row) for row in reader]
    require(len(rows) == spec["rows"], f"{language} test row-count mismatch")
    require(
        all(
            all(row.get(column) is not None for column in spec["columns"])
            for row in rows
        ),
        f"{language} test contains a ragged row",
    )
    observed = sorted({row["category"].strip() for row in rows})
    require(observed == spec["observed_labels"], f"{language} label-set mismatch")
    require(set(observed).issubset(LABELS), f"{language} contains an unknown label")
    return rows


def summarize_predictions(*, rows: list[dict[str, Any]], common: Any) -> dict[str, Any]:
    require(bool(rows), "cannot summarize zero predictions")
    gold = [row["gold"] for row in rows]
    predicted = [row["prediction"] for row in rows]
    metrics = common.manual_classification_metrics(gold, predicted)
    common.validate_metrics_against_sklearn(gold, predicted, metrics)
    gold_counts = Counter(gold)
    prediction_counts = Counter(predicted)
    return {
        "n": len(rows),
        **metrics,
        "gold_class_frequency": {label: gold_counts[label] for label in LABELS},
        "prediction_class_frequency": {
            label: prediction_counts[label] for label in LABELS
        },
    }


def validate_prediction_integrity(
    *,
    predictions: list[dict[str, Any]],
    selected_prompts: dict[str, str],
) -> dict[str, Any]:
    expected_keys = {
        (language, source_index)
        for language, count in LANGUAGE_ROWS.items()
        for source_index in range(count)
    }
    observed_keys = [(row["language"], row["source_index"]) for row in predictions]
    require(len(observed_keys) == len(set(observed_keys)), "duplicate official row")
    require(set(observed_keys) == expected_keys, "official row completeness mismatch")
    exact_ties = 0
    for row in predictions:
        require(row["gold"] in LABELS, "unknown official gold label")
        require(row["prediction"] in LABELS, "unknown official prediction label")
        require(
            row["prompt_id"] == selected_prompts[row["language"]],
            "official prompt selection drift",
        )
        scores = row["first_token_log_probabilities"]
        repeated = row["first_token_repeat_log_probabilities"]
        require(set(scores) == set(LABELS), "central score label drift")
        require(set(repeated) == set(LABELS), "repeated score label drift")
        require(
            all(math.isfinite(float(value)) for value in scores.values()),
            "non-finite official central score",
        )
        require(
            all(math.isfinite(float(value)) for value in repeated.values()),
            "non-finite official repeated score",
        )
        winner = max(LABELS, key=scores.__getitem__)
        repeated_winner = max(LABELS, key=repeated.__getitem__)
        require(
            row["prediction"] == winner, "official prediction is not central argmax"
        )
        require(
            winner == repeated_winner, "official central argmax is not repeat-stable"
        )
        ranked = sorted((float(value) for value in scores.values()), reverse=True)
        exact_ties += int(abs(ranked[0] - ranked[1]) <= 1e-12)
    return {
        "expected_rows": sum(LANGUAGE_ROWS.values()),
        "observed_rows": len(predictions),
        "unique_row_keys": len(set(observed_keys)),
        "exact_top_score_ties": exact_ties,
        "completeness": "passed",
        "numerical_finiteness": "passed",
        "central_argmax_repeat_stability": "passed",
        "exact_tie_break": "first_label_in_frozen_order",
    }


def _validate_model_vocabulary(*, model: Any, tokenizer: Any) -> dict[str, int]:
    tokenizer_length = len(tokenizer)
    maximum_token_id = max(int(value) for value in tokenizer.get_vocab().values())
    input_size = int(model.get_input_embeddings().weight.shape[0])
    output_size = int(model.get_output_embeddings().weight.shape[0])
    require(input_size == tokenizer_length, "input embeddings differ from tokenizer")
    require(output_size == tokenizer_length, "output embeddings differ from tokenizer")
    require(maximum_token_id < input_size, "tokenizer ID exceeds model vocabulary")
    return {
        "tokenizer_length": tokenizer_length,
        "maximum_token_id": maximum_token_id,
        "input_embeddings": input_size,
        "output_embeddings": output_size,
    }


def run_score(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    require(args.architecture in ARCHITECTURES, "unknown architecture")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "GPU1 is required")
    protocol, dataset_manifest, _ = load_and_validate_protocol(
        protocol_path=protocol_path,
        repo=repo,
    )
    expected_output = Path(protocol["output_root"]) / args.architecture
    output_root = args.output_root.resolve()
    require(output_root == expected_output.resolve(), "official output-root drift")
    require(not output_root.exists(), "official model output already exists")

    common = _load_module(
        repo / "scripts/run_mamba_news_common_validation.py", "news_common_official"
    )
    diagnostic = _load_module(
        repo / "scripts/run_mamba_news_interface_diagnostic.py",
        "news_diagnostic_official",
    )
    diagnostic.common = common
    validation_protocol_path = Path(protocol["validation_protocol"]["path"])
    validation_protocol = common.load_and_check_protocol(repo, validation_protocol_path)
    bound: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for architecture in ARCHITECTURES:
        model_spec = protocol["models"][architecture]
        binding_path = _validated_ref(
            model_spec["binding"], name=f"{architecture} General binding"
        )
        asset_root = Path(model_spec["asset_root"]).resolve()
        require(asset_root.is_dir(), f"missing {architecture} asset root")
        selected, binding = diagnostic._select_bound_arm(
            common=common,
            protocol=validation_protocol,
            binding_path=binding_path,
            asset_root=asset_root,
            protocol_path=validation_protocol_path,
        )
        bound[architecture] = (selected, binding)

    output_root.mkdir(parents=True)
    started = {
        "schema": "sallm.news_general_official_model_score_started/v1",
        "architecture": args.architecture,
        "protocol_sha256": sha256_file(protocol_path),
        "data_boundary": "held_out_test",
        "no_score_based_retry": True,
        "started_monotonic_seconds": time.monotonic(),
    }
    _write_json(output_root / "SCORE_STARTED.json", started)

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
        }
    )
    sys.path.insert(0, str(repo / "src/main"))
    from sallm.config import ModelEvalConfig
    from sallm.evaluation.classification_metrics import ClassificationEvaluator
    from sallm.evaluation.harness import load_model_and_tokenizer

    model_spec = protocol["models"][args.architecture]
    asset_root = Path(model_spec["asset_root"]).resolve()
    model, tokenizer = load_model_and_tokenizer(
        ModelEvalConfig(
            checkpoint=str(asset_root / "base"),
            peft_adapter=str(asset_root / "runtime_adapters/general"),
            dtype=protocol["runtime"]["dtype"],
            device=protocol["runtime"]["device"],
            merge_lora=True,
            tie_word_embeddings=False,
        )
    )
    model.eval()
    vocabulary = _validate_model_vocabulary(model=model, tokenizer=tokenizer)
    evaluator = ClassificationEvaluator(tokenizer, max_samples_per_lang=None)

    test_rows = {}
    for language in LANGUAGE_ROWS:
        spec = dataset_manifest["files"][language]
        path = Path(spec["path"]).resolve()
        payload = path.read_bytes()
        test_rows[language] = parse_test_tsv(
            payload=payload,
            spec=spec,
            language=language,
        )

    templates = common._load_templates(repo, validation_protocol)
    selected_prompts = protocol["selected_prompts"]
    predictions: list[dict[str, Any]] = []
    predictions_path = output_root / "predictions.jsonl"
    started_at = time.monotonic()
    with predictions_path.open("x", encoding="utf-8") as handle:
        for language in LANGUAGE_ROWS:
            prompt_id = selected_prompts[language]
            template = templates[prompt_id]
            for source_index, row in enumerate(test_rows[language]):
                prompt = template.format(headline=row["headline"], text=row["text"])
                scored = diagnostic._score_interfaces(
                    model=model,
                    tokenizer=tokenizer,
                    evaluator=evaluator,
                    prompt=prompt,
                    labels=list(LABELS),
                )
                scored.pop("root_text")
                prediction = {
                    "architecture": args.architecture,
                    "language": language,
                    "prompt_id": prompt_id,
                    "source_index": source_index,
                    "gold": row["category"].strip(),
                    "prediction": scored["first_token_prediction"],
                    **scored,
                }
                predictions.append(prediction)
                handle.write(json.dumps(prediction, sort_keys=True) + "\n")
                if (source_index + 1) % 50 == 0:
                    handle.flush()
                    print(
                        json.dumps(
                            {
                                "architecture": args.architecture,
                                "language": language,
                                "rows": source_index + 1,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    integrity = validate_prediction_integrity(
        predictions=predictions,
        selected_prompts=selected_prompts,
    )
    language_summaries = {
        language: summarize_predictions(
            rows=[row for row in predictions if row["language"] == language],
            common=common,
        )
        for language in LANGUAGE_ROWS
    }
    overall = summarize_predictions(rows=predictions, common=common)
    sensitivity = {
        "full_label_disagreement_count": sum(
            row["full_prediction"] != row["prediction"] for row in predictions
        ),
        "batched_first_token_disagreement_count": sum(
            row["batched_first_token_prediction"] != row["prediction"]
            for row in predictions
        ),
        "maximum_singleton_repeat_score_delta": max(
            row["first_token_repeat_max_abs_score_delta"] for row in predictions
        ),
        "maximum_batched_singleton_score_delta": max(
            row["first_token_batch_max_abs_score_delta"] for row in predictions
        ),
        "exact_central_top_score_ties": integrity["exact_top_score_ties"],
    }
    require(
        sensitivity["maximum_singleton_repeat_score_delta"] == 0,
        "central singleton scores are not exactly repeatable",
    )
    binding = bound[args.architecture][1]
    summary = {
        "schema": "sallm.news_general_official_model_summary/v1",
        "status": "OFFICIAL_MODEL_VERIFIED",
        "architecture": args.architecture,
        "display_name": DISPLAY_NAMES[args.architecture],
        "split": "test",
        "regime": "General",
        "central_metric": "singleton_natural_first_label_token_log_probability",
        "reported_metric": "support_weighted_f1",
        "selected_prompts": selected_prompts,
        "languages": language_summaries,
        "overall": overall,
        "sensitivity": sensitivity,
        "integrity": integrity,
        "vocabulary": vocabulary,
        "runtime_seconds": time.monotonic() - started_at,
        "predictions_sha256": sha256_file(predictions_path),
    }
    summary_path = output_root / "summary.json"
    _write_json(summary_path, summary)
    manifest = {
        "schema": "sallm.news_general_official_model_manifest/v1",
        "status": "OFFICIAL_MODEL_VERIFIED",
        "data_boundary": "held_out_test",
        "test_accessed": True,
        "architecture": args.architecture,
        "display_name": DISPLAY_NAMES[args.architecture],
        "protocol_sha256": sha256_file(protocol_path),
        "validation_protocol_sha256": sha256_file(validation_protocol_path),
        "binding_sha256": sha256_file(Path(model_spec["binding"]["path"]).resolve()),
        "adapter_revision": binding["arm"]["adapter_revision"],
        "dataset_manifest_sha256": sha256_file(
            Path(protocol["dataset_manifest"]["path"]).resolve()
        ),
        "selected_prompts": selected_prompts,
        "no_score_based_retry": True,
        "gates": {
            "binding": "passed",
            "training_token_reconstruction_per_row": "passed",
            "unique_first_label_tokens_per_row": "passed",
            "completeness": "passed",
            "numerical_finiteness": "passed",
            "metric_recomputation": "passed",
            "central_argmax_repeat_stability": "passed",
            "exact_top_score_ties": "reported_with_frozen_label_order_tie_break",
            "full_label_agreement": "reported_sensitivity_not_validity_gate",
        },
        "predictions_sha256": sha256_file(predictions_path),
        "summary_sha256": sha256_file(summary_path),
    }
    _write_json(output_root / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "architecture": args.architecture,
                "output": str(output_root),
                "summary_sha256": manifest["summary_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_verify(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    protocol, _, _ = load_and_validate_protocol(
        protocol_path=protocol_path,
        repo=repo,
    )
    common = _load_module(
        repo / "scripts/run_mamba_news_common_validation.py", "news_common_verify"
    )
    results_root = Path(protocol["output_root"]).resolve()
    reports: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        root = results_root / architecture
        predictions_path = root / "predictions.jsonl"
        summary_path = root / "summary.json"
        manifest_path = root / "manifest.json"
        predictions = _load_jsonl(predictions_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("status") == "OFFICIAL_MODEL_VERIFIED", "model failed")
        require(manifest.get("architecture") == architecture, "architecture drift")
        require(
            manifest.get("protocol_sha256") == sha256_file(protocol_path),
            "protocol drift",
        )
        require(
            manifest.get("predictions_sha256") == sha256_file(predictions_path),
            "prediction artifact drift",
        )
        require(
            manifest.get("summary_sha256") == sha256_file(summary_path),
            "summary artifact drift",
        )
        integrity = validate_prediction_integrity(
            predictions=predictions,
            selected_prompts=protocol["selected_prompts"],
        )
        recomputed_languages = {
            language: summarize_predictions(
                rows=[row for row in predictions if row["language"] == language],
                common=common,
            )
            for language in LANGUAGE_ROWS
        }
        recomputed_overall = summarize_predictions(rows=predictions, common=common)
        require(summary["integrity"] == integrity, "integrity report drift")
        require(summary["languages"] == recomputed_languages, "language metrics drift")
        require(summary["overall"] == recomputed_overall, "overall metrics drift")
        reports[architecture] = {
            "display_name": DISPLAY_NAMES[architecture],
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "predictions_sha256": sha256_file(predictions_path),
            "summary_sha256": sha256_file(summary_path),
            "languages": recomputed_languages,
            "overall": recomputed_overall,
        }
    verified = {
        "schema": "sallm.news_general_official_replacement_verified/v1",
        "status": "OFFICIAL_CORRECTED_REPLACEMENT_VERIFIED",
        "scope": "General",
        "data_boundary": "held_out_test",
        "test_accessed": True,
        "central_metric": "singleton_natural_first_label_token_log_probability",
        "reported_metric": "support_weighted_f1",
        "selected_prompts": protocol["selected_prompts"],
        "expected_rows_per_model": sum(LANGUAGE_ROWS.values()),
        "expected_total_rows": sum(LANGUAGE_ROWS.values()) * len(ARCHITECTURES),
        "reports": reports,
        "no_score_based_retry": True,
    }
    _write_json(args.output.resolve(), verified)
    print(
        json.dumps(
            {
                "status": verified["status"],
                "output": str(args.output.resolve()),
                "sha256": sha256_file(args.output.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score")
    score.add_argument("--repo", type=Path, required=True)
    score.add_argument("--protocol", type=Path, required=True)
    score.add_argument("--architecture", required=True)
    score.add_argument("--output-root", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--repo", type=Path, required=True)
    verify.add_argument("--protocol", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "score":
        run_score(args)
    else:
        run_verify(args)


if __name__ == "__main__":
    main()
