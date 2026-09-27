#!/usr/bin/env python3
"""Fail-closed corrected SIB General official replacement evaluator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import yaml
from datasets import Dataset

LABELS = (
    "science/technology", "travel", "politics", "sports", "health",
    "entertainment", "geography",
)
ARCHITECTURES = ("mzansilm", "mamba2", "xlstm", "gdn")
EXPECTED_PROMPTS = {"afr": 4, "eng": 3, "nso": 2, "sot": 4, "xho": 5, "zul": 5}
EXPECTED_PROTOCOL_SCHEMA = "sallm.sib_general_corrected_official_replacement/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_once(path: Path, payload: dict[str, Any], *, readonly: bool = False) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    if readonly:
        path.chmod(0o444)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    require(protocol["schema"] == EXPECTED_PROTOCOL_SCHEMA, "Wrong protocol schema")
    require(protocol["validation"]["selected_prompts"] == EXPECTED_PROMPTS, "Prompt freeze differs")
    return protocol


def prompt_path(protocol: dict[str, Any], language: str) -> Path:
    runtime = Path(protocol["runtime"]["snapshot"])
    prompt = protocol["test"]["languages"][language]["prompt"]
    return runtime / "src/conf/eval/lm_eval_tasks/sib_validation" / f"sallm_sib_{language}_val_prompt_{prompt}.yaml"


def split_path(protocol: dict[str, Any], language: str, split: str) -> Path:
    entry = protocol["test"]["languages"][language]
    return (
        Path(protocol["test"]["cache_root"]) / entry["subset"] / "0.0.0"
        / protocol["test"]["revision"] / f"sib200-{split}.arrow"
    )


def validate_artifact_hashes(protocol: dict[str, Any]) -> None:
    for key in ("full_manifest", "selection", "token_audit", "generation_manifest", "bindings", "source_results"):
        path = Path(protocol["validation"][key])
        require(path.is_file(), f"Missing validation artifact {path}")
        require(sha256(path) == protocol["validation"][f"{key}_sha256"], f"Hash mismatch: {path}")
    for key in ("validation_scorer", "validation_core"):
        path = Path(protocol["runtime"][key])
        require(path.is_file(), f"Missing runtime artifact {path}")
        require(sha256(path) == protocol["runtime"][f"{key}_sha256"], f"Hash mismatch: {path}")
    manifest = Path(protocol["runtime"]["snapshot"]) / "SNAPSHOT_MANIFEST.sha256"
    require(sha256(manifest) == protocol["runtime"]["snapshot_manifest_sha256"], "Runtime snapshot manifest changed")


def run_preflight(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    require(not Path(protocol["output_root"]).exists(), "Corrected official output already exists")
    validate_artifact_hashes(protocol)
    validation = protocol["validation"]
    selection = json.loads(Path(validation["selection"]).read_text(encoding="utf-8"))
    require(selection["split"] == "validation", "Prompts were not selected on validation")
    require(selection["heldout_action"] == "none; validation only", "Selection touched held-out data")
    require({k: v["prompt"] for k, v in selection["selected"].items()} == EXPECTED_PROMPTS, "Selected prompts differ")
    require(selection["bindings_sha256"] == validation["bindings_sha256"], "Selection binding differs")

    full_root = Path(validation["selection"]).parent / "full"
    generation_root = Path(validation["generation_manifest"]).parent
    model_checks = {}
    for architecture in ARCHITECTURES:
        full = json.loads((full_root / f"{architecture}.json").read_text(encoding="utf-8"))
        require(full["split"] == "validation", f"{architecture} full report is not validation")
        require(full["heldout_action"] == "none; validation only", f"{architecture} validation touched held-out")
        require(full["completeness_gate"]["pass"], f"{architecture} validation incomplete")
        require(full["completeness_gate"]["observed_rows"] == 2970, f"{architecture} row count differs")
        require(full["natural_first_token"]["all"]["top_score_ties"] == 0, f"{architecture} validation ties")
        generation = json.loads((generation_root / f"{architecture}.json").read_text(encoding="utf-8"))
        require(generation["split"] == "validation", f"{architecture} generation report is not validation")
        require(generation["completeness_gate"]["pass"], f"{architecture} generation cross-check incomplete")
        require(generation["summary"]["agreement_conditional_on_valid"] == 1.0, f"{architecture} generation disagreement")
        model = protocol["models"][architecture]
        checkpoint, adapter = Path(model["checkpoint"]), Path(model["adapter"])
        require(sha256(checkpoint / model["base_file"]) == model["base_sha256"], f"{architecture} base changed")
        require(sha256(checkpoint / "config.json") == model["config_sha256"], f"{architecture} config changed")
        require(sha256(adapter / "adapter_model.safetensors") == model["adapter_sha256"], f"{architecture} adapter changed")
        require(full["run"]["checkpoint"] == str(checkpoint), f"{architecture} validation checkpoint differs")
        require(full["run"]["adapter"] == str(adapter), f"{architecture} validation adapter differs")
        require(full["run"]["dtype"] == "float32", f"{architecture} validation dtype differs")
        require(full["run"]["merge_lora"] == model["merge_lora"], f"{architecture} merge flag differs")
        require(full["run"]["tie_word_embeddings"] == model["tie_word_embeddings"], f"{architecture} tie flag differs")
        model_checks[architecture] = {"validation_rows": 2970, "ties": 0, "generation_agreement_on_valid": 1.0}

    scorer = load_module(Path(protocol["runtime"]["validation_scorer"]), "sealed_sib_validation")
    scorer.self_check()
    core = scorer.load_core(Path(protocol["runtime"]["validation_core"]))
    source = core.load_validation_source(Path(validation["source_results"]))
    prompt_checks = {}
    test_metadata = {}
    for language, prompt in EXPECTED_PROMPTS.items():
        entry = protocol["test"]["languages"][language]
        yaml_path = prompt_path(protocol, language)
        require(sha256(yaml_path) == entry["prompt_yaml_sha256"], f"{language} prompt YAML changed")
        template = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["doc_to_text"]
        task = f"sallm_sib_{language}_val_prompt_{prompt}"
        validation_ds = Dataset.from_file(str(split_path(protocol, language, "validation")))
        rows = source["samples"][task]
        require(len(rows) == 99 == len(validation_ds), f"{language} validation prompt rows differ")
        for row in rows:
            text = validation_ds[int(row["doc_id"])]["text"]
            reconstructed = template.replace("{{text}}", text)
            require(reconstructed == str(row["arguments"][0][0]), f"{language} prompt reconstruction differs")
        prompt_checks[language] = {"prompt": prompt, "validation_rows_exact": 99, "prompt_yaml_sha256": sha256(yaml_path)}

        directory = split_path(protocol, language, "test").parent
        arrow = directory / "sib200-test.arrow"
        metadata = directory / "dataset_info.json"
        require(sha256(arrow) == entry["arrow_sha256"], f"{language} cached test bytes changed")
        require(sha256(metadata) == entry["metadata_sha256"], f"{language} cached metadata changed")
        info = json.loads(metadata.read_text(encoding="utf-8"))
        require(info["splits"]["test"]["num_examples"] == protocol["test"]["rows_per_language"], f"{language} metadata count differs")
        with pa.memory_map(str(arrow), "r") as source_file:
            schema = ipc.open_stream(source_file).schema
        require(schema.names == ["index_id", "category", "text"], f"{language} test schema differs")
        test_metadata[language] = {"arrow_sha256": sha256(arrow), "expected_rows": 204, "schema": schema.names}

    payload = {
        "status": "READY_FOR_SINGLE_PROSPECTIVE_CORRECTED_REPLACEMENT",
        "heldout_rows_opened": False,
        "protocol": str(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "scorer_self_check": "PASS",
        "prompt_selection_source": "corrected validation only",
        "selected_prompts": EXPECTED_PROMPTS,
        "prompt_reconstruction": prompt_checks,
        "models": model_checks,
        "test_cache_metadata_only": test_metadata,
        "expected_rows_per_model": 1224,
        "expected_total_rows": 4896,
        "prior_raw_label_access": protocol["protocol_amendment"]["supersedes"],
        "prior_raw_label_status": protocol["protocol_amendment"]["superseded_status"],
        "retry_policy": protocol["protocol_amendment"]["retry_policy"],
    }
    write_once(args.output, payload, readonly=True)
    print(json.dumps({"status": payload["status"], "output": str(args.output), "sha256": sha256(args.output)}, sort_keys=True))


def run_authorize(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    validate_artifact_hashes(protocol)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    require(preflight["status"] == "READY_FOR_SINGLE_PROSPECTIVE_CORRECTED_REPLACEMENT", "Preflight did not pass")
    require(preflight["heldout_rows_opened"] is False, "Preflight opened held-out rows")
    require(preflight["protocol_sha256"] == sha256(args.protocol), "Preflight protocol changed")
    payload = {
        "status": "CORRECTED_REPLACEMENT_AUTHORIZED",
        "wording": protocol["protocol_amendment"]["access_wording"],
        "superseded_quarantined_access": protocol["protocol_amendment"]["supersedes"],
        "selection_source": "corrected validation only",
        "selected_prompts": EXPECTED_PROMPTS,
        "protocol_sha256": sha256(args.protocol),
        "preflight_sha256": sha256(args.preflight),
        "authorized_at": datetime.now(timezone.utc).isoformat(),
        "retry_policy": protocol["protocol_amendment"]["retry_policy"],
    }
    write_once(args.output, payload, readonly=True)
    print(json.dumps({"status": payload["status"], "output": str(args.output), "sha256": sha256(args.output)}, sort_keys=True))


def load_and_validate_marker(protocol: dict[str, Any], protocol_path: Path, preflight: Path, marker: Path) -> dict[str, Any]:
    pre = json.loads(preflight.read_text(encoding="utf-8"))
    mark = json.loads(marker.read_text(encoding="utf-8"))
    require(mark["status"] == "CORRECTED_REPLACEMENT_AUTHORIZED", "Access marker invalid")
    require(mark["protocol_sha256"] == sha256(protocol_path) == pre["protocol_sha256"], "Protocol binding changed")
    require(mark["preflight_sha256"] == sha256(preflight), "Preflight binding changed")
    require(mark["selected_prompts"] == EXPECTED_PROMPTS, "Marker prompts differ")
    return mark


def run_score(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    require(args.architecture in ARCHITECTURES, "Unknown architecture")
    validate_artifact_hashes(protocol)
    marker = load_and_validate_marker(protocol, args.protocol, args.preflight, args.marker)
    model_entry = protocol["models"][args.architecture]
    validation_root = Path(protocol["validation"]["selection"]).parent
    audit = json.loads((validation_root / "token_audit.json").read_text(encoding="utf-8"))["architectures"][args.architecture]
    scorer = load_module(Path(protocol["runtime"]["validation_scorer"]), "sealed_sib_validation")
    core = scorer.load_core(Path(protocol["runtime"]["validation_core"]))
    started = time.monotonic()
    model, tokenizer = scorer.load_model_and_tokenizer(
        scorer.ModelEvalConfig(
            checkpoint=model_entry["checkpoint"], dtype="float32", device="cuda:0",
            peft_adapter=model_entry["adapter"], merge_lora=model_entry["merge_lora"],
            tie_word_embeddings=model_entry["tie_word_embeddings"],
        )
    )
    require(scorer.tokenizer_identity_sha256(tokenizer) == audit["in_memory_tokenizer_identity_sha256"], "Tokenizer identity changed")
    require(hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest() == audit["serialization"]["chat_template_sha256"], "Chat template changed")
    model.eval()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    require(pad_token_id is not None, "Tokenizer has no PAD/EOS")
    pad_multiple = scorer.ClassificationEvaluator._get_model_chunk_size(model)

    first_access = args.marker.parent / "FIRST_CORRECTED_TEST_ACCESS.json"
    if not first_access.exists():
        write_once(first_access, {
            "wording": protocol["protocol_amendment"]["access_wording"],
            "first_corrected_test_access_at": datetime.now(timezone.utc).isoformat(),
            "first_architecture": args.architecture,
            "protocol_sha256": sha256(args.protocol),
            "marker_sha256": sha256(args.marker),
            "superseded_quarantined_access": protocol["protocol_amendment"]["supersedes"],
        }, readonly=True)

    rows = []
    language_summaries = {}
    with torch.inference_mode():
        for language, prompt_number in EXPECTED_PROMPTS.items():
            template = yaml.safe_load(prompt_path(protocol, language).read_text(encoding="utf-8"))["doc_to_text"]
            dataset = Dataset.from_file(str(split_path(protocol, language, "test")))
            require(len(dataset) == protocol["test"]["rows_per_language"], f"{language} test rows differ")
            language_rows = []
            for index, item in enumerate(dataset):
                gold = str(item["category"])
                require(gold in LABELS, f"Unexpected gold label {gold}")
                prompt = template.replace("{{text}}", str(item["text"]))
                scores, first_ids, hand = scorer.score_prompt(
                    model=model, tokenizer=tokenizer, prompt=prompt,
                    pad_token_id=int(pad_token_id), pad_to_multiple_of=pad_multiple,
                    device=model.device, core=core,
                )
                guess, margin, tie = scorer.prediction(scores)
                row = {
                    "language": language, "prompt": prompt_number,
                    "doc_id": int(item["index_id"]), "gold": gold,
                    "prediction": guess, "scores": scores,
                    "first_token_ids": first_ids, "winning_margin": margin, "tie": tie,
                    "context_tokens": hand["training_label_prefix_tokens"],
                }
                rows.append(row); language_rows.append(row)
                if (index + 1) % 25 == 0:
                    print(json.dumps({"architecture": args.architecture, "language": language, "rows": index + 1}), flush=True)
            language_summaries[language] = scorer.summarize(language_rows)
    summary = scorer.summarize(rows)
    ties = sum(bool(row["tie"]) for row in rows)
    completeness = len(rows) == 1224 and all(v["n"] == 204 for v in language_summaries.values())
    require(completeness, "Official test output incomplete")
    require(ties == 0, f"Official test contains {ties} numerical top-score ties")
    payload = {
        "schema": "sallm.sib_natural_first_token_general_official_replacement_result/v1",
        "split": "test", "regime": "General", "architecture": args.architecture,
        "protocol_sha256": sha256(args.protocol), "preflight_sha256": sha256(args.preflight),
        "access_marker_sha256": sha256(args.marker), "access_marker": marker,
        "selected_prompts": EXPECTED_PROMPTS, "selection_source": "corrected validation only",
        "run": {"checkpoint": model_entry["checkpoint"], "adapter": model_entry["adapter"], "dtype": "float32", "device": "cuda:0", "merge_lora": model_entry["merge_lora"], "tie_word_embeddings": model_entry["tie_word_embeddings"]},
        "completeness_gate": {"pass": completeness, "expected_rows": 1224, "observed_rows": len(rows)},
        "numerical_gate": {"pass": ties == 0, "top_score_ties": ties, "finite_scores": all(all(math.isfinite(float(x)) for x in row["scores"].values()) for row in rows)},
        "metrics": summary["metrics"], "summary": summary,
        "languages": language_summaries, "rows": rows,
        "runtime_seconds": time.monotonic() - started,
        "retry_policy": protocol["protocol_amendment"]["retry_policy"],
    }
    require(payload["numerical_gate"]["finite_scores"], "Non-finite score found")
    write_once(args.output, payload, readonly=True)
    print(json.dumps({"architecture": args.architecture, "output": str(args.output), "sha256": sha256(args.output), "f1": summary["metrics"]["f1"]}, sort_keys=True))


def manual_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows); correct = sum(row["gold"] == row["prediction"] for row in rows)
    f1s, weighted = [], 0.0
    for label in LABELS:
        tp = sum(r["gold"] == label and r["prediction"] == label for r in rows)
        fp = sum(r["gold"] != label and r["prediction"] == label for r in rows)
        fn = sum(r["gold"] == label and r["prediction"] != label for r in rows)
        support = tp + fn
        f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
        f1s.append(f1); weighted += support * f1
    return {"accuracy": correct / n, "f1": weighted / n, "macro_f1": sum(f1s) / len(f1s)}


def run_verify(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    load_and_validate_marker(protocol, args.protocol, args.preflight, args.marker)
    reports = {}; all_rows = []
    for architecture in ARCHITECTURES:
        path = args.results / f"{architecture}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        require(result["split"] == "test" and result["regime"] == "General", f"{architecture} scope differs")
        require(result["completeness_gate"]["pass"] and result["completeness_gate"]["observed_rows"] == 1224, f"{architecture} incomplete")
        require(result["numerical_gate"]["pass"] and result["numerical_gate"]["top_score_ties"] == 0, f"{architecture} numerical failure")
        require(len({(r["language"], r["doc_id"]) for r in result["rows"]}) == 1224, f"{architecture} duplicate rows")
        require(Counter(r["language"] for r in result["rows"]) == Counter({x: 204 for x in EXPECTED_PROMPTS}), f"{architecture} language counts differ")
        for row in result["rows"]:
            require(row["prediction"] == max(LABELS, key=lambda label: row["scores"][label]), f"{architecture} argmax differs")
            require(row["prompt"] == EXPECTED_PROMPTS[row["language"]], f"{architecture} prompt differs")
        manual = manual_summary(result["rows"])
        for metric, value in manual.items():
            require(abs(value - float(result["metrics"][metric])) <= 1e-12, f"{architecture} {metric} recomputation differs")
        reports[architecture] = {"path": str(path), "sha256": sha256(path), "metrics": manual}
        all_rows.extend((architecture, row["language"], row["doc_id"]) for row in result["rows"])
    require(len(all_rows) == 4896, "Total row count differs")
    payload = {"status": "OFFICIAL_CORRECTED_REPLACEMENT_VERIFIED", "scope": "General only", "reports": reports, "total_rows": 4896, "selected_prompts": EXPECTED_PROMPTS, "no_score_based_retry": True, "first_access": json.loads((args.marker.parent / "FIRST_CORRECTED_TEST_ACCESS.json").read_text(encoding="utf-8"))}
    write_once(args.output, payload, readonly=True)
    print(json.dumps({"status": payload["status"], "output": str(args.output), "sha256": sha256(args.output)}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(); commands = root.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight"); preflight.add_argument("--protocol", type=Path, required=True); preflight.add_argument("--output", type=Path, required=True)
    authorize = commands.add_parser("authorize"); authorize.add_argument("--protocol", type=Path, required=True); authorize.add_argument("--preflight", type=Path, required=True); authorize.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score"); score.add_argument("--protocol", type=Path, required=True); score.add_argument("--preflight", type=Path, required=True); score.add_argument("--marker", type=Path, required=True); score.add_argument("--architecture", required=True); score.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify"); verify.add_argument("--protocol", type=Path, required=True); verify.add_argument("--preflight", type=Path, required=True); verify.add_argument("--marker", type=Path, required=True); verify.add_argument("--results", type=Path, required=True); verify.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    {"preflight": run_preflight, "authorize": run_authorize, "score": run_score, "verify": run_verify}[args.command](args)


if __name__ == "__main__":
    main()
