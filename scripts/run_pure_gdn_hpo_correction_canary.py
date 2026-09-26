#!/usr/bin/env python3
"""Run the preregistered validation-only pure-GDN correction canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from sallm.config import (
    ExperimentConfig,
    FinetuneTaskType,
    ModelEvalConfig,
)
from sallm.data.factory import build_datasets
from sallm.evaluation.constrained_label_scoring import chat_messages_prefix
from sallm.evaluation.generation_metrics import GenerationEvaluator
from sallm.evaluation.harness import load_model_and_tokenizer
from sallm.evaluation.pos_metrics import PosEvaluator
from sallm.evaluation.task_metrics import (
    _normalize_ner_prediction,
    _split_ner_segments,
    _tags_to_spans,
    compute_pos_token_accuracy,
)
from sallm.training.general_validation import (
    GENERAL_SELECTION_PROTOCOL,
    GENERAL_TOTAL_PROCESSED_ROWS,
    compute_equal_family_token_nll,
    validate_general_coverage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "src" / "conf"
PREREGISTRATION_SHA256 = (
    "a27f55cd35a48bb1d22c5ca101ec536ffb89a64229559ce6883adb6fe744a04c"
)
PROMPT_CONTRACT_PREREGISTRATION_SHA256 = (
    "0f84b1f50df5d713164fc3ff62c7e9bc61c5be22505116287956f93c7ddb86d8"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compose_experiment(
    config_name: str,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=f"finetune/{config_name}",
            overrides=overrides or [],
        )
    keys = list(cfg.keys())
    unwrapped: DictConfig = cfg[keys[0]] if len(keys) == 1 else cfg
    merged = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), unwrapped)
    return cast(ExperimentConfig, merged)


def validation_dataset(
    config_name: str,
    overrides: list[str] | None = None,
) -> Dataset:
    config = compose_experiment(config_name, overrides)
    _, validation, _ = build_datasets(config, tokenizer=cast(Any, None), is_hpo=False)
    if not isinstance(validation, Dataset):
        raise TypeError(f"{config_name} validation is not a Dataset.")
    return validation


def prompt_contract(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    messages = cast(list[dict[str, str]], row["messages"])
    text, token_ids = chat_messages_prefix(tokenizer, messages[:-1])
    encoded = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return {
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_ids_sha256": hashlib.sha256(encoded).hexdigest(),
        "token_count": len(token_ids),
        "first_token_id": token_ids[0],
        "bos_token_id": tokenizer.bos_token_id,
        "first_is_bos": token_ids[0] == tokenizer.bos_token_id,
        "bos_count": token_ids.count(tokenizer.bos_token_id),
        "terminal_token_id": token_ids[-1],
        "eos_token_id": tokenizer.eos_token_id,
        "terminal_is_eos": token_ids[-1] == tokenizer.eos_token_id,
    }


def audit_ner_references(dataset: Dataset) -> dict[str, Any]:
    fixtures = [
        "PER: David A. Gross",
        "LOC: Kazan, Russia",
        "ORG: The Bomb Shelter Film Company",
        "ORG: Stimela",
    ]
    fixture_spans = {fixture: _tags_to_spans(fixture) for fixture in fixtures}
    if any(len(spans) != 1 for spans in fixture_spans.values()):
        raise ValueError(f"NER parser fixture failed: {fixture_spans}")

    prompt_counts: dict[str, int] = {}
    total_spans = 0
    nonempty_references = 0
    for index, row in enumerate(dataset):
        messages = cast(list[dict[str, str]], row["messages"])
        reference = messages[-1]["content"]
        segments = _split_ner_segments(reference)
        spans = _tags_to_spans(reference)
        if len(segments) != len(spans):
            raise ValueError(
                f"NER reference {index} has {len(segments)} explicit segments "
                f"but {len(spans)} parsed spans."
            )
        if spans != _tags_to_spans(_normalize_ner_prediction(reference)):
            raise ValueError(f"NER reference {index} is not parser-roundtrip stable.")
        prompt = str(row["template_id"])
        prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
        total_spans += len(spans)
        nonempty_references += int(bool(spans))
    if len(dataset) != 10_760 or set(prompt_counts.values()) != {2_152}:
        raise ValueError(f"NER validation coverage mismatch: {prompt_counts}")
    return {
        "rows": len(dataset),
        "prompt_counts": dict(sorted(prompt_counts.items())),
        "total_spans": total_spans,
        "nonempty_references": nonempty_references,
        "fixture_spans": fixture_spans,
    }


def pos_canary_subset(dataset: Dataset) -> Dataset:
    indices: list[int] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(dataset):
        key = (str(row["lang"]), str(row["template_id"]))
        if key in seen:
            continue
        seen.add(key)
        indices.append(index)
    if len(indices) != 12:
        raise ValueError(f"POS canary expected 12 language-prompt cells, got {seen}")
    return dataset.select(indices)


def generation_probe(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    task_type: FinetuneTaskType,
    prompt_format: str = "chat",
) -> dict[str, Any]:
    evaluator = GenerationEvaluator(
        tokenizer,
        max_new_tokens=8,
        max_samples_per_lang=1,
        sample_seed=42,
        batch_size=1,
        task_type=task_type,
        prompt_format=prompt_format,
    )
    result = evaluator.evaluate(
        model,
        dataset.select([0]),
        collect_examples=True,
        example_limit_per_lang=1,
    )
    examples = [
        example
        for language in result.per_language.values()
        for example in language.examples
    ]
    if len(examples) != 1:
        raise ValueError("Generation canary did not retain exactly one example.")
    prediction = examples[0].prediction
    return {
        "prediction": prediction,
        "prediction_sha256": hashlib.sha256(prediction.encode("utf-8")).hexdigest(),
        "prompt_format": prompt_format,
        "note": "Output quality/non-emptiness is not a canary success criterion.",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError(
            "Kombuys canary requires exactly one visible CUDA device (RTX 3080 Ti)."
        )
    visible_gpu = torch.cuda.get_device_name(torch.device(args.device))
    if visible_gpu != args.expected_gpu:
        raise ValueError(
            f"Visible GPU is {visible_gpu!r}; expected {args.expected_gpu!r}."
        )

    ner = validation_dataset(
        "llama_ner_all",
        ["++finetune.dataset.eval_template_choice=ALL"],
    )
    pos = validation_dataset("gdn_pure_pos_all_hpo_r2")
    t2x = validation_dataset("llama_t2x_xho")
    afrihg = validation_dataset("gdn_afrihg_all_hpo_r1")
    general = validation_dataset("llama_sa_general_tokenbalanced_r1")

    model_config = ModelEvalConfig(
        checkpoint=str(args.checkpoint),
        peft_adapter=None,
        dtype="bfloat16",
        device=args.device,
        merge_lora=False,
    )
    model, tokenizer = load_model_and_tokenizer(model_config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    architecture = type(model).__name__
    attention = getattr(model.config, "attn", None)
    if architecture != "GatedDeltaNetForCausalLM":
        raise ValueError(f"Unexpected model architecture {architecture}.")
    if attention is not None or parameter_count != 127_425_448:
        raise ValueError(
            f"Canonical identity mismatch: attn={attention}, params={parameter_count}"
        )
    parameter_dtypes = sorted(
        {str(parameter.dtype) for parameter in model.parameters()}
    )
    if "torch.bfloat16" not in parameter_dtypes:
        raise ValueError(f"Model did not load in BF16: {parameter_dtypes}")

    prompts = {
        "ner": prompt_contract(tokenizer, dict(ner[0])),
        "pos": prompt_contract(tokenizer, dict(pos[0])),
        "t2x": prompt_contract(tokenizer, dict(t2x[0])),
        "afrihg": prompt_contract(tokenizer, dict(afrihg[0])),
    }
    if any(
        details["terminal_is_eos"]
        or not details["first_is_bos"]
        or details["bos_count"] != 1
        for details in prompts.values()
    ):
        raise ValueError(f"At least one chat prompt violates BOS/EOS: {prompts}")

    raw_evaluator = GenerationEvaluator.__new__(GenerationEvaluator)
    raw_evaluator.tokenizer = tokenizer
    raw_evaluator.prompt_format = "raw"
    raw_messages = cast(list[dict[str, str]], t2x[0]["messages"])
    raw_text, raw_ids = raw_evaluator._render_generation_prompt(
        raw_messages[:-1],
        None,
        None,
    )
    raw_prompt = {
        "text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "token_ids_sha256": hashlib.sha256(
            json.dumps(raw_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "first_is_bos": raw_ids[0] == tokenizer.bos_token_id,
        "bos_count": raw_ids.count(tokenizer.bos_token_id),
        "terminal_is_eos": raw_ids[-1] == tokenizer.eos_token_id,
        "contains_chat_marker": any(
            marker in raw_text
            for marker in ("<|system|>", "<|user|>", "<|assistant|>")
        ),
    }
    if (
        not raw_prompt["first_is_bos"]
        or raw_prompt["bos_count"] != 1
        or raw_prompt["terminal_is_eos"]
        or raw_prompt["contains_chat_marker"]
    ):
        raise ValueError(f"Raw base prompt contract failed: {raw_prompt}")

    ner_audit = audit_ner_references(ner)
    pos_evaluator = PosEvaluator(
        tokenizer,
        score_mode="mean",
        pad_to_multiple_of=64,
        strict_contract=True,
    )
    pos_metrics = pos_evaluator.evaluate(model, pos_canary_subset(pos))
    if not math.isfinite(pos_metrics["eval/all_token_accuracy"]):
        raise ValueError("POS canary returned non-finite token accuracy.")
    prefix_extra_score = compute_pos_token_accuracy(
        ["NOUN VERB"],
        ["NOUN VERB X"],
    )
    if prefix_extra_score >= 1.0:
        raise ValueError("POS scorer gave prefix-plus-extra output full credit.")

    general_coverage = validate_general_coverage(general)
    toy_sums = {
        task: float(index + 1)
        for index, task in enumerate(("sib", "news", "ner", "pos", "afrihg", "t2x"))
    }
    toy_counts = {
        task: index + 1
        for index, task in enumerate(("sib", "news", "ner", "pos", "afrihg", "t2x"))
    }
    toy_macro, toy_family = compute_equal_family_token_nll(toy_sums, toy_counts)
    if toy_macro != 1.0 or set(toy_family.values()) != {1.0}:
        raise ValueError("Toy General token-NLL calculation failed.")

    preregistration = REPO_ROOT / (
        "sallm_memory/notes/2026-08-09-pure-gdn-hpo-correction-preregistration.md"
    )
    preregistration_sha = sha256_file(preregistration)
    if preregistration_sha != PREREGISTRATION_SHA256:
        raise ValueError(
            "Correction preregistration hash drifted before canary execution: "
            f"{preregistration_sha}"
        )
    prompt_preregistration = REPO_ROOT / (
        "sallm_memory/notes/"
        "2026-08-09-pure-gdn-prompt-contract-correction-preregistration.md"
    )
    prompt_preregistration_sha = sha256_file(prompt_preregistration)
    if prompt_preregistration_sha != PROMPT_CONTRACT_PREREGISTRATION_SHA256:
        raise ValueError(
            "Prompt-contract preregistration hash drifted before canary execution: "
            f"{prompt_preregistration_sha}"
        )
    return {
        "schema": "pure_gdn_hpo_correction_canary/v1",
        "data_boundary": "validation-only; no held-out split loaded or scored",
        "gpu_visibility": {
            "expected": args.expected_gpu,
            "actual": visible_gpu,
            "visible_device_count": torch.cuda.device_count(),
        },
        "model": {
            "checkpoint": str(args.checkpoint),
            "architecture": architecture,
            "attn": attention,
            "parameter_count": parameter_count,
            "parameter_dtypes": parameter_dtypes,
            "peft_adapter": None,
            "merge_lora": False,
        },
        "prompt_contracts": prompts,
        "raw_base_prompt_contract": raw_prompt,
        "ner_reference_audit": ner_audit,
        "pos_canary": {
            "rows": 12,
            "metrics": pos_metrics,
            "details": pos_evaluator.last_details,
            "prefix_plus_extra_score": prefix_extra_score,
        },
        "general_coverage": general_coverage,
        "general_objective": {
            "protocol": GENERAL_SELECTION_PROTOCOL,
            "expected_rows": GENERAL_TOTAL_PROCESSED_ROWS,
            "toy_family_nll": toy_family,
            "toy_macro_nll": toy_macro,
        },
        "generation_path_probes": {
            "ner": generation_probe(
                model,
                tokenizer,
                ner,
                FinetuneTaskType.NAMED_ENTITY_RECOGNITION,
            ),
            "t2x": generation_probe(
                model,
                tokenizer,
                t2x,
                FinetuneTaskType.INSTRUCTION,
                prompt_format="raw",
            ),
            "afrihg": generation_probe(
                model,
                tokenizer,
                afrihg,
                FinetuneTaskType.INSTRUCTION,
                prompt_format="raw",
            ),
        },
        "preregistration": {
            "path": str(preregistration),
            "sha256": preregistration_sha,
        },
        "prompt_contract_preregistration": {
            "path": str(prompt_preregistration),
            "sha256": prompt_preregistration_sha,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = run(args)
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
    print(f"PASS: wrote {args.output} ({digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
