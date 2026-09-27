#!/usr/bin/env python3
"""Audit and compare training-matched Mamba News interfaces on validation only."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _load_common(repo: Path) -> Any:
    path = repo / "scripts/run_mamba_news_common_validation.py"
    spec = importlib.util.spec_from_file_location("mamba_news_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load common runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _common_prefix_length(sequences: Sequence[Sequence[int]]) -> int:
    common = 0
    for items in zip(*sequences, strict=False):
        if len(set(items)) != 1:
            break
        common += 1
    return common


def _left_truncate_for_generation(
    token_ids: Sequence[int], *, context_limit: int, maximum_new_tokens: int
) -> tuple[list[int], int]:
    maximum_prompt_tokens = context_limit - maximum_new_tokens
    if maximum_prompt_tokens <= 0:
        raise ValueError("generation budget must be smaller than the context limit")
    removed = max(0, len(token_ids) - maximum_prompt_tokens)
    return list(token_ids[removed:]), removed


def _align_base_anchor_vocabulary(*, model: Any, tokenizer: Any) -> dict[str, int]:
    """Match only the diagnostic base anchor to the selected adapter tokenizer."""
    target = len(tokenizer)
    before_input = int(model.get_input_embeddings().weight.shape[0])
    before_output = int(model.get_output_embeddings().weight.shape[0])
    if before_input != target or before_output != target:
        model.resize_token_embeddings(target)
    after_input = int(model.get_input_embeddings().weight.shape[0])
    after_output = int(model.get_output_embeddings().weight.shape[0])
    if after_input != target or after_output != target:
        raise RuntimeError("base-anchor embeddings do not match runtime tokenizer")
    maximum_token_id = max(int(value) for value in tokenizer.get_vocab().values())
    if maximum_token_id >= after_input:
        raise RuntimeError("runtime tokenizer contains an out-of-range token ID")
    return {
        "tokenizer_length": target,
        "maximum_token_id": maximum_token_id,
        "input_embeddings_before": before_input,
        "output_embeddings_before": before_output,
        "input_embeddings_after": after_input,
        "output_embeddings_after": after_output,
    }


def _read_validation_rows(common: Any, protocol: dict[str, Any], asset_root: Path):
    rows = {}
    specs = {item["language"]: item for item in protocol["dataset"]["allowed_files"]}
    for language, spec in specs.items():
        payload = (asset_root / "dataset" / f"{language}-dev.tsv").read_bytes()
        rows[language] = common.parse_validation_tsv(payload, spec)
    return rows


def _exact_candidates(
    *, tokenizer: Any, evaluator: Any, prompt: str, labels: list[str]
) -> dict[str, Any]:
    messages = [{"role": "user", "content": prompt}]
    rendered = evaluator._build_prompt_text(
        prompt_messages=messages,
        fallback_template=None,
        system_message=None,
    )
    common.require(
        rendered.endswith(common.GENERATION_PROMPT_SUFFIX),
        "diagnostic prompt suffix drift",
    )
    root_text = rendered.rstrip()
    root_ids = tokenizer.encode(root_text, add_special_tokens=False)
    continuations: list[list[int]] = []
    for label in labels:
        context_ids, continuation_ids = evaluator._encode_choice_pair(rendered, label)
        exact_ids = tokenizer.encode(rendered + label, add_special_tokens=False)
        training_text = tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": label}],
            tokenize=False,
            add_generation_prompt=False,
        )
        common.require(
            training_text.startswith(rendered + label),
            "training/evaluation rendered text mismatch",
        )
        common.require(
            context_ids == root_ids,
            "choice context differs from whitespace-trimmed training prefix",
        )
        common.require(
            context_ids + continuation_ids == exact_ids,
            "choice token IDs do not reconstruct training serialization",
        )
        continuations.append(continuation_ids)

    shared_length = _common_prefix_length(continuations)
    common.require(
        shared_length < min(len(items) for items in continuations),
        "candidate continuations never diverge",
    )
    first_label_ids = [items[shared_length] for items in continuations]
    common.require(
        len(set(first_label_ids)) == len(labels),
        "first natural-label token IDs are not unique",
    )
    return {
        "rendered": rendered,
        "root_text": root_text,
        "root_ids": root_ids,
        "continuations": continuations,
        "continuation_lengths": dict(
            zip(labels, (len(items) for items in continuations), strict=True)
        ),
        "shared_prefix_length": shared_length,
        "shared_prefix_ids": continuations[0][:shared_length],
        "first_label_token_ids": dict(zip(labels, first_label_ids, strict=True)),
    }


def _audit_all_contexts(
    *,
    common: Any,
    protocol: dict[str, Any],
    repo: Path,
    asset_root: Path,
    validation_rows: dict[str, list[dict[str, str]]],
    output_path: Path,
) -> dict[str, Any]:
    from sallm.evaluation.classification_metrics import ClassificationEvaluator
    from transformers import AutoTokenizer

    templates = common._load_templates(repo, protocol)
    labels = list(protocol["labels"])
    record_count = 0
    length_patterns: Counter[str] = Counter()
    shared_patterns: Counter[str] = Counter()
    tokenizer_hashes: dict[str, str] = {}
    with output_path.open("w", encoding="utf-8") as handle:
        for arm in protocol["arms"]:
            adapter = asset_root / "runtime_adapters" / arm["id"]
            tokenizer_hashes[arm["id"]] = common.file_sha256(
                adapter / "tokenizer.json"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                adapter,
                local_files_only=True,
                trust_remote_code=True,
            )
            evaluator = ClassificationEvaluator(tokenizer, max_samples_per_lang=None)
            for language in arm["languages"]:
                for source_index, row in enumerate(validation_rows[language]):
                    for prompt_id, template in templates.items():
                        prompt = template.format(
                            headline=row["headline"], text=row["text"]
                        )
                        exact = _exact_candidates(
                            tokenizer=tokenizer,
                            evaluator=evaluator,
                            prompt=prompt,
                            labels=labels,
                        )
                        lengths = exact["continuation_lengths"]
                        length_patterns[json.dumps(lengths, sort_keys=True)] += 1
                        shared_patterns[
                            json.dumps(
                                {
                                    "shared_prefix_ids": exact["shared_prefix_ids"],
                                    "first_label_token_ids": exact[
                                        "first_label_token_ids"
                                    ],
                                },
                                sort_keys=True,
                            )
                        ] += 1
                        record = {
                            "arm": arm["id"],
                            "language": language,
                            "prompt_id": prompt_id,
                            "source_index": source_index,
                            "continuation_lengths": lengths,
                            "shared_prefix_ids": exact["shared_prefix_ids"],
                            "first_label_token_ids": exact["first_label_token_ids"],
                            "training_token_reconstruction": "passed",
                        }
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                        record_count += 1
    expected_records = sum(
        len(validation_rows[language]) * len(templates)
        for arm in protocol["arms"]
        for language in arm["languages"]
    )
    common.require(
        record_count == expected_records,
        "full token-audit completeness mismatch",
    )
    return {
        "records": record_count,
        "expected_records": expected_records,
        "training_token_reconstruction": "passed",
        "unique_first_label_tokens": "passed",
        "continuation_length_patterns": dict(length_patterns),
        "candidate_token_patterns": dict(shared_patterns),
        "tokenizer_json_sha256_by_arm": tokenizer_hashes,
        "audit_jsonl_sha256": common.file_sha256(output_path),
    }


def _select_bound_arm(
    *,
    common: Any,
    protocol: dict[str, Any],
    binding_path: Path,
    asset_root: Path,
    protocol_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    schema = binding.get("schema")
    common.require(
        schema
        in {
            "sallm.mamba_news_validation_arm_binding/v1",
            "sallm.news_general_validation_arm_binding/v1",
        },
        "unexpected validation-arm binding schema",
    )
    common.require(
        binding.get("data_boundary") == "validation_only",
        "validation-arm binding is not validation-only",
    )
    common.require(binding.get("test_access_allowed") is False, "test access allowed")
    arm = binding["arm"]
    common.require(
        arm.get("id") == "general",
        "only the sealed General validation arm is supported",
    )
    common.require(
        arm.get("languages") == ["eng", "xho"],
        "General validation language coverage drift",
    )
    adapter = asset_root / "runtime_adapters" / arm["id"]
    common.require(adapter.is_dir(), "missing bound General runtime adapter")
    if schema == "sallm.mamba_news_validation_arm_binding/v1":
        common.require(
            binding.get("base_revision") == protocol["base"]["revision"],
            "validation-arm base revision mismatch",
        )
        related = protocol["related_retained_general"]
        for key in (
            "adapter_repo",
            "adapter_revision",
            "adapter_weights_sha256",
            "adapter_config_sha256",
            "source_chat_template_sha256",
        ):
            common.require(
                arm.get(key) == related.get(key),
                f"General binding differs from the canonical retained adapter: {key}",
            )
        runtime_files = {
            "adapter_model.safetensors": arm["adapter_weights_sha256"],
            "adapter_config.json": arm["adapter_config_sha256"],
            "chat_template.jinja": protocol["runtime"]["prompt_normalization"][
                "normalized_sha256"
            ],
            "tokenizer.json": arm["tokenizer_json_sha256"],
            "tokenizer_config.json": arm["tokenizer_config_sha256"],
        }
        expected_adapter_config = {
            "target_modules": related["expected_target_modules"]
        }
    else:
        common.require(protocol_path is not None, "missing task protocol path")
        common.require(
            common.file_sha256(protocol_path) == binding["task_protocol_sha256"],
            "General binding task-protocol hash mismatch",
        )
        base = asset_root / "base"
        source_adapter = asset_root / "source_adapters" / arm["id"]
        common.require(base.is_dir(), "missing bound General base model")
        common.require(source_adapter.is_dir(), "missing General source adapter")
        for filename, expected_sha256 in binding["base"]["files"].items():
            common.require(
                common.file_sha256(base / filename) == expected_sha256,
                f"General base hash mismatch: {filename}",
            )
        for filename, expected_sha256 in arm["source_files"].items():
            common.require(
                common.file_sha256(source_adapter / filename) == expected_sha256,
                f"General source-adapter hash mismatch: {filename}",
            )
        runtime_files = arm["runtime_files"]
        expected_adapter_config = arm["expected_adapter_config"]
    for filename, expected_sha256 in runtime_files.items():
        common.require(
            common.file_sha256(adapter / filename) == expected_sha256,
            f"General runtime adapter hash mismatch: {filename}",
        )
    adapter_config = json.loads(
        (adapter / "adapter_config.json").read_text(encoding="utf-8")
    )
    for key, expected in expected_adapter_config.items():
        actual = adapter_config.get(key)
        if key == "target_modules":
            actual = sorted(actual or [])
            expected = sorted(expected)
        common.require(actual == expected, f"General adapter-config mismatch: {key}")
    selected = copy.deepcopy(protocol)
    selected["arms"] = [{"id": arm["id"], "languages": arm["languages"]}]
    return selected, binding


def _score_interfaces(
    *, model: Any, tokenizer: Any, evaluator: Any, prompt: str, labels: list[str]
) -> dict[str, Any]:
    import torch
    from sallm.evaluation.classification_metrics import _gather_target_log_probs

    exact = _exact_candidates(
        tokenizer=tokenizer,
        evaluator=evaluator,
        prompt=prompt,
        labels=labels,
    )
    pad_id = evaluator._resolve_pad_id(tokenizer.pad_token_id, tokenizer.eos_token_id)
    input_ids, attention_mask, choice_starts = evaluator._build_choice_inputs(
        prompt_text=exact["rendered"],
        label_choices=labels,
        model_ctx_limit=1024,
        pad_token_id=pad_id,
        device=model.device,
        pad_to_multiple_of=evaluator._get_model_chunk_size(model),
    )
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits[:, :-1, :].float()
    target_ids = input_ids[:, 1:]
    token_log_probs = _gather_target_log_probs(logits=logits, target_ids=target_ids)
    common.require(
        bool(torch.isfinite(token_log_probs).all().item()),
        "non-finite diagnostic token log probability",
    )
    sequence_lengths = attention_mask.sum(dim=1)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, start in enumerate(choice_starts):
        mask[index, start : int(sequence_lengths[index].item())] = True
    mask = mask[:, 1:] & attention_mask[:, 1:].bool()
    totals_tensor = token_log_probs.masked_fill(~mask, 0.0).sum(dim=1)
    counts_tensor = mask.sum(dim=1)
    totals = [float(value) for value in totals_tensor.cpu().tolist()]
    counts = [int(value) for value in counts_tensor.cpu().tolist()]
    common.require(
        counts == list(exact["continuation_lengths"].values()),
        "scored continuation lengths differ from token audit",
    )
    full_scores = dict(zip(labels, totals, strict=True))
    mean_scores = {
        label: total / count
        for label, total, count in zip(labels, totals, counts, strict=True)
    }

    shared = int(exact["shared_prefix_length"])
    positions = [start + shared for start in choice_starts]
    common.require(len(set(positions)) == 1, "candidate label positions differ")
    label_position = positions[0]
    for row_index in range(1, len(labels)):
        common.require(
            bool(
                torch.equal(
                    input_ids[0, :label_position],
                    input_ids[row_index, :label_position],
                )
            ),
            "candidate prefixes differ before first label token",
        )
    first_ids = list(exact["first_label_token_ids"].values())
    batched_first_log_probs = torch.log_softmax(
        logits[0, label_position - 1], dim=-1
    )
    batched_first_scores = {
        label: float(batched_first_log_probs[token_id].item())
        for label, token_id in zip(labels, first_ids, strict=True)
    }

    # The task is closed-set natural-label classification. Score the seven
    # unique label tokens from one shared causal prefix, independently of the
    # seven-row full-label sensitivity batch. Repeat the singleton forward to
    # fail closed if the central argmax is not numerically stable.
    central_input_ids = input_ids[0:1, :label_position]
    central_attention_mask = attention_mask[0:1, :label_position]
    chunk_size = evaluator._get_model_chunk_size(model)
    if chunk_size is not None and chunk_size > 1:
        padded_length = min(
            1024,
            ((label_position + chunk_size - 1) // chunk_size) * chunk_size,
        )
        common.require(
            padded_length >= label_position,
            "xLSTM singleton prefix cannot fit its chunk-aligned context",
        )
        padding = padded_length - label_position
        if padding:
            central_input_ids = torch.nn.functional.pad(
                central_input_ids, (0, padding), value=pad_id
            )
            central_attention_mask = torch.nn.functional.pad(
                central_attention_mask, (0, padding), value=0
            )

    def singleton_first_scores() -> dict[str, float]:
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            central_logits = model(
                input_ids=central_input_ids,
                attention_mask=central_attention_mask,
                use_cache=False,
            ).logits[:, label_position - 1, :].float()
        central_log_probs = torch.log_softmax(central_logits[0], dim=-1)
        return {
            label: float(central_log_probs[token_id].item())
            for label, token_id in zip(labels, first_ids, strict=True)
        }

    first_scores = singleton_first_scores()
    repeated_first_scores = singleton_first_scores()
    first_prediction = max(labels, key=first_scores.__getitem__)
    repeated_first_prediction = max(labels, key=repeated_first_scores.__getitem__)
    common.require(
        first_prediction == repeated_first_prediction,
        "singleton first-token argmax is not repeat-stable",
    )
    repeat_max_abs_delta = max(
        abs(first_scores[label] - repeated_first_scores[label]) for label in labels
    )
    batch_max_abs_delta = max(
        abs(first_scores[label] - batched_first_scores[label]) for label in labels
    )
    common.require(
        all(
            math.isfinite(value)
            for value in [
                *full_scores.values(),
                *first_scores.values(),
                *repeated_first_scores.values(),
                *batched_first_scores.values(),
            ]
        ),
        "non-finite interface score",
    )
    return {
        "full_prediction": max(labels, key=full_scores.__getitem__),
        "mean_prediction": max(labels, key=mean_scores.__getitem__),
        "first_token_prediction": first_prediction,
        "first_token_repeat_prediction": repeated_first_prediction,
        "batched_first_token_prediction": max(
            labels, key=batched_first_scores.__getitem__
        ),
        "full_log_likelihoods": full_scores,
        "mean_log_likelihoods": mean_scores,
        "first_token_log_probabilities": first_scores,
        "first_token_repeat_log_probabilities": repeated_first_scores,
        "batched_first_token_log_probabilities": batched_first_scores,
        "first_token_repeat_max_abs_score_delta": repeat_max_abs_delta,
        "first_token_batch_max_abs_score_delta": batch_max_abs_delta,
        "continuation_lengths": dict(zip(labels, counts, strict=True)),
        "first_label_token_ids": exact["first_label_token_ids"],
        "shared_prefix_ids": exact["shared_prefix_ids"],
        "root_text": exact["root_text"],
        "model_chunk_size": chunk_size,
        "singleton_input_length": int(central_input_ids.shape[1]),
    }


def _greedy_label(
    *, model: Any, tokenizer: Any, root_text: str, labels: list[str]
) -> dict[str, Any]:
    import torch

    root_ids = tokenizer.encode(root_text, add_special_tokens=False)
    maximum_new_tokens = 8
    root_ids, removed_prompt_tokens = _left_truncate_for_generation(
        root_ids,
        context_limit=1024,
        maximum_new_tokens=maximum_new_tokens,
    )
    input_ids = torch.tensor([root_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=maximum_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    generated_ids = [int(value) for value in generated[0, len(root_ids) :].tolist()]
    decoded = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()
    valid = decoded if decoded in labels else None
    return {
        "generated_token_ids": generated_ids,
        "decoded": decoded,
        "valid_label": valid,
        "removed_prompt_tokens": removed_prompt_tokens,
    }


def _summaries(
    common: Any,
    rows: list[dict[str, Any]],
    labels: list[str],
    *,
    include_greedy: bool,
) -> Any:
    output = {}
    for language in sorted({row["language"] for row in rows}):
        for prompt_id in sorted({row["prompt_id"] for row in rows}):
            subset = [
                row
                for row in rows
                if row["language"] == language and row["prompt_id"] == prompt_id
            ]
            gold = [row["gold"] for row in subset]
            item: dict[str, Any] = {"n": len(subset)}
            interfaces = [
                ("full", "full_prediction"),
                ("first_token", "first_token_prediction"),
            ]
            if include_greedy:
                interfaces.append(("greedy", "greedy_prediction"))
            for interface, field in interfaces:
                predicted = [row[field] for row in subset]
                metrics = common.manual_classification_metrics(gold, predicted)
                common.validate_metrics_against_sklearn(gold, predicted, metrics)
                item[interface] = {
                    **metrics,
                    "prediction_class_frequency": dict(Counter(predicted)),
                }
            item["full_first_agreement"] = sum(
                row["full_prediction"] == row["first_token_prediction"]
                for row in subset
            ) / len(subset)
            if include_greedy:
                item["greedy_valid_rate"] = sum(
                    row["greedy_prediction"] in labels for row in subset
                ) / len(subset)
                item["greedy_first_agreement"] = sum(
                    row["greedy_prediction"] == row["first_token_prediction"]
                    for row in subset
                ) / len(subset)
            output[f"{language}/{prompt_id}"] = item
    return output


def _top_two_margin(scores: dict[str, float], labels: Sequence[str]) -> float:
    ranked = sorted((float(scores[label]) for label in labels), reverse=True)
    return ranked[0] - ranked[1]


def _full_first_disagreement(
    row: dict[str, Any], labels: Sequence[str]
) -> dict[str, Any]:
    full_scores = row["full_log_likelihoods"]
    first_scores = row["first_token_log_probabilities"]
    return {
        "arm": row["arm"],
        "language": row["language"],
        "prompt_id": row["prompt_id"],
        "selected_source_index": row["selected_source_index"],
        "gold": row["gold"],
        "full_prediction": row["full_prediction"],
        "first_token_prediction": row["first_token_prediction"],
        "full_top_two_margin": _top_two_margin(full_scores, labels),
        "first_token_top_two_margin": _top_two_margin(first_scores, labels),
        "maximum_abs_score_delta": max(
            abs(float(full_scores[label]) - float(first_scores[label]))
            for label in labels
        ),
        "full_log_likelihoods": full_scores,
        "first_token_log_probabilities": first_scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm-binding", type=Path)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--skip-greedy", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    global common
    common = _load_common(repo)
    protocol = common.load_and_check_protocol(repo, args.protocol.resolve())
    common.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "GPU1 is required")
    common.require(args.limit in (0, 8), "sealed diagnostic limit must be 0 or 8")
    asset_root = args.asset_root.resolve()
    output_root = args.output_root.resolve()
    common.require(asset_root.is_dir(), "missing retained validation assets")
    selected_binding = None
    if args.arm_binding is not None:
        protocol, selected_binding = _select_bound_arm(
            common=common,
            protocol=protocol,
            binding_path=args.arm_binding.resolve(),
            asset_root=asset_root,
            protocol_path=args.protocol.resolve(),
        )
    common.require(not output_root.exists(), "diagnostic output already exists")
    output_root.mkdir(parents=True)

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
    from sallm.evaluation.harness import _prepare_tokenizer, load_model_and_tokenizer
    from transformers import AutoTokenizer

    validation_rows = _read_validation_rows(common, protocol, asset_root)
    token_audit_path = output_root / "token_audit.jsonl"
    token_audit = _audit_all_contexts(
        common=common,
        protocol=protocol,
        repo=repo,
        asset_root=asset_root,
        validation_rows=validation_rows,
        output_path=token_audit_path,
    )
    common._write_json(output_root / "token_audit_summary.json", token_audit)

    templates = common._load_templates(repo, protocol)
    labels = list(protocol["labels"])
    base_spec = (
        selected_binding.get("base", protocol["base"])
        if selected_binding is not None
        else protocol["base"]
    )
    anchors = {}
    base_model, base_tokenizer = load_model_and_tokenizer(
        ModelEvalConfig(
            checkpoint=str(asset_root / "base"),
            dtype=protocol["runtime"]["dtype"],
            device=protocol["runtime"]["device"],
            merge_lora=False,
            tie_word_embeddings=False,
        )
    )
    if selected_binding is not None:
        runtime_adapter = asset_root / "runtime_adapters" / protocol["arms"][0]["id"]
        base_tokenizer = _prepare_tokenizer(
            AutoTokenizer.from_pretrained(
                runtime_adapter,
                local_files_only=True,
                trust_remote_code=True,
            )
        )
        base_anchor_vocabulary = _align_base_anchor_vocabulary(
            model=base_model,
            tokenizer=base_tokenizer,
        )
    else:
        base_tokenizer.chat_template = common.NORMALIZED_CHAT_TEMPLATE
        base_anchor_vocabulary = _align_base_anchor_vocabulary(
            model=base_model,
            tokenizer=base_tokenizer,
        )
    base_evaluator = ClassificationEvaluator(base_tokenizer, max_samples_per_lang=None)
    for arm in protocol["arms"]:
        language = arm["languages"][0]
        row = common.select_label_round_robin(
            validation_rows[language], labels, args.limit
        )[0]
        prompt = templates["p1"].format(headline=row["headline"], text=row["text"])
        anchors[arm["id"]] = _score_interfaces(
            model=base_model,
            tokenizer=base_tokenizer,
            evaluator=base_evaluator,
            prompt=prompt,
            labels=labels,
        )
    import torch

    del base_evaluator, base_tokenizer, base_model
    torch.cuda.empty_cache()

    arm_reports = []
    expected_records = {
        arm["id"]: sum(
            (len(validation_rows[language]) if args.limit == 0 else args.limit)
            * len(templates)
            for language in arm["languages"]
        )
        for arm in protocol["arms"]
    }
    for arm in protocol["arms"]:
        runtime_adapter = asset_root / "runtime_adapters" / arm["id"]
        model, tokenizer = load_model_and_tokenizer(
            ModelEvalConfig(
                checkpoint=str(asset_root / "base"),
                peft_adapter=str(runtime_adapter),
                dtype=protocol["runtime"]["dtype"],
                device=protocol["runtime"]["device"],
                merge_lora=True,
                tie_word_embeddings=False,
            )
        )
        model.eval()
        evaluator = ClassificationEvaluator(tokenizer, max_samples_per_lang=None)
        predictions = []
        for language in arm["languages"]:
            selected = common.select_label_round_robin(
                validation_rows[language], labels, args.limit
            )
            for source_index, row in enumerate(selected):
                for prompt_id, template in templates.items():
                    prompt = template.format(
                        headline=row["headline"], text=row["text"]
                    )
                    scored = _score_interfaces(
                        model=model,
                        tokenizer=tokenizer,
                        evaluator=evaluator,
                        prompt=prompt,
                        labels=labels,
                    )
                    prediction = {
                        "arm": arm["id"],
                        "language": language,
                        "prompt_id": prompt_id,
                        "selected_source_index": source_index,
                        "gold": row["category"].strip(),
                        **scored,
                    }
                    if args.skip_greedy:
                        prediction.pop("root_text")
                    else:
                        greedy = _greedy_label(
                            model=model,
                            tokenizer=tokenizer,
                            root_text=prediction.pop("root_text"),
                            labels=labels,
                        )
                        prediction["greedy_prediction"] = (
                            greedy["valid_label"] or "__invalid__"
                        )
                        prediction["greedy"] = greedy
                    predictions.append(prediction)
        common.require(
            len(predictions) == expected_records[arm["id"]],
            f"diagnostic prediction completeness mismatch for {arm['id']}",
        )
        common.require(
            all(
                row["first_token_prediction"]
                == row["first_token_repeat_prediction"]
                for row in predictions
            ),
            f"central first-token argmax instability for {arm['id']}",
        )
        anchor = predictions[0]
        base_anchor = anchors[arm["id"]]
        full_delta = max(
            abs(
                anchor["full_log_likelihoods"][label]
                - base_anchor["full_log_likelihoods"][label]
            )
            for label in labels
        )
        first_delta = max(
            abs(
                anchor["first_token_log_probabilities"][label]
                - base_anchor["first_token_log_probabilities"][label]
            )
            for label in labels
        )
        common.require(
            max(full_delta, first_delta) > 1e-7,
            f"adapter does not change anchor choice logits for {arm['id']}",
        )
        predictions_path = output_root / f"{arm['id']}_predictions.jsonl"
        predictions_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
            encoding="utf-8",
        )
        disagreements = [
            _full_first_disagreement(row, labels)
            for row in predictions
            if row["full_prediction"] != row["first_token_prediction"]
        ]
        disagreements_path = (
            output_root / f"{arm['id']}_full_first_disagreements.jsonl"
        )
        disagreements_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in disagreements),
            encoding="utf-8",
        )
        summary = {
            "arm": arm["id"],
            "records": len(predictions),
            "expected_records": expected_records[arm["id"]],
            "adapter_vs_base_max_abs_full_score_delta": full_delta,
            "adapter_vs_base_max_abs_first_token_score_delta": first_delta,
            "full_first_disagreement_count": len(disagreements),
            "full_first_disagreements_sha256": common.file_sha256(
                disagreements_path
            ),
            "batched_first_singleton_disagreement_count": sum(
                row["batched_first_token_prediction"]
                != row["first_token_prediction"]
                for row in predictions
            ),
            "maximum_singleton_repeat_score_delta": max(
                row["first_token_repeat_max_abs_score_delta"]
                for row in predictions
            ),
            "maximum_batched_singleton_first_score_delta": max(
                row["first_token_batch_max_abs_score_delta"]
                for row in predictions
            ),
            "subsets": _summaries(
                common,
                predictions,
                labels,
                include_greedy=not args.skip_greedy,
            ),
            "predictions_sha256": common.file_sha256(predictions_path),
        }
        summary_path = output_root / f"{arm['id']}_summary.json"
        common._write_json(summary_path, summary)
        arm_reports.append(
            {
                "arm": arm["id"],
                "summary_sha256": common.file_sha256(summary_path),
                "predictions_sha256": common.file_sha256(predictions_path),
                "full_first_disagreements_sha256": common.file_sha256(
                    disagreements_path
                ),
            }
        )
        del evaluator, tokenizer, model
        torch.cuda.empty_cache()

    manifest = {
        "schema": "sallm.mamba_news_interface_diagnostic/v1",
        "data_boundary": "validation_only",
        "test_accessed": False,
        "protocol_sha256": common.file_sha256(args.protocol.resolve()),
        "arm_binding_sha256": (
            common.file_sha256(args.arm_binding.resolve())
            if args.arm_binding is not None
            else None
        ),
        "selected_adapter_revision": (
            selected_binding["arm"]["adapter_revision"]
            if selected_binding is not None
            else None
        ),
        "base_revision": base_spec.get("revision"),
        "selected_architecture": (
            selected_binding.get("architecture")
            if selected_binding is not None
            else "mamba"
        ),
        "base_anchor_vocabulary": base_anchor_vocabulary,
        "limit_per_language_per_prompt": args.limit,
        "greedy_generation": (
            "skipped_full_closed_set_run_after_v7_canary"
            if args.skip_greedy
            else "enabled"
        ),
        "gates": {
            "training_token_reconstruction": "passed",
            "unique_first_label_tokens": "passed",
            "completeness": "passed",
            "numerical_finiteness": "passed",
            "metric_recomputation": "passed",
            "adapter_changes_base_scores": "passed",
            "central_first_token_argmax_repeat_stability": "passed",
            "full_label_agreement": "reported_sensitivity_not_validity_gate",
            "prediction_distribution": "diagnostic_only_not_an_admission_gate",
        },
        "token_audit": token_audit,
        "arms": arm_reports,
    }
    common._write_json(output_root / "manifest.json", manifest)
    print(f"MAMBA_NEWS_INTERFACE_DIAGNOSTIC_COMPLETE {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
