#!/usr/bin/env python3
"""Evaluate retained Mamba News adapters on a sealed validation-only protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

PROTOCOL_OK = "MAMBA_NEWS_COMMON_VALIDATION_PROTOCOL_OK"
NORMALIZED_CHAT_TEMPLATE = """{%- if system_message %}
<|system|>
{{ system_message }}{{ eos_token }}
{%- endif %}
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        <|user|>
        {{ message['content'] }}{{ eos_token }}
    {%- elif message['role'] == 'assistant' %}
        {%- generation -%}
        <|assistant|>
        {{ message['content'] }}{{ eos_token }}
        {%- endgeneration -%}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|assistant|>
{{ '        ' }}{% endif %}
"""
GENERATION_PROMPT_SUFFIX = "<|assistant|>\n        "


class ProtocolError(ValueError):
    """Raised when a sealed validation input fails closed."""


class _TaggedSafeLoader(yaml.SafeLoader):
    pass


def _construct_function_tag(loader: yaml.SafeLoader, node: yaml.nodes.Node) -> object:
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)


_TaggedSafeLoader.add_constructor("!function", _construct_function_tag)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_tree_sha256(repo: Path, root: Path) -> str:
    lines = "".join(
        f"{file_sha256(path)}  {path.relative_to(repo).as_posix()}\n"
        for path in sorted(root.iterdir())
        if path.is_file()
    )
    return bytes_sha256(lines.encode())


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=_TaggedSafeLoader)
    require(isinstance(value, dict), f"expected YAML mapping: {path}")
    return value


def _template_prompt_from_task(prompt: str) -> str:
    return prompt.replace("{{headline}}", "{headline}").replace("{{text}}", "{text}")


def _check_task_definitions(repo: Path, protocol: dict[str, Any]) -> None:
    pack = protocol["task_pack"]
    root = repo / pack["task_definition_root"]
    common = load_yaml(root / "_masakhanews_val_common.yaml")
    require(common.get("dataset_path") == "csv", "News validation must use CSV")
    require(
        common.get("validation_split") == "validation",
        "News task must expose only the validation split",
    )
    require("test_split" not in common, "News common task declares a test split")

    templates = {item["id"]: item for item in protocol["templates"]}
    files = {item["language"]: item for item in protocol["dataset"]["allowed_files"]}
    for language in ("eng", "xho"):
        for prompt_number in range(1, 6):
            prompt_id = f"p{prompt_number}"
            task_path = root / (
                f"sallm_masakhanews_{language}_val_prompt_{prompt_number}.yaml"
            )
            task = load_yaml(task_path)
            require(
                task.get("include") == "_masakhanews_val_common.yaml",
                f"unexpected include in {task_path}",
            )
            require(
                "dataset_name" not in task,
                f"dataset_name present in {task_path}",
            )
            data_files = task.get("dataset_kwargs", {}).get("data_files", {})
            require(
                set(data_files) == {"validation"},
                f"non-validation data file declared in {task_path}",
            )
            expected_url = files[language]["url"]
            require(
                data_files["validation"] == expected_url,
                f"unpinned validation URL in {task_path}",
            )
            require(
                expected_url.endswith(f"/data/{language}/dev.tsv"),
                f"allowed URL is not the {language} development split",
            )
            require(
                "/test.tsv" not in task_path.read_text(encoding="utf-8").lower(),
                f"test filename appears in {task_path}",
            )
            template_path = repo / templates[prompt_id]["path"]
            template = load_yaml(template_path)
            require(
                _template_prompt_from_task(str(task["doc_to_text"]))
                == str(template["prompt"]),
                f"prompt text drift for {language} {prompt_id}",
            )
            require(
                list(template["label_mapping"].values()) == protocol["labels"],
                f"label order drift for {prompt_id}",
            )


def _check_training_bindings(repo: Path, protocol: dict[str, Any]) -> None:
    expected_targets = {"in_proj", "x_proj"}
    for arm in protocol["arms"]:
        config_spec = arm["training_config"]
        config_path = repo / config_spec["path"]
        require(
            file_sha256(config_path) == config_spec["sha256"],
            f"training config hash mismatch for {arm['id']}",
        )
        config = load_yaml(config_path)
        targets = set(config["peft"]["kwargs"]["target_modules"])
        require(targets == expected_targets, f"target drift for {arm['id']}")

        sweep_spec = arm["bayesian_sweep_config"]
        sweep_path = repo / sweep_spec["path"]
        require(
            file_sha256(sweep_path) == sweep_spec["sha256"],
            f"sweep config hash mismatch for {arm['id']}",
        )
        sweep = load_yaml(sweep_path)
        require(
            sweep.get("method") == "bayes",
            f"non-Bayesian sweep for {arm['id']}",
        )
        parameters = sweep.get("parameters", {})
        require(
            "peft.kwargs.target_modules" not in parameters,
            f"target modules were unexpectedly swept for {arm['id']}",
        )

        source_log = Path(arm["source_log"])
        if source_log.exists():
            require(
                file_sha256(source_log) == arm["source_log_sha256"],
                f"source log hash mismatch for {arm['id']}",
            )
            log_text = source_log.read_text(encoding="utf-8", errors="replace")
            require(
                f"Pushing adapter to HuggingFace Hub: {arm['adapter_repo']}"
                in log_text,
                f"job-to-Hub push binding missing for {arm['id']}",
            )
            require(
                f"Successfully pushed to {arm['adapter_repo']}" in log_text,
                f"job-to-Hub success binding missing for {arm['id']}",
            )

    general_spec = protocol["related_retained_general"]["training_config"]
    general_path = repo / general_spec["path"]
    require(
        file_sha256(general_path) == general_spec["sha256"],
        "retained General Mamba config hash mismatch",
    )
    general = load_yaml(general_path)
    require(
        set(general["peft"]["kwargs"]["target_modules"]) == expected_targets,
        "retained General Mamba target modules differ from News",
    )


def load_and_check_protocol(repo: Path, path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    require(
        protocol.get("schema") == "sallm.mamba_news_common_validation/v2",
        "unexpected protocol schema",
    )
    require(
        protocol["data_boundary"] == "validation_only",
        "invalid data boundary",
    )
    require(
        protocol["test_access_allowed"] is False,
        "test access is not forbidden",
    )
    require(
        protocol["final_matrix_eligible"] is False,
        "canary must be diagnostic",
    )

    source = protocol["source_standardization"]
    require(
        file_sha256(repo / source["path"]) == source["sha256"],
        "parent standardization protocol hash mismatch",
    )
    pack = protocol["task_pack"]
    require(pack["scope"] == "rerank", "unexpected task-pack scope")
    require(pack["name"] == "masakhanews_all_val", "unexpected task pack")
    require(pack["apply_chat_template"] is True, "chat templating is disabled")
    require(
        file_sha256(repo / pack["path"]) == pack["sha256"],
        "task-pack hash mismatch",
    )
    require(
        task_tree_sha256(repo, repo / pack["task_definition_root"])
        == pack["task_definition_tree_sha256"],
        "task-definition tree hash mismatch",
    )

    for template in protocol["templates"]:
        require(
            file_sha256(repo / template["path"]) == template["sha256"],
            f"template hash mismatch for {template['id']}",
        )
    require(
        [arm["id"] for arm in protocol["arms"]] == ["mono_eng", "mono_xho", "multi"],
        "arm order drift",
    )
    require(
        bytes_sha256(NORMALIZED_CHAT_TEMPLATE.encode())
        == protocol["runtime"]["prompt_normalization"]["normalized_sha256"],
        "runtime chat-template hash mismatch",
    )
    require(
        NORMALIZED_CHAT_TEMPLATE.endswith("{{ '        ' }}{% endif %}\n"),
        "runtime chat-template suffix drift",
    )
    require(
        protocol["runtime"]["classification_decoding"]
        == "deterministic_constrained_mean_continuation_log_probability",
        "classification score mode drift",
    )
    require(
        protocol["runtime"]["maximum_input_tokens"] == 1024,
        "input-token cap drift",
    )
    _check_task_definitions(repo, protocol)
    _check_training_bindings(repo, protocol)
    metric_self_check()
    return protocol


def parse_validation_tsv(payload: bytes, spec: dict[str, Any]) -> list[dict[str, str]]:
    require(
        bytes_sha256(payload) == spec["sha256"],
        "validation TSV hash mismatch",
    )
    require(len(payload) == spec["bytes"], "validation TSV byte count mismatch")
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    require(reader.fieldnames == spec["columns"], "validation TSV schema mismatch")
    rows = [dict(row) for row in reader]
    require(len(rows) == spec["rows"], "validation TSV row count mismatch")
    observed = sorted({row["category"].strip() for row in rows})
    require(observed == spec["observed_labels"], "validation label set mismatch")
    for row in rows:
        require(
            all(row.get(column) is not None for column in spec["columns"]),
            "ragged TSV row",
        )
    return rows


def materialize_validation_files(
    dataset_spec: dict[str, Any],
    destination: Path,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, list[dict[str, str]]]:
    require(
        not destination.exists(),
        f"validation destination exists: {destination}",
    )
    destination.mkdir(parents=True)
    materialized: dict[str, list[dict[str, str]]] = {}
    for spec in dataset_spec["allowed_files"]:
        url = spec["url"]
        language = spec["language"]
        require(
            url.endswith(f"/data/{language}/dev.tsv"),
            f"refusing non-development URL: {url}",
        )
        require("/test.tsv" not in url.lower(), f"refusing test URL: {url}")
        with opener(url, timeout=120) as response:
            payload = response.read()
        rows = parse_validation_tsv(payload, spec)
        path = destination / f"{language}-dev.tsv"
        path.write_bytes(payload)
        materialized[language] = rows
    return materialized


def stage_snapshot(repo_id: str, revision: str, destination: Path) -> Path:
    from huggingface_hub import snapshot_download

    require(
        not destination.exists(),
        f"snapshot destination exists: {destination}",
    )
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=destination,
            max_workers=2,
        )
    ).resolve()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    }


def copy_and_normalize_adapter(
    source: Path,
    destination: Path,
    *,
    expected_source_sha256: str,
    expected_normalized_sha256: str,
) -> Path:
    require(not destination.exists(), f"runtime adapter exists: {destination}")
    source_template = source / "chat_template.jinja"
    require(
        file_sha256(source_template) == expected_source_sha256,
        "source adapter chat-template hash mismatch",
    )
    source_hashes_before = _tree_hashes(source)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".cache"))
    runtime_template = destination / "chat_template.jinja"
    runtime_template.write_text(NORMALIZED_CHAT_TEMPLATE, encoding="utf-8")
    require(
        file_sha256(runtime_template) == expected_normalized_sha256,
        "normalized runtime chat-template hash mismatch",
    )
    require(
        _tree_hashes(source) == source_hashes_before,
        "immutable source adapter changed during normalization",
    )
    runtime_hashes = _tree_hashes(destination)
    for relative, digest in source_hashes_before.items():
        if relative != "chat_template.jinja":
            require(
                runtime_hashes.get(relative) == digest,
                f"runtime adapter changed non-template file: {relative}",
            )
    require(
        set(runtime_hashes) == set(source_hashes_before),
        "runtime adapter file set differs from source",
    )
    return destination


def verify_adapter_snapshot(source: Path, arm: dict[str, Any]) -> None:
    require(
        file_sha256(source / "adapter_model.safetensors")
        == arm["adapter_weights_sha256"],
        f"adapter weight hash mismatch for {arm['id']}",
    )
    require(
        file_sha256(source / "adapter_config.json") == arm["adapter_config_sha256"],
        f"adapter config hash mismatch for {arm['id']}",
    )
    require(
        file_sha256(source / "chat_template.jinja")
        == arm["source_chat_template_sha256"],
        f"source chat-template hash mismatch for {arm['id']}",
    )
    require(
        file_sha256(source / "tokenizer_config.json") == arm["tokenizer_config_sha256"],
        f"tokenizer config hash mismatch for {arm['id']}",
    )
    actual = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    expected = arm["expected_adapter_config"]
    for key in (
        "base_model_name_or_path",
        "r",
        "lora_alpha",
        "lora_dropout",
        "modules_to_save",
    ):
        require(
            actual.get(key) == expected[key],
            f"adapter {key} mismatch for {arm['id']}",
        )
    require(
        set(actual.get("target_modules", [])) == set(expected["target_modules"]),
        f"adapter target-module mismatch for {arm['id']}",
    )


def select_label_round_robin(
    rows: Sequence[dict[str, str]], labels: Sequence[str], limit: int
) -> list[dict[str, str]]:
    if limit == 0:
        return list(rows)
    require(limit <= len(rows), "label-round-robin limit exceeds row count")
    queues = {
        label: [row for row in rows if row["category"].strip() == label]
        for label in labels
    }
    offsets = {label: 0 for label in labels}
    selected: list[dict[str, str]] = []
    while len(selected) < limit:
        previous = len(selected)
        for label in labels:
            offset = offsets[label]
            if offset < len(queues[label]):
                selected.append(queues[label][offset])
                offsets[label] += 1
                if len(selected) == limit:
                    break
        require(
            len(selected) > previous,
            "unable to satisfy label-round-robin limit",
        )
    return selected


def manual_classification_metrics(
    gold: Sequence[str], predicted: Sequence[str]
) -> dict[str, float]:
    require(len(gold) == len(predicted), "gold/prediction length mismatch")
    require(bool(gold), "cannot compute metrics for zero rows")
    observed = list(dict.fromkeys(gold))
    f1_by_label: dict[str, float] = {}
    support = Counter(gold)
    for label in observed:
        true_positive = sum(
            actual == label and guess == label
            for actual, guess in zip(gold, predicted, strict=True)
        )
        false_positive = sum(
            actual != label and guess == label
            for actual, guess in zip(gold, predicted, strict=True)
        )
        false_negative = sum(
            actual == label and guess != label
            for actual, guess in zip(gold, predicted, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1_by_label[label] = 2 * true_positive / denominator if denominator else 0.0
    total = len(gold)
    return {
        "accuracy": sum(
            actual == guess for actual, guess in zip(gold, predicted, strict=True)
        )
        / total,
        "weighted_f1": sum(f1_by_label[label] * support[label] for label in observed)
        / total,
        "macro_f1": sum(f1_by_label.values()) / len(observed),
    }


def validate_metrics_against_sklearn(
    gold: Sequence[str], predicted: Sequence[str], metrics: dict[str, float]
) -> None:
    from sklearn.metrics import accuracy_score, f1_score

    observed = list(dict.fromkeys(gold))
    reference = {
        "accuracy": float(accuracy_score(gold, predicted)),
        "weighted_f1": float(
            f1_score(gold, predicted, average="weighted", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(
                gold,
                predicted,
                labels=observed,
                average="macro",
                zero_division=0,
            )
        ),
    }
    for name, expected in reference.items():
        require(
            math.isclose(metrics[name], expected, rel_tol=0.0, abs_tol=1e-12),
            f"independent {name} recomputation mismatch",
        )


def rank_mean_scores(
    labels: Sequence[str],
    totals: Sequence[float],
    token_counts: Sequence[int],
) -> tuple[str, dict[str, float]]:
    require(
        len(labels) == len(totals) == len(token_counts),
        "choice score shape mismatch",
    )
    require(all(count > 0 for count in token_counts), "empty label continuation")
    require(
        all(math.isfinite(float(total)) for total in totals),
        "non-finite continuation log-probability total",
    )
    scores = {
        label: total / count
        for label, total, count in zip(labels, totals, token_counts, strict=True)
    }
    require(
        all(math.isfinite(float(score)) for score in scores.values()),
        "non-finite mean continuation score",
    )
    best = max(range(len(labels)), key=lambda index: scores[labels[index]])
    return labels[best], scores


def metric_self_check() -> None:
    gold = ["a", "a", "b", "b"]
    predicted = ["a", "b", "b", "b"]
    metrics = manual_classification_metrics(gold, predicted)
    require(
        math.isclose(metrics["accuracy"], 0.75),
        "accuracy self-check failed",
    )
    require(
        math.isclose(metrics["weighted_f1"], 11 / 15),
        "weighted-F1 self-check failed",
    )
    require(
        math.isclose(metrics["macro_f1"], 11 / 15),
        "macro-F1 self-check failed",
    )
    validate_metrics_against_sklearn(gold, predicted, metrics)
    winner, scores = rank_mean_scores(["short", "long"], [-2.0, -3.0], [1, 3])
    require(winner == "long", "mean-token ranking self-check failed")
    raw = {"short": -2.0, "long": -3.0}
    require(max(raw, key=raw.get) == "short", "raw-sum counterexample failed")
    require(scores["long"] > scores["short"], "mean-score ordering failed")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _score_one(
    *,
    model: Any,
    tokenizer: Any,
    evaluator: Any,
    prompt: str,
    labels: list[str],
    maximum_input_tokens: int,
    tie_tolerance: float,
    near_tie_tolerance: float,
) -> dict[str, Any]:
    import torch
    from sallm.evaluation.classification_metrics import _gather_target_log_probs

    prompt_text = evaluator._build_prompt_text(
        prompt_messages=[{"role": "user", "content": prompt}],
        fallback_template=None,
        system_message=None,
    )
    require(
        prompt_text.endswith(GENERATION_PROMPT_SUFFIX),
        "rendered generation-prompt suffix differs from repaired contract",
    )
    pad_token_id = evaluator._resolve_pad_id(
        tokenizer.pad_token_id, tokenizer.eos_token_id
    )
    device = getattr(model, "device", torch.device("cuda:0"))
    input_ids, attention_mask, choice_starts = evaluator._build_choice_inputs(
        prompt_text=prompt_text,
        label_choices=labels,
        model_ctx_limit=maximum_input_tokens,
        pad_token_id=pad_token_id,
        device=device,
        pad_to_multiple_of=None,
    )
    # FLA keeps Mamba residuals in fp32; autocast returns them to bf16 at each
    # projection, matching the mixed-precision training path.
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits[:, :-1, :].float()
    target_ids = input_ids[:, 1:]
    token_log_probs = _gather_target_log_probs(
        logits=logits,
        target_ids=target_ids,
    )
    require(
        bool(torch.isfinite(token_log_probs).all().item()),
        "non-finite target-token log probability",
    )
    continuation_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    sequence_lengths = attention_mask.sum(dim=1)
    for index, start in enumerate(choice_starts):
        sequence_length = int(sequence_lengths[index].item())
        if sequence_length > start:
            continuation_mask[index, start:sequence_length] = True
    continuation_mask = continuation_mask[:, 1:] & attention_mask[:, 1:].bool()
    totals_tensor = token_log_probs.masked_fill(~continuation_mask, 0.0).sum(dim=1)
    counts_tensor = continuation_mask.sum(dim=1)
    totals = [float(value) for value in totals_tensor.detach().cpu().tolist()]
    counts = [int(value) for value in counts_tensor.detach().cpu().tolist()]
    prediction, scores = rank_mean_scores(labels, totals, counts)

    ranking = sorted(
        range(len(labels)),
        key=lambda index: (-scores[labels[index]], index),
    )
    best_score = scores[labels[ranking[0]]]
    second_score = scores[labels[ranking[1]]]
    exact_ties = [
        label for label in labels if abs(scores[label] - best_score) <= tie_tolerance
    ]
    margin = best_score - second_score
    require(math.isfinite(margin), "non-finite top-two margin")
    return {
        "prediction": prediction,
        "scores": scores,
        "token_counts": dict(zip(labels, counts, strict=True)),
        "top_two_margin": margin,
        "exact_tie": len(exact_ties) > 1,
        "exact_tie_labels": exact_ties,
        "near_tie": margin <= near_tie_tolerance,
        "maximum_scored_sequence_tokens": max(
            int(value) for value in sequence_lengths.tolist()
        ),
    }


def validate_prediction_integrity(
    predictions: list[dict[str, Any]],
    *,
    arm: dict[str, Any],
    validation_rows: dict[str, list[dict[str, str]]],
    template_ids: Sequence[str],
    labels: Sequence[str],
    limit: int,
) -> dict[str, Any]:
    """Fail closed on missing, duplicate, or numerically invalid predictions."""
    expected_keys: set[tuple[str, str, int]] = set()
    expected_by_language_prompt: dict[str, int] = {}
    for language in arm["languages"]:
        selected_count = len(validation_rows[language]) if limit == 0 else limit
        for prompt_id in template_ids:
            expected_by_language_prompt[f"{language}/{prompt_id}"] = selected_count
            expected_keys.update(
                (language, prompt_id, source_index)
                for source_index in range(selected_count)
            )

    observed_keys = [
        (row["language"], row["prompt_id"], row["selected_source_index"])
        for row in predictions
    ]
    require(
        len(observed_keys) == len(set(observed_keys)),
        f"duplicate prediction key for {arm['id']}",
    )
    require(
        set(observed_keys) == expected_keys,
        f"prediction completeness mismatch for {arm['id']}",
    )
    label_set = set(labels)
    for row in predictions:
        require(row["arm"] == arm["id"], f"prediction arm drift for {arm['id']}")
        require(row["gold"] in label_set, f"unknown gold label for {arm['id']}")
        require(
            row["prediction"] in label_set,
            f"unknown predicted label for {arm['id']}",
        )
        require(
            set(row["scores"]) == label_set,
            f"choice score labels drift for {arm['id']}",
        )
        require(
            set(row["token_counts"]) == label_set,
            f"choice token-count labels drift for {arm['id']}",
        )
        require(
            all(math.isfinite(float(value)) for value in row["scores"].values()),
            f"non-finite choice score for {arm['id']}",
        )
        require(
            math.isfinite(float(row["top_two_margin"])),
            f"non-finite top-two margin for {arm['id']}",
        )

    return {
        "expected_prediction_records": len(expected_keys),
        "observed_prediction_records": len(observed_keys),
        "expected_records_by_language_prompt": expected_by_language_prompt,
        "completeness": "passed",
        "numerical_finiteness": "passed",
    }


def _summarize_predictions(
    predictions: list[dict[str, Any]], labels: list[str], evaluator: Any
) -> dict[str, Any]:
    subsets: dict[str, Any] = {}
    for language in sorted({row["language"] for row in predictions}):
        for prompt_id in sorted({row["prompt_id"] for row in predictions}):
            rows = [
                row
                for row in predictions
                if row["language"] == language and row["prompt_id"] == prompt_id
            ]
            if not rows:
                continue
            gold = [row["gold"] for row in rows]
            predicted = [row["prediction"] for row in rows]
            metrics = manual_classification_metrics(gold, predicted)
            validate_metrics_against_sklearn(gold, predicted, metrics)
            sallm_metrics = evaluator._compute_classification_metrics(gold, predicted)
            require(
                math.isclose(
                    metrics["accuracy"],
                    sallm_metrics["accuracy"],
                    abs_tol=1e-12,
                ),
                "SALLM accuracy disagrees with independent recomputation",
            )
            require(
                math.isclose(
                    metrics["weighted_f1"],
                    sallm_metrics["f1"],
                    abs_tol=1e-12,
                ),
                "SALLM weighted F1 disagrees with independent recomputation",
            )
            require(
                math.isclose(
                    metrics["macro_f1"],
                    sallm_metrics["macro_f1"],
                    abs_tol=1e-12,
                ),
                "SALLM macro F1 disagrees with independent recomputation",
            )
            gold_counts = Counter(gold)
            prediction_counts = Counter(predicted)
            margins = [float(row["top_two_margin"]) for row in rows]
            subsets[f"{language}/{prompt_id}"] = {
                "n": len(rows),
                **metrics,
                "gold_class_frequency": {label: gold_counts[label] for label in labels},
                "prediction_class_frequency": {
                    label: prediction_counts[label] for label in labels
                },
                "exact_tie_count": sum(bool(row["exact_tie"]) for row in rows),
                "exact_tie_rate": sum(bool(row["exact_tie"]) for row in rows)
                / len(rows),
                "near_tie_count": sum(bool(row["near_tie"]) for row in rows),
                "minimum_top_two_margin": min(margins),
                "mean_top_two_margin": sum(margins) / len(margins),
            }
    language_means: dict[str, dict[str, float]] = {}
    languages = sorted({key.split("/", 1)[0] for key in subsets})
    for language in languages:
        items = [
            value for key, value in subsets.items() if key.startswith(f"{language}/")
        ]
        language_means[language] = {
            metric: sum(item[metric] for item in items) / len(items)
            for metric in ("accuracy", "weighted_f1", "macro_f1")
        }
    return {
        "subsets": subsets,
        "unweighted_prompt_means_by_language": language_means,
    }


def _load_templates(repo: Path, protocol: dict[str, Any]) -> dict[str, str]:
    return {
        template["id"]: str(load_yaml(repo / template["path"])["prompt"])
        for template in protocol["templates"]
    }


def run_evaluation(
    repo: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    asset_root: Path,
    output_root: Path,
    limit: int,
) -> None:
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "1",
        "CUDA_VISIBLE_DEVICES must be exactly 1; GPU0 is protected",
    )
    require(not asset_root.exists(), f"asset root already exists: {asset_root}")
    require(
        not output_root.exists(),
        f"output root already exists: {output_root}",
    )
    asset_root.mkdir(parents=True)

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    base_spec = protocol["base"]
    base = stage_snapshot(
        base_spec["repo"],
        base_spec["revision"],
        asset_root / "base",
    )
    require(
        file_sha256(base / base_spec["weights_file"]) == base_spec["weights_sha256"],
        "base checkpoint weight hash mismatch",
    )
    validation_rows = materialize_validation_files(
        protocol["dataset"],
        asset_root / "dataset",
    )

    runtime_adapters: dict[str, Path] = {}
    for arm in protocol["arms"]:
        source = stage_snapshot(
            arm["adapter_repo"],
            arm["adapter_revision"],
            asset_root / "source_adapters" / arm["id"],
        )
        verify_adapter_snapshot(source, arm)
        runtime_adapters[arm["id"]] = copy_and_normalize_adapter(
            source,
            asset_root / "runtime_adapters" / arm["id"],
            expected_source_sha256=arm["source_chat_template_sha256"],
            expected_normalized_sha256=protocol["runtime"]["prompt_normalization"][
                "normalized_sha256"
            ],
        )

    for name in (
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        os.environ[name] = "1"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["PYTHONPATH"] = str(repo / "src/main")
    if str(repo / "src/main") not in sys.path:
        sys.path.insert(0, str(repo / "src/main"))

    import torch
    from sallm.config import ModelEvalConfig
    from sallm.evaluation.classification_metrics import (
        ChoiceScoreMode,
        ClassificationEvaluator,
    )
    from sallm.evaluation.harness import load_model_and_tokenizer
    from transformers import AutoTokenizer

    templates = _load_templates(repo, protocol)
    labels = list(protocol["labels"])
    output_root.mkdir(parents=True)
    arm_manifests: list[dict[str, Any]] = []
    for arm in protocol["arms"]:
        runtime_adapter = runtime_adapters[arm["id"]]
        tokenizer_check = AutoTokenizer.from_pretrained(
            runtime_adapter,
            trust_remote_code=True,
            local_files_only=True,
        )
        require(
            tokenizer_check.chat_template == NORMALIZED_CHAT_TEMPLATE,
            f"tokenizer did not load repaired chat template for {arm['id']}",
        )
        rendered_check = tokenizer_check.apply_chat_template(
            [{"role": "user", "content": "protocol check"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        require(
            rendered_check.endswith(GENERATION_PROMPT_SUFFIX),
            f"tokenizer rendered wrong generation suffix for {arm['id']}",
        )
        del tokenizer_check

        model, tokenizer = load_model_and_tokenizer(
            ModelEvalConfig(
                checkpoint=str(base),
                peft_adapter=str(runtime_adapter),
                dtype=protocol["runtime"]["dtype"],
                device=protocol["runtime"]["device"],
                merge_lora=True,
                tie_word_embeddings=False,
            )
        )
        model.eval()
        evaluator = ClassificationEvaluator(
            tokenizer,
            max_samples_per_lang=None,
            choice_score_mode=ChoiceScoreMode.MEAN,
        )
        predictions: list[dict[str, Any]] = []
        for language in arm["languages"]:
            rows = select_label_round_robin(validation_rows[language], labels, limit)
            for source_index, row in enumerate(rows):
                for prompt_id, template in templates.items():
                    prompt = template.format(
                        headline=row["headline"],
                        text=row["text"],
                    )
                    scored = _score_one(
                        model=model,
                        tokenizer=tokenizer,
                        evaluator=evaluator,
                        prompt=prompt,
                        labels=labels,
                        maximum_input_tokens=protocol["runtime"][
                            "maximum_input_tokens"
                        ],
                        tie_tolerance=protocol["runtime"]["tie_absolute_tolerance"],
                        near_tie_tolerance=protocol["runtime"][
                            "near_tie_absolute_tolerance"
                        ],
                    )
                    predictions.append(
                        {
                            "arm": arm["id"],
                            "language": language,
                            "prompt_id": prompt_id,
                            "selected_source_index": source_index,
                            "gold": row["category"].strip(),
                            **scored,
                        }
                    )
        integrity = validate_prediction_integrity(
            predictions,
            arm=arm,
            validation_rows=validation_rows,
            template_ids=list(templates),
            labels=labels,
            limit=limit,
        )
        arm_output = output_root / arm["id"]
        arm_output.mkdir()
        predictions_path = arm_output / "predictions.jsonl"
        predictions_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in predictions
            ),
            encoding="utf-8",
        )
        summary = {
            "arm": arm["id"],
            "adapter_repo": arm["adapter_repo"],
            "adapter_revision": arm["adapter_revision"],
            "limit_per_language_per_prompt": limit,
            "score_mode": "mean_continuation_token_log_probability",
            "maximum_input_tokens": protocol["runtime"]["maximum_input_tokens"],
            "integrity": integrity,
            **_summarize_predictions(predictions, labels, evaluator),
        }
        summary_path = arm_output / "summary.json"
        _write_json(summary_path, summary)
        arm_manifests.append(
            {
                "arm": arm["id"],
                "predictions": {
                    "path": str(predictions_path),
                    "sha256": file_sha256(predictions_path),
                },
                "summary": {
                    "path": str(summary_path),
                    "sha256": file_sha256(summary_path),
                },
            }
        )
        del evaluator, tokenizer, model
        torch.cuda.empty_cache()

    manifest = {
        "schema": "sallm.mamba_news_common_validation_result/v1",
        "protocol": {
            "path": str(protocol_path),
            "sha256": file_sha256(protocol_path),
        },
        "data_boundary": "validation_only",
        "test_accessed": False,
        "diagnostic_only": True,
        "limit_per_language_per_prompt": limit,
        "gates": {
            "binding": "passed",
            "completeness": "passed",
            "numerical_finiteness": "passed",
            "metric_recomputation": "passed",
            "prediction_distribution": "diagnostic_only_not_an_admission_gate",
        },
        "arms": arm_manifests,
    }
    _write_json(output_root / "manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(".audit/mamba_news_common_validation_protocol_20260914.json"),
    )
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Rows per language per prompt; 0 evaluates the complete validation split."
        ),
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    protocol_path = args.protocol
    if not protocol_path.is_absolute():
        protocol_path = repo / protocol_path
    protocol = load_and_check_protocol(repo, protocol_path)
    if args.check:
        print(PROTOCOL_OK)
        return 0
    if args.asset_root is None or args.output_root is None:
        raise SystemExit(
            "--asset-root and --output-root are required unless --check is used"
        )
    limit = (
        protocol["runtime"]["canary_limit_per_language_per_prompt"]
        if args.limit is None
        else args.limit
    )
    if limit < 0:
        raise SystemExit("--limit must be zero or positive")
    run_evaluation(
        repo,
        protocol_path,
        protocol,
        args.asset_root.resolve(),
        args.output_root.resolve(),
        limit,
    )
    print(f"MAMBA_NEWS_COMMON_VALIDATION_COMPLETE {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
