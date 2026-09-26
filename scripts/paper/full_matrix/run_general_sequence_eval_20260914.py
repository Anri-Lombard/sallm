#!/usr/bin/env python3
"""Run one hash-bound General NER or POS validation/test work unit."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from lm_eval import evaluator
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from lm_eval.utils import load_yaml_config
from sallm.config import ModelEvalConfig
from sallm.data.formatters.base import safe_format_prompt
from sallm.data.loaders.huggingface import _load_masakhapos_split
from sallm.evaluation.constrained_label_scoring import (
    append_text,
    chat_messages_prefix,
    continuation_ids,
    score_labels,
)
from sallm.evaluation.harness import load_model_and_tokenizer
from sallm.evaluation.lm_eval_runner import (
    _format_model_args,
    _materialize_model_for_lm_eval,
    _prepare_include_paths,
    _prepare_tokenizer_for_lm_eval,
    _resolve_ephemeral_eval_root,
    _to_serializable,
)
from sallm.evaluation.pos_metrics import UPOS_LABELS
from sallm.templates import registry as templates
from transformers import AutoTokenizer

ARCHITECTURES = ("mzansilm", "mamba2", "xlstm", "gdn")
LANGUAGES = ("tsn", "xho", "zul")
NER_CODES = {"tsn": "tn", "xho": "xh", "zul": "zu"}
NER_REVISION = "6aa65cdbfa22d66e5b4ed176ac525c364cda08d1"
NER_DATASET = "anrilombard/masakhaner-x-parquet"
POS_TEMPLATES = tuple(f"masakhane_pos_tagging/lm_eval_p{i}" for i in range(1, 5))
MAXIMUM_INPUT_TOKENS = 1024
LOGGER = logging.getLogger("general_sequence_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("ner", "pos"))
    parser.add_argument("--phase", choices=("validation", "test"))
    parser.add_argument("--architecture", choices=ARCHITECTURES)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    temporary.replace(path)


def verify_binding(
    protocol: dict[str, Any],
    architecture: str,
    checkpoint: Path,
    adapter: Path | None,
) -> dict[str, Any]:
    binding = protocol["models"][architecture]
    if str(binding.get("execution_status", "runnable")).startswith("blocked"):
        raise ValueError(f"{architecture} binding is blocked by the protocol")
    expected_checkpoint = Path(binding["base_path"])
    expected_adapter = (
        Path(binding["adapter_path"])
        if binding.get("adapter_path") is not None
        else None
    )
    if checkpoint.absolute() != expected_checkpoint.absolute():
        raise ValueError(f"Unexpected {architecture} base path: {checkpoint}")
    if (adapter is None) != (expected_adapter is None):
        raise ValueError(f"Unexpected {architecture} adapter presence")
    if adapter is not None and expected_adapter is not None and adapter.absolute() != expected_adapter.absolute():
        raise ValueError(f"Unexpected {architecture} adapter path: {adapter}")

    checked: dict[str, dict[str, str]] = {"base": {}, "adapter": {}}
    for group, root, field in (
        ("base", checkpoint, "base_files"),
        ("adapter", adapter, "adapter_files"),
    ):
        if root is None:
            if binding.get(field):
                raise ValueError(f"Unexpected {architecture}/{group} files")
            continue
        for relative, expected in binding[field].items():
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            observed = sha256(path)
            if observed != expected:
                raise ValueError(
                    f"{architecture}/{group}/{relative} hash mismatch: {observed}"
                )
            checked[group][relative] = observed
    return {
        "base_path": str(checkpoint.absolute()),
        "adapter_path": str(adapter.absolute()) if adapter is not None else None,
        "base_tree_sha256": binding["base_tree_sha256"],
        "adapter_tree_sha256": binding["adapter_tree_sha256"],
        "files": checked,
    }


def verify_phase_gate(
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str | None]:
    if args.phase == "validation":
        if args.selection is not None or args.release is not None:
            raise ValueError("Validation must not read a selection or test release")
        return None, None
    if args.selection is None or args.release is None or args.task is None:
        raise ValueError("Test requires selection and task-specific release")
    selection = json.loads(args.selection.read_text())
    selection_hash = sha256(args.selection)
    release = json.loads(args.release.read_text())
    expected = {
        "schema": "sallm.general_sequence_test_release/v1",
        "task": args.task,
        "selection_sha256": selection_hash,
        "authorized": True,
        "no_score_based_retry": True,
    }
    if any(release.get(key) != value for key, value in expected.items()):
        raise ValueError("Test release does not match the frozen selection and task")
    return selection, selection_hash


def chosen_prompts(
    task: str,
    phase: str,
    selection: dict[str, Any] | None,
) -> dict[str, list[int]]:
    if phase == "validation":
        count = 5 if task == "ner" else 4
        return {language: list(range(1, count + 1)) for language in LANGUAGES}
    assert selection is not None
    selected = selection["selected_prompts"][task]
    if set(selected) != set(LANGUAGES):
        raise ValueError(f"Incomplete {task} prompt selection")
    output = {language: [int(selected[language])] for language in LANGUAGES}
    maximum = 5 if task == "ner" else 4
    if any(not 1 <= prompts[0] <= maximum for prompts in output.values()):
        raise ValueError(f"Invalid selected {task} prompt")
    return output


def ner_task_name(language: str, prompt: int, phase: str) -> str:
    suffix = "val" if phase == "validation" else "test"
    return f"sallm_masakhaner_{NER_CODES[language]}_prompt_{prompt}_{suffix}"


def load_ner_split(
    *,
    language: str,
    split: str,
    **_: Any,
) -> DatasetDict:
    code = NER_CODES[language]
    dataset = load_dataset(
        NER_DATASET,
        data_files={split: f"data/{code}/{split}.parquet"},
        revision=NER_REVISION,
        split=split,
    )
    return DatasetDict({split: dataset})


def build_ner_tasks(
    source_root: Path,
    phase: str,
    prompts: dict[str, list[int]],
) -> tuple[TaskManager, list[ConfigurableTask], dict[str, Any]]:
    directory = source_root / "src/conf/eval/lm_eval_tasks" / (
        "masakhaner_validation" if phase == "validation" else "masakhaner_test"
    )
    manager = TaskManager(
        include_path=_prepare_include_paths([str(directory)]),
        include_defaults=False,
    )
    tasks: list[ConfigurableTask] = []
    evidence: dict[str, Any] = {}
    split = "validation" if phase == "validation" else "test"
    for language in LANGUAGES:
        for prompt in prompts[language]:
            name = ner_task_name(language, prompt, phase)
            yaml_path = Path(manager.task_index[name]["yaml_path"])
            config = load_yaml_config(yaml_path=yaml_path)
            config["task"] = name
            config["custom_dataset"] = partial(
                load_ner_split,
                language=language,
                split=split,
            )
            config["fewshot_split"] = None
            if phase == "validation":
                config["validation_split"] = "validation"
                config.pop("test_split", None)
            else:
                config["test_split"] = "test"
                config["validation_split"] = None
            task = ConfigurableTask(config=config)
            rows = len(task.eval_docs)
            tasks.append(task)
            evidence[name] = {
                "language": language,
                "prompt": prompt,
                "split": split,
                "rows": rows,
                "fingerprint": task.dataset[split]._fingerprint,
                "dataset_revision": NER_REVISION,
                "yaml_sha256": sha256(yaml_path),
            }
    return manager, tasks, evidence


def unwrap_singleton(value: Any) -> Any:
    while isinstance(value, list | tuple):
        if len(value) != 1:
            raise ValueError(f"Expected singleton response, found {len(value)}")
        value = value[0]
    return value


def compact_ner_rows(
    call: dict[str, Any], tokenizer_path: str, maximum_input_tokens: int
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, local_files_only=True
    )
    rows: list[dict[str, Any]] = []
    for task_name, samples in sorted(call["samples"].items()):
        for sample in samples:
            if sample.get("filter") != "flexible-extract":
                continue
            prompt_text = str(sample["arguments"][0][0])
            raw_tokens = len(
                tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            )
            prediction = str(unwrap_singleton(sample["filtered_resps"]))
            raw_response = str(unwrap_singleton(sample["resps"]))
            rows.append(
                {
                    "task": task_name,
                    "doc_id": str(sample["doc_id"]),
                    "doc_hash": str(sample["doc_hash"]),
                    "prompt_hash": str(sample["prompt_hash"]),
                    "target_hash": str(sample["target_hash"]),
                    "target": sample["target"],
                    "prediction": prediction,
                    "raw_response": raw_response,
                    "input_token_count_before_cap": raw_tokens,
                    "input_token_count": min(raw_tokens, maximum_input_tokens),
                    "input_truncated": raw_tokens > maximum_input_tokens,
                }
            )
    return rows


def run_ner(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    binding: dict[str, Any],
    selection: dict[str, Any] | None,
    selection_hash: str | None,
) -> dict[str, Any]:
    assert args.architecture and args.checkpoint and args.phase
    prompts = chosen_prompts("ner", args.phase, selection)
    source_root = Path(protocol["source_snapshot"]["path"])
    manager, tasks, evidence = build_ner_tasks(source_root, args.phase, prompts)
    settings = protocol["models"][args.architecture]
    tie_word_embeddings = settings["tie_word_embeddings"]
    if args.architecture == "mamba2":
        tie_word_embeddings = False
    model_config = ModelEvalConfig(
        checkpoint=str(args.checkpoint),
        peft_adapter=str(args.adapter) if args.adapter is not None else None,
        merge_lora=bool(settings["merge_lora"]),
        tie_word_embeddings=tie_word_embeddings,
        dtype=str(settings["dtype"]),
        device="cuda:0",
        lm_eval_model_args={"max_length": MAXIMUM_INPUT_TOKENS},
    )
    temporary_parent = _resolve_ephemeral_eval_root()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=f"general_ner_{args.architecture}_{args.phase}_",
        dir=temporary_parent,
    ) as temporary:
        work_root = Path(temporary)
        pretrained, peft_adapter = _materialize_model_for_lm_eval(
            model_config, work_root / "model"
        )
        tokenizer_path = _prepare_tokenizer_for_lm_eval(
            pretrained, work_root / "tokenizer", True
        )
        if tokenizer_path is None:
            raise RuntimeError("Unable to materialize the NER tokenizer")
        model_args = _format_model_args(
            pretrained_path=pretrained,
            dtype=model_config.dtype,
            peft_adapter=peft_adapter,
            tokenizer_override=tokenizer_path,
            tie_word_embeddings=tie_word_embeddings,
            extra_model_args={"max_length": MAXIMUM_INPUT_TOKENS},
            default_add_bos_token=False,
        )
        result = evaluator.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=tasks,
            num_fewshot=0,
            batch_size="auto:4",
            max_batch_size=16,
            device="cuda:0",
            limit=args.limit,
            bootstrap_iters=0,
            write_out=True,
            log_samples=True,
            apply_chat_template=True,
            task_manager=manager,
            random_seed=42,
            numpy_random_seed=42,
            torch_random_seed=42,
            fewshot_random_seed=42,
        )
        if result is None:
            raise RuntimeError("lm-eval returned no NER result")
        call = _to_serializable(result)
        rows = compact_ner_rows(call, tokenizer_path, MAXIMUM_INPUT_TOKENS)
        metrics = {
            name: float(values["f1,flexible-extract"])
            for name, values in call["results"].items()
        }
    shutil.rmtree(work_root, ignore_errors=True)
    return {
        "schema": "sallm.general_sequence_eval/v1",
        "task": "ner",
        "phase": args.phase,
        "architecture": args.architecture,
        "test_accessed": args.phase == "test",
        "selection_sha256": selection_hash,
        "protocol_sha256": sha256(args.protocol),
        "maximum_input_tokens": MAXIMUM_INPUT_TOKENS,
        "deterministic": True,
        "seed": 42,
        "limit_per_cell": args.limit,
        "binding": binding,
        "task_evidence": evidence,
        "reported_metrics": metrics,
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
    }


def tokens_repr(tokens: list[str]) -> str:
    return "[" + ", ".join(repr(token) for token in tokens) + "]"


def pos_prompt(template_id: str, tokens: list[str]) -> str:
    spec = templates.get(template_id)
    return safe_format_prompt(spec.prompt, {"tokens": tokens_repr(tokens)})


def decode_pos_row(
    model: Any,
    tokenizer: Any,
    tokens: list[str],
    prompt: str,
    device: torch.device,
) -> tuple[list[str], list[float], int, int, int, int, int | None, int | None]:
    context_text, context_ids = chat_messages_prefix(
        tokenizer, [{"role": "user", "content": prompt}]
    )
    initial_tokens = len(context_ids)
    maximum_forward_tokens = min(initial_tokens, MAXIMUM_INPUT_TOKENS)
    maximum_untruncated_forward_tokens = initial_tokens
    maximum_prefix_tokens_removed = 0
    first_prefix_truncation_token: int | None = None
    first_prefix_truncation_forward_tokens: int | None = None
    predictions: list[str] = []
    selected_scores: list[float] = []
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("POS scoring requires a pad or EOS token")

    for index, token in enumerate(tokens):
        prefix = ("[" if index == 0 else ", ") + f"({token!r}, '"
        context_text, context_ids = append_text(
            tokenizer, context_text, context_ids, prefix
        )
        label_ids = {
            label: continuation_ids(tokenizer, context_text, context_ids, label)
            for label in UPOS_LABELS
        }
        untruncated_forward_tokens = max(
            len(context_ids) + len(ids) for ids in label_ids.values()
        )
        maximum_untruncated_forward_tokens = max(
            maximum_untruncated_forward_tokens, untruncated_forward_tokens
        )
        maximum_label_tokens = max(len(ids) for ids in label_ids.values())
        maximum_context_tokens = MAXIMUM_INPUT_TOKENS - maximum_label_tokens
        if maximum_context_tokens <= 0:
            raise ValueError(
                "POS label continuation is too long for the 1024-token cap"
            )
        prefix_tokens_removed = max(0, len(context_ids) - maximum_context_tokens)
        scoring_context_ids = context_ids[prefix_tokens_removed:]
        forward_tokens = max(
            len(scoring_context_ids) + len(ids) for ids in label_ids.values()
        )
        maximum_forward_tokens = max(maximum_forward_tokens, forward_tokens)
        maximum_prefix_tokens_removed = max(
            maximum_prefix_tokens_removed, prefix_tokens_removed
        )
        if prefix_tokens_removed > 0 and first_prefix_truncation_token is None:
            first_prefix_truncation_token = index
            first_prefix_truncation_forward_tokens = untruncated_forward_tokens
        if forward_tokens > MAXIMUM_INPUT_TOKENS:
            raise AssertionError("bounded POS context exceeds the 1024-token cap")
        label, score, _ = score_labels(
            model=model,
            context_ids=scoring_context_ids,
            label_ids=label_ids,
            labels=list(UPOS_LABELS),
            score_mode="mean",
            pad_token_id=int(pad_token_id),
            pad_to_multiple_of=64,
            device=device,
        )
        predictions.append(label)
        selected_scores.append(score)
        suffix = label + "')"
        if index == len(tokens) - 1:
            suffix += "]"
        context_text, context_ids = append_text(
            tokenizer, context_text, context_ids, suffix
        )
    return (
        predictions,
        selected_scores,
        initial_tokens,
        maximum_forward_tokens,
        maximum_untruncated_forward_tokens,
        maximum_prefix_tokens_removed,
        first_prefix_truncation_token,
        first_prefix_truncation_forward_tokens,
    )


def dataset_digest(dataset: Dataset) -> str:
    rows = [
        {
            "id": str(row.get("id", index)),
            "tokens": [str(value) for value in row["tokens"]],
            "gold": [str(value).upper() for value in row["upos"]],
        }
        for index, row in enumerate(dataset)
    ]
    return canonical_sha256(rows)


def run_pos(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    binding: dict[str, Any],
    selection: dict[str, Any] | None,
    selection_hash: str | None,
) -> dict[str, Any]:
    assert args.architecture and args.checkpoint and args.phase
    prompts = chosen_prompts("pos", args.phase, selection)
    split = "validation" if args.phase == "validation" else "test"
    datasets = {
        language: _load_masakhapos_split(language, split)
        for language in LANGUAGES
    }
    evidence = {
        language: {
            "split": split,
            "rows": len(dataset),
            "tokens": sum(len(tokens) for tokens in dataset["tokens"]),
            "dataset_revision": protocol["tasks"]["pos"]["revision"],
            "rows_sha256": dataset_digest(dataset),
        }
        for language, dataset in datasets.items()
    }
    settings = protocol["models"][args.architecture]
    model_config = ModelEvalConfig(
        checkpoint=str(args.checkpoint),
        peft_adapter=str(args.adapter) if args.adapter is not None else None,
        merge_lora=bool(settings["merge_lora"]),
        tie_word_embeddings=settings["tie_word_embeddings"],
        dtype=str(settings["dtype"]),
        device="cuda:0",
    )
    started = time.monotonic()
    model, tokenizer = load_model_and_tokenizer(model_config)
    model.eval()
    device = torch.device("cuda:0")
    rows: list[dict[str, Any]] = []
    cell_counts: dict[str, dict[str, int | float]] = {}
    for language in LANGUAGES:
        dataset = datasets[language]
        if args.limit is not None:
            dataset = dataset.select(range(min(args.limit, len(dataset))))
        for prompt_number in prompts[language]:
            template_id = POS_TEMPLATES[prompt_number - 1]
            correct = 0
            total = 0
            for index, source in enumerate(dataset):
                tokens = [str(token) for token in source["tokens"]]
                gold = [str(tag).upper() for tag in source["upos"]]
                if len(tokens) != len(gold) or not tokens:
                    raise ValueError(f"Invalid POS row {language}/{index}")
                prompt = pos_prompt(template_id, tokens)
                (
                    prediction,
                    scores,
                    initial_tokens,
                    max_forward_tokens,
                    max_untruncated_forward_tokens,
                    max_prefix_tokens_removed,
                    first_prefix_truncation_token,
                    first_prefix_truncation_forward_tokens,
                ) = decode_pos_row(model, tokenizer, tokens, prompt, device)
                if len(prediction) != len(gold) or any(
                    label not in UPOS_LABELS for label in prediction
                ):
                    raise ValueError(f"Invalid POS prediction {language}/{index}")
                mask = [
                    predicted == expected
                    for predicted, expected in zip(prediction, gold, strict=True)
                ]
                correct += sum(mask)
                total += len(gold)
                source_id = str(source.get("id", index))
                source_hash = canonical_sha256(
                    {"id": source_id, "tokens": tokens, "gold": gold}
                )
                rows.append(
                    {
                        "language": language,
                        "prompt": prompt_number,
                        "template_id": template_id,
                        "id": source_id,
                        "source_sha256": source_hash,
                        "prompt_sha256": text_sha256(prompt),
                        "tokens": tokens,
                        "gold": gold,
                        "prediction": prediction,
                        "correct": mask,
                        "selected_mean_logprobs": scores,
                        "input_token_count": initial_tokens,
                        "maximum_forward_tokens": max_forward_tokens,
                        "maximum_untruncated_forward_tokens": (
                            max_untruncated_forward_tokens
                        ),
                        "maximum_prefix_tokens_removed": max_prefix_tokens_removed,
                        "first_prefix_truncation_token": (
                            first_prefix_truncation_token
                        ),
                        "first_prefix_truncation_forward_tokens": (
                            first_prefix_truncation_forward_tokens
                        ),
                    }
                )
                if (index + 1) % 25 == 0:
                    LOGGER.info(
                        "POS %s/%s/P%d: %d/%d rows",
                        args.architecture,
                        language,
                        prompt_number,
                        index + 1,
                        len(dataset),
                    )
            if total <= 0:
                raise ValueError(f"Empty POS cell {language}/P{prompt_number}")
            cell_counts[f"{language}/P{prompt_number}"] = {
                "correct": correct,
                "total": total,
                "token_accuracy": correct / total,
            }
    return {
        "schema": "sallm.general_sequence_eval/v1",
        "task": "pos",
        "phase": args.phase,
        "architecture": args.architecture,
        "test_accessed": args.phase == "test",
        "selection_sha256": selection_hash,
        "protocol_sha256": sha256(args.protocol),
        "maximum_input_tokens": MAXIMUM_INPUT_TOKENS,
        "deterministic": True,
        "seed": 42,
        "limit_per_cell": args.limit,
        "binding": binding,
        "task_evidence": evidence,
        "interface": "closed_label_tuple_mean_logprob_v1",
        "label_set": list(UPOS_LABELS),
        "score_mode": "mean",
        "pad_to_multiple_of": 64,
        "reported_metrics": cell_counts,
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
    }


def self_check() -> None:
    class FakeTokenizer:
        chat_template = "fake"
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, text: str, **_: Any) -> dict[str, list[int]]:
            return {"input_ids": [ord(character) % 255 + 1 for character in text]}

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str | list[int]:
            assert add_generation_prompt
            text = "U:" + messages[0]["content"] + "\nA:"
            return self(text)["input_ids"] if tokenize else text

    class FakeModel:
        def __call__(self, input_ids: torch.Tensor, **_: Any) -> Any:
            return type(
                "Output",
                (),
                {"logits": torch.zeros((*input_ids.shape, 256))},
            )()

    assert ner_task_name("tsn", 1, "validation") == (
        "sallm_masakhaner_tn_prompt_1_val"
    )
    assert ner_task_name("zul", 5, "test") == (
        "sallm_masakhaner_zu_prompt_5_test"
    )
    assert unwrap_singleton([[["x"]]]) == "x"
    assert tokens_repr(["a", "b"]) == "['a', 'b']"
    assert len(UPOS_LABELS) == 17
    assert chosen_prompts("ner", "validation", None)["xho"] == [1, 2, 3, 4, 5]
    (
        prediction,
        scores,
        initial,
        maximum,
        untruncated,
        removed,
        first_truncation,
        first_truncation_forward,
    ) = decode_pos_row(
        FakeModel(), FakeTokenizer(), ["one", "two"], "tag", torch.device("cpu")
    )
    assert prediction == ["ADJ", "ADJ"]
    assert len(scores) == 2 and all(math.isfinite(score) for score in scores)
    assert 0 < initial <= maximum <= MAXIMUM_INPUT_TOKENS
    assert maximum == untruncated and removed == 0
    assert first_truncation is None and first_truncation_forward is None

    (
        long_prediction,
        _,
        _,
        long_maximum,
        long_untruncated,
        long_removed,
        long_first_truncation,
        long_first_truncation_forward,
    ) = decode_pos_row(
        FakeModel(),
        FakeTokenizer(),
        ["x" * 900, "y" * 900],
        "tag",
        torch.device("cpu"),
    )
    assert len(long_prediction) == 2
    assert long_maximum <= MAXIMUM_INPUT_TOKENS
    assert long_untruncated > MAXIMUM_INPUT_TOKENS
    assert long_removed > 0
    assert long_first_truncation is not None
    assert long_first_truncation_forward is not None
    print("SELF_CHECK_OK")


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    required = (
        args.task,
        args.phase,
        args.architecture,
        args.checkpoint,
        args.protocol,
        args.output,
    )
    if any(value is None for value in required):
        raise ValueError(
            "task, phase, architecture, paths, protocol and output required"
        )
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("schema") != "sallm.general_sequence_replacement_protocol/v1":
        raise ValueError("Unexpected sequence replacement protocol")
    source = Path(protocol["source_snapshot"]["path"])
    manifest = source / "SNAPSHOT_MANIFEST.sha256"
    if sha256(manifest) != protocol["source_snapshot"]["manifest_sha256"]:
        raise ValueError("Source snapshot manifest changed")
    selection, selection_hash = verify_phase_gate(args)
    binding = verify_binding(
        protocol, args.architecture, args.checkpoint, args.adapter
    )
    if args.task == "ner":
        payload = run_ner(args, protocol, binding, selection, selection_hash)
    else:
        payload = run_pos(args, protocol, binding, selection, selection_hash)
    if not math.isfinite(float(payload["elapsed_seconds"])):
        raise ValueError("Non-finite elapsed time")
    atomic_json(args.output, payload)
    print(f"GENERAL_SEQUENCE_COMPLETE {args.output}")


if __name__ == "__main__":
    main()
