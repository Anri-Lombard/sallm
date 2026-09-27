#!/usr/bin/env python3
"""Run the validation or frozen-test lm-eval block for one General adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset
from lm_eval import evaluator
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from lm_eval.utils import load_yaml_config
from sallm.config import ModelEvalConfig
from sallm.evaluation.lm_eval_runner import (
    _format_model_args,
    _materialize_model_for_lm_eval,
    _prepare_tokenizer_for_lm_eval,
    _resolve_ephemeral_eval_root,
    _to_serializable,
)

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
PROMPTS = range(1, 6)
VALIDATION_ROWS = {"afrixnli": 450, "afrimmlu": 83}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validation", "test"))
    parser.add_argument("--architecture", choices=ARCHITECTURES)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--tie-word-embeddings", choices=("true", "false"))
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--batch-size",
        type=parse_batch_size,
        default="auto:4",
        help="lm-eval batch size; pass a positive integer to disable auto probing",
    )
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def parse_batch_size(value: str) -> int | str:
    if value.startswith("auto"):
        return value
    size = int(value)
    if size <= 0:
        raise argparse.ArgumentTypeError("batch size must be positive")
    return size


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_name(family: str, language: str, prompt: int) -> str:
    if family == "afrixnli":
        return f"afrixnli_{language}_prompt_{prompt}"
    if family == "afrimmlu":
        return f"afrimmlu_direct_{language}_prompt_{prompt}"
    raise ValueError(f"Unsupported validation family {family!r}")


def register_validation_alias(
    manager: TaskManager,
    original_name: str,
    validation_name: str,
) -> None:
    """Register an in-memory validation task name for lm-eval's task logger."""
    original = manager.task_index[original_name]
    existing = manager.task_index.get(validation_name)
    if existing is not None and existing != original:
        raise ValueError(f"Conflicting validation task alias {validation_name}")
    manager.task_index[validation_name] = dict(original)


def load_validation_only(
    dataset_path: str,
    dataset_name: str,
    **_: Any,
) -> DatasetDict:
    dataset = load_dataset(
        dataset_path,
        dataset_name,
        split="validation",
    )
    return DatasetDict({"validation": dataset})


def validation_tasks(
    manager: TaskManager,
) -> tuple[list[ConfigurableTask], dict[str, Any]]:
    tasks = []
    evidence: dict[str, Any] = {}
    for family in VALIDATION_ROWS:
        for language in LANGUAGES:
            for prompt in PROMPTS:
                original_name = task_name(family, language, prompt)
                entry = manager.task_index[original_name]
                yaml_path = Path(entry["yaml_path"])
                config = load_yaml_config(yaml_path=yaml_path)
                dataset_path = str(config["dataset_path"])
                dataset_name = str(config["dataset_name"])
                validation_name = f"sallm_val_{original_name}"
                register_validation_alias(manager, original_name, validation_name)
                config["task"] = validation_name
                config["test_split"] = "validation"
                config["custom_dataset"] = partial(
                    load_validation_only,
                    dataset_path,
                    dataset_name,
                )
                task = ConfigurableTask(config=config)
                rows = len(task.eval_docs)
                expected_rows = VALIDATION_ROWS[family]
                if rows != expected_rows:
                    raise ValueError(
                        f"{validation_name} expected {expected_rows} rows, found {rows}"
                    )
                tasks.append(task)
                evidence[validation_name] = {
                    "source_task": original_name,
                    "yaml_path": str(yaml_path),
                    "yaml_sha256": file_sha256(yaml_path),
                    "dataset_path": dataset_path,
                    "dataset_name": dataset_name,
                    "split": "validation",
                    "rows": rows,
                    "fingerprint": task.dataset["validation"]._fingerprint,
                }
    return tasks, evidence


def test_task_groups(selection: dict[str, Any]) -> dict[str, list[str]]:
    chosen = selection["selected_prompts"]
    closed = []
    for family in ("afrixnli", "afrimmlu"):
        for language in LANGUAGES:
            closed.append(task_name(family, language, int(chosen[family][language])))
    closed.extend(f"afrimgsm_{language}_prompt_1" for language in LANGUAGES)
    return {
        "closed_and_generation": closed,
        "belebele": [
            f"belebele_{language}_prompt_1" for language in BELEBELE_LANGUAGES
        ],
    }


def run_call(
    *,
    tasks: list[Any],
    model_args: str,
    manager: TaskManager,
    apply_chat_template: bool,
    batch_size: int | str,
    max_batch_size: int,
    limit: int | None,
) -> dict[str, Any]:
    result = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=0,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        device="cuda:0",
        limit=limit,
        log_samples=True,
        write_out=True,
        apply_chat_template=apply_chat_template,
        task_manager=manager,
    )
    if result is None:
        raise RuntimeError("lm-eval returned no result")
    return _to_serializable(result)


def self_check() -> None:
    class FakeManager:
        task_index = {"source": {"type": "task", "yaml_path": "/tmp/source.yaml"}}

    manager = FakeManager()
    register_validation_alias(manager, "source", "sallm_val_source")
    assert manager.task_index["sallm_val_source"] == manager.task_index["source"]
    selection = {
        "selected_prompts": {
            "afrixnli": {language: 2 for language in LANGUAGES},
            "afrimmlu": {language: 4 for language in LANGUAGES},
        }
    }
    groups = test_task_groups(selection)
    assert len(groups["closed_and_generation"]) == 12
    assert len(groups["belebele"]) == 8
    assert "belebele_tso_prompt_1" in groups["belebele"]
    assert groups["closed_and_generation"][0] == "afrixnli_eng_prompt_2"
    assert groups["closed_and_generation"][4] == "afrimmlu_direct_eng_prompt_4"
    assert parse_batch_size("8") == 8
    assert parse_batch_size("auto:4") == "auto:4"


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        print("SELF_CHECK_OK")
        return
    required = (
        args.phase,
        args.architecture,
        args.checkpoint,
        args.adapter,
        args.output,
    )
    if any(value is None for value in required):
        raise ValueError(
            "phase, architecture, checkpoint, adapter and output are required"
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.phase == "test" and args.selection is None:
        raise ValueError("Test phase requires a frozen selection")
    if args.phase == "validation" and args.selection is not None:
        raise ValueError("Validation phase must not read a test selection")
    if args.max_batch_size <= 0:
        raise ValueError("--max-batch-size must be positive")

    tie_word_embeddings = None
    if args.tie_word_embeddings is not None:
        tie_word_embeddings = args.tie_word_embeddings == "true"
    model_config = ModelEvalConfig(
        checkpoint=str(args.checkpoint.resolve()),
        peft_adapter=str(args.adapter.resolve()),
        merge_lora=args.merge_lora,
        tie_word_embeddings=tie_word_embeddings,
        dtype=args.dtype,
        device="cuda:0",
        lm_eval_model_args={"max_length": 1024},
    )
    manager = TaskManager()
    temp_parent = _resolve_ephemeral_eval_root()
    with tempfile.TemporaryDirectory(
        prefix=f"general_prompt_{args.architecture}_{args.phase}_",
        dir=temp_parent,
    ) as temp_dir:
        work_root = Path(temp_dir)
        pretrained, peft_adapter = _materialize_model_for_lm_eval(
            model_config,
            work_root / "model",
        )
        calls: dict[str, Any] = {}
        task_evidence: dict[str, Any] = {}
        if args.phase == "validation":
            tasks, task_evidence = validation_tasks(manager)
            tokenizer = _prepare_tokenizer_for_lm_eval(
                pretrained,
                work_root / "tokenizer_raw",
                False,
            )
            model_args = _format_model_args(
                pretrained_path=pretrained,
                dtype=model_config.dtype,
                peft_adapter=peft_adapter,
                tokenizer_override=tokenizer,
                tie_word_embeddings=tie_word_embeddings,
                extra_model_args={"max_length": 1024},
                default_add_bos_token=True,
            )
            calls["validation"] = run_call(
                tasks=tasks,
                model_args=model_args,
                manager=manager,
                apply_chat_template=False,
                batch_size=args.batch_size,
                max_batch_size=args.max_batch_size,
                limit=args.limit,
            )
        else:
            selection = json.loads(args.selection.read_text())
            groups = test_task_groups(selection)
            for group, names in groups.items():
                apply_chat = group == "belebele"
                tokenizer = _prepare_tokenizer_for_lm_eval(
                    pretrained,
                    work_root / f"tokenizer_{group}",
                    apply_chat,
                )
                model_args = _format_model_args(
                    pretrained_path=pretrained,
                    dtype=model_config.dtype,
                    peft_adapter=peft_adapter,
                    tokenizer_override=tokenizer,
                    tie_word_embeddings=tie_word_embeddings,
                    extra_model_args={"max_length": 1024},
                    default_add_bos_token=not apply_chat,
                )
                calls[group] = run_call(
                    tasks=names,
                    model_args=model_args,
                    manager=manager,
                    apply_chat_template=apply_chat,
                    batch_size=args.batch_size,
                    max_batch_size=args.max_batch_size,
                    limit=args.limit,
                )
                for name in names:
                    yaml_path = Path(manager.task_index[name]["yaml_path"])
                    task_evidence[name] = {
                        "yaml_path": str(yaml_path),
                        "yaml_sha256": file_sha256(yaml_path),
                    }
        payload = {
            "schema": "sallm.general_prompt_lm_eval/v1",
            "phase": args.phase,
            "architecture": args.architecture,
            "checkpoint": str(args.checkpoint.resolve()),
            "adapter": str(args.adapter.resolve()),
            "checkpoint_config_sha256": file_sha256(
                args.checkpoint.resolve() / "config.json"
            ),
            "adapter_config_sha256": file_sha256(
                args.adapter.resolve() / "adapter_config.json"
            ),
            "model_interface": {
                "dtype": args.dtype,
                "merge_lora": args.merge_lora,
                "tie_word_embeddings": tie_word_embeddings,
            },
            "maximum_input_tokens": 1024,
            "batch_size": args.batch_size,
            "max_batch_size": args.max_batch_size,
            "limit": args.limit,
            "selection_sha256": (
                file_sha256(args.selection) if args.selection is not None else None
            ),
            "task_evidence": task_evidence,
            "calls": calls,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp_output = args.output.with_suffix(args.output.suffix + ".tmp")
        temp_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        temp_output.replace(args.output)
    shutil.rmtree(work_root, ignore_errors=True)
    print(f"GENERAL_PROMPT_LM_EVAL_COMPLETE {args.output}")


if __name__ == "__main__":
    main()
