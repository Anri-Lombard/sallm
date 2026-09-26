#!/usr/bin/env python3
"""Run one frozen validation arm or one selected-prompt official unit.

Rollout copy (fft_rollout) of full_matrix/run_lm_eval_unit.py. Only change: LM_EVAL_BATCH (default 8) sets the batch size;
xLSTM uses 1, as sequence/xlstm_bs1/run_lm_eval_unit_bs1.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from sallm.config import ModelEvalConfig
from sallm.evaluation.lm_eval_runner import (
    _format_model_args,
    _materialize_model_for_lm_eval,
    _prepare_include_paths,
    _prepare_tokenizer_for_lm_eval,
    _resolve_ephemeral_eval_root,
    _to_serializable,
)


NER_CODES = {"tsn": "tn", "xho": "xh", "zul": "zu"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def ref_id(reference: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(reference).encode()).hexdigest()


def interface(architecture: str) -> dict[str, Any]:
    return {
        "mzansilm": {"dtype": "bfloat16", "merge_lora": False, "tie_word_embeddings": None},
        "mamba2": {"dtype": "float32", "merge_lora": False, "tie_word_embeddings": False},
        "xlstm": {"dtype": "float32", "merge_lora": True, "tie_word_embeddings": False},
        "gdn": {"dtype": "bfloat16", "merge_lora": False, "tie_word_embeddings": None},
    }[architecture]


def resolved(entries: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    value = entries[ref_id(reference)]
    path = value.get("path")
    if path is not None and not Path(path).is_dir():
        raise FileNotFoundError(path)
    return value


def validation_tasks(arm: dict[str, Any]) -> tuple[list[str], bool, list[str]]:
    if arm["task_group"] == "ner":
        tasks = [
            f"sallm_masakhaner_{NER_CODES[language]}_prompt_{prompt}_val"
            for language in arm["languages"]
            for prompt in range(1, 6)
        ]
        return tasks, True, ["src/conf/eval/lm_eval_tasks/masakhaner_validation"]
    if arm["task_group"] == "sib":
        tasks = [
            f"sallm_sib_{language}_val_prompt_{prompt}"
            for language in arm["languages"]
            for prompt in range(1, 6)
        ]
        return tasks, False, ["src/conf/eval/lm_eval_tasks/sib_validation"]
    raise ValueError(f"Unsupported ambiguity task: {arm['task_group']}")


def official_groups(unit: dict[str, Any]) -> list[tuple[str, list[str], bool, list[str]]]:
    groups: dict[str, list[str]] = {"chat": [], "raw": []}
    for task in unit["task_names"]:
        if task.startswith("sallm_masakhanews_") or task.startswith("injongointent_") or task.startswith("belebele_"):
            groups["chat"].append(task)
        else:
            groups["raw"].append(task)
    output = []
    if groups["chat"]:
        output.append(
            (
                "chat",
                groups["chat"],
                True,
                ["src/conf/eval/lm_eval_tasks/masakhanews_test"],
            )
        )
    if groups["raw"]:
        output.append(("raw", groups["raw"], False, []))
    return output


def run_call(
    *,
    tasks: list[str],
    apply_chat_template: bool,
    include_paths: list[str],
    source: Path,
    pretrained: str,
    peft_adapter: str | None,
    model_config: ModelEvalConfig,
    work_root: Path,
) -> dict[str, Any]:
    manager = TaskManager(
        include_path=(
            _prepare_include_paths([str(source / path) for path in include_paths])
            if include_paths
            else None
        ),
        include_defaults=True,
    )
    missing = [task for task in tasks if task not in manager.task_index]
    if missing:
        raise ValueError(f"Missing frozen tasks: {missing}")
    tokenizer = _prepare_tokenizer_for_lm_eval(
        pretrained, work_root / ("tokenizer_chat" if apply_chat_template else "tokenizer_raw"), apply_chat_template
    )
    model_args = _format_model_args(
        pretrained_path=pretrained,
        dtype=model_config.dtype,
        peft_adapter=peft_adapter,
        tokenizer_override=tokenizer,
        tie_word_embeddings=model_config.tie_word_embeddings,
        extra_model_args={"max_length": 1024},
        default_add_bos_token=not apply_chat_template,
    )
    result = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=0,
        batch_size=int(os.environ.get("LM_EVAL_BATCH", "8")),
        max_batch_size=int(os.environ.get("LM_EVAL_BATCH", "8")),
        device="cuda:0",
        limit=int(os.environ["LM_EVAL_LIMIT"]) if os.environ.get("LM_EVAL_LIMIT") else None,
        bootstrap_iters=0,
        log_samples=True,
        write_out=True,
        apply_chat_template=apply_chat_template,
        task_manager=manager,
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        fewshot_random_seed=42,
    )
    if result is None:
        raise RuntimeError("lm-eval returned no result")
    return _to_serializable(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validation", "official"), required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".tmp").exists():
        raise FileExistsError(args.output)
    entries = json.loads(args.bindings.read_text())["entries"]
    if args.mode == "validation":
        item = json.loads((args.inventory / "validation_arms.json").read_text())[args.index]
        base_ref, adapter_ref = item["base"], item["adapter"]
        groups = [("validation", *validation_tasks(item))]
    else:
        if args.release is None:
            raise ValueError("Official mode requires a release")
        release = json.loads(args.release.read_text())
        if not release.get("authorized") or not release.get("no_score_based_retry"):
            raise ValueError("Official release is not valid")
        item = json.loads((args.inventory / "official_units.json").read_text())[args.index]
        if item["task_group"] in {"ner", "pos", "generation"}:
            raise ValueError("This runner only accepts prompt units")
        base_ref = item["binding"]["base"]
        candidates = item["binding"]["adapter_candidates"]
        selection = release["selected_bindings"].get(item["unit_id"])
        adapter_ref = candidates[int(selection)] if selection is not None else candidates[0]
        groups = official_groups(item)
    base = resolved(entries, base_ref)
    adapter = resolved(entries, adapter_ref)
    model_settings = interface(item["architecture"])
    model_config = ModelEvalConfig(
        checkpoint=base["path"],
        peft_adapter=adapter["path"],
        merge_lora=model_settings["merge_lora"],
        tie_word_embeddings=model_settings["tie_word_embeddings"],
        dtype=model_settings["dtype"],
        device="cuda:0",
        lm_eval_model_args={"max_length": 1024},
    )
    temporary_parent = _resolve_ephemeral_eval_root()
    with tempfile.TemporaryDirectory(prefix="full_matrix_lm_eval_", dir=temporary_parent) as temporary:
        work_root = Path(temporary)
        pretrained, peft_adapter = _materialize_model_for_lm_eval(model_config, work_root / "model")
        calls = {}
        for name, tasks, apply_chat, includes in groups:
            calls[name] = run_call(
                tasks=tasks,
                apply_chat_template=apply_chat,
                include_paths=includes,
                source=args.source,
                pretrained=pretrained,
                peft_adapter=peft_adapter,
                model_config=model_config,
                work_root=work_root,
            )
    shutil.rmtree(work_root, ignore_errors=True)
    payload = {
        "schema": "sallm.full_matrix_lm_eval_unit/v1",
        "mode": args.mode,
        "index": args.index,
        "item": item,
        "base": base,
        "adapter": adapter,
        "maximum_input_tokens": 1024,
        "fewshot": 0,
        "batch_size": int(os.environ.get("LM_EVAL_BATCH", "8")),
        "calls": calls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(f"FULL_MATRIX_{args.mode.upper()}_UNIT_OK index={args.index} sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
