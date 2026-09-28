#!/usr/bin/env python3
"""Run one hash-bound General NER or POS validation/test work unit.

Rollout copy (fft_rollout) of run_general_sequence_eval_20260917_v2.py (sha 9cad0519, the BOS-repair runner). Changes:
- the adapter is optional (a fully fine-tuned model is passed as --checkpoint); a protocol binding with base_path null
  accepts any checkpoint, whose tree sha256 is recorded instead of checked;
- env SEQ_LANGUAGES restricts the languages (validation and test); SEQ_VALIDATION_PROMPTS=protocol scores only the
  General protocol prompts on validation (NER tsn P2, xho P5, zul P5; POS P3); SEQ_BATCH1=1 forces lm-eval batch 1
  (xLSTM ignores the attention mask). These are the knobs of monomulti/reselect/kit/run_seq_wrapped.py.
- --task nchlt_ner / nchlt_pos (28 Sep 2026): NCHLT NER (PER/ORG/LOC/MISC) and POS (coarse NCHLT tags, candidates =
  that language's own tags) for nbl/ssw/ven/tso from the private anrilombard/nchlt-{ner,pos}-sa4 datasets. Prompt 1 on
  validation and test (fixed in advance, no prompt selection); test is gated by the rollout's own select unit, so it
  needs no General-protocol selection/release files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from lm_eval import evaluator, utils as lm_eval_utils
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from lm_eval.utils import load_yaml_config
from sallm.config import ModelEvalConfig
from sallm.data.formatters.base import safe_format_prompt
from sallm.data.formatters.ner import reconstruct_entities_from_iob
from sallm.data.loaders.huggingface import _load_masakhapos_split
from sallm.evaluation.constrained_label_scoring import (
    append_text,
    chat_messages_prefix,
    continuation_ids,
    score_labels,
)
from sallm.evaluation.harness import load_model_and_tokenizer

import pos_cached  # noqa: E402 - runners/ is on sys.path when seq_eval runs
from sallm.evaluation.lm_eval_runner import (
    _format_model_args,
    _materialize_model_for_lm_eval,
    _prepare_include_paths,
    _prepare_tokenizer_for_lm_eval,
    _resolve_ephemeral_eval_root,
    _to_serializable,
)
from sallm.evaluation import pos_metrics
from sallm.evaluation.pos_metrics import UPOS_LABELS
from sallm.templates import registry as templates
from transformers import AutoTokenizer

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from val_subsample import indices as val_subsample_indices  # noqa: E402

ARCHITECTURES = ("mzansilm", "mamba2", "xlstm", "gdn")
LANGUAGES = tuple(os.environ.get("SEQ_LANGUAGES", "tsn,xho,zul").split(","))
PROTOCOL_PROMPTS = {"ner": {"tsn": 2, "xho": 5, "zul": 5}, "pos": {"tsn": 3, "xho": 3, "zul": 3}}
BATCH1 = os.environ.get("SEQ_BATCH1") == "1"
NER_CODES = {"tsn": "tn", "xho": "xh", "zul": "zu"}
NER_REVISION = "6aa65cdbfa22d66e5b4ed176ac525c364cda08d1"
NER_DATASET = "anrilombard/masakhaner-x-parquet"
POS_TEMPLATES = tuple(f"masakhane_pos_tagging/lm_eval_p{i}" for i in range(1, 5))
MAXIMUM_INPUT_TOKENS = 1024
NCHLT_DATASETS = {"nchlt_ner": ("anrilombard/nchlt-ner-sa4", "db2569f7478726434264b3dac48d2352b22ff540"),
                  "nchlt_pos": ("anrilombard/nchlt-pos-sa4", "cd3b8fab81b1eec4ca512e4e6f9896cd6b7ddc8b")}
NCHLT_NER_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
NCHLT_POS_LABELS = json.loads(Path(pos_metrics.__file__).with_name("nchlt_pos_labels.json").read_text())["by_language"]
NCHLT_PROMPT = 1


def is_nchlt(task: str | None) -> bool:
    return bool(task) and task.startswith("nchlt_")


def pos_labels(task: str, language: str) -> list[str]:
    return list(NCHLT_POS_LABELS[language]) if is_nchlt(task) else list(UPOS_LABELS)


def pos_template(task: str, prompt: int) -> str:
    return f"nchlt_pos_tagging/lm_eval_p{prompt}" if is_nchlt(task) else POS_TEMPLATES[prompt - 1]
LOGGER = logging.getLogger("general_sequence_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("ner", "pos", "nchlt_ner", "nchlt_pos"))
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
    if binding.get("base_path") is None and adapter is None:
        import sys
        sys.path.insert(0, "/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1")
        from prepare_bindings import tree_sha256

        return {"base_path": str(checkpoint.absolute()), "adapter_path": None,
                "base_tree_sha256": tree_sha256(checkpoint), "adapter_tree_sha256": None, "files": {}}
    expected_checkpoint = Path(binding["base_path"])
    expected_adapter = Path(binding["adapter_path"])
    if checkpoint.absolute() != expected_checkpoint.absolute():
        raise ValueError(f"Unexpected {architecture} base path: {checkpoint}")
    if adapter.absolute() != expected_adapter.absolute():
        raise ValueError(f"Unexpected {architecture} adapter path: {adapter}")

    checked: dict[str, dict[str, str]] = {"base": {}, "adapter": {}}
    for group, root, field in (
        ("base", checkpoint, "base_files"),
        ("adapter", adapter, "adapter_files"),
    ):
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
        "adapter_path": str(adapter.absolute()),
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
    if is_nchlt(args.task):  # test access is gated by the rollout's select unit
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
    if is_nchlt(task):
        return {language: [NCHLT_PROMPT] for language in LANGUAGES}
    if phase == "validation":
        if os.environ.get("SEQ_VALIDATION_PROMPTS") == "protocol":
            return {language: [PROTOCOL_PROMPTS[task][language]] for language in LANGUAGES}
        count = 5 if task == "ner" else 4
        return {language: list(range(1, count + 1)) for language in LANGUAGES}
    assert selection is not None
    selected = selection["selected_prompts"][task]
    if not set(LANGUAGES) <= set(selected):
        raise ValueError(f"Incomplete {task} prompt selection")
    output = {language: [int(selected[language])] for language in LANGUAGES}
    maximum = 5 if task == "ner" else 4
    if any(not 1 <= prompts[0] <= maximum for prompts in output.values()):
        raise ValueError(f"Invalid selected {task} prompt")
    return output


def ner_task_name(language: str, prompt: int, phase: str, task: str = "ner") -> str:
    suffix = "val" if phase == "validation" else "test"
    if is_nchlt(task):
        return f"sallm_nchlt_ner_{language}_prompt_{prompt}_{suffix}"
    return f"sallm_masakhaner_{NER_CODES[language]}_prompt_{prompt}_{suffix}"


def load_nchlt_split(task: str, language: str, split: str) -> Dataset:
    name, revision = NCHLT_DATASETS[task]
    dataset = load_dataset(name, name=language, split=split, revision=revision)
    if split == "validation":  # fixed selection subsample (rollout: FFT_VAL_SUBSAMPLE=1)
        keep = val_subsample_indices(task, language, len(dataset))
        if keep is not None:
            print(f"VAL_SUBSAMPLE {task}/{language} {len(dataset)} -> {len(keep)}", flush=True)
            dataset = dataset.select(keep)
    return dataset


def load_nchlt_ner_split(*, language: str, split: str, **_: Any) -> DatasetDict:
    """text / target exactly as the training formatter builds them (format_ner, ' $$ '-joined 'LABEL: span')."""
    dataset = load_nchlt_split("nchlt_ner", language, split).map(lambda row: {
        "text": " ".join(row["tokens"]),
        "target": " $$ ".join(reconstruct_entities_from_iob(row["tokens"], row["ner_tags"], NCHLT_NER_TAGS))})
    return DatasetDict({split: dataset})


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
    if split == "validation":  # fixed selection subsample (rollout: FFT_VAL_SUBSAMPLE=1)
        keep = val_subsample_indices("ner", language, len(dataset))
        if keep is not None:
            print(f"VAL_SUBSAMPLE ner/{language} {len(dataset)} -> {len(keep)}", flush=True)
            dataset = dataset.select(keep)
    return DatasetDict({split: dataset})


def serialize_ner_prompt(doc: dict[str, Any], *, template: str) -> str:
    content = lm_eval_utils.apply_template(template, doc)
    return f"[BOS]        <|user|>\n        {content}[EOS]<|assistant|>"


def build_ner_tasks(
    source_root: Path,
    phase: str,
    prompts: dict[str, list[int]],
    task_name: str = "ner",
) -> tuple[TaskManager, list[ConfigurableTask], dict[str, Any]]:
    stem = "nchlt_ner" if is_nchlt(task_name) else "masakhaner"
    directory = source_root / "src/conf/eval/lm_eval_tasks" / (
        f"{stem}_validation" if phase == "validation" else f"{stem}_test"
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
            name = ner_task_name(language, prompt, phase, task_name)
            yaml_path = Path(manager.task_index[name]["yaml_path"])
            config = load_yaml_config(yaml_path=yaml_path)
            config["task"] = name
            prompt_template = config["doc_to_text"]
            if not isinstance(prompt_template, str):
                raise TypeError(f"Expected a string NER prompt template for {name}")

            config["doc_to_text"] = partial(
                serialize_ner_prompt, template=prompt_template
            )
            config["custom_dataset"] = partial(
                load_nchlt_ner_split if is_nchlt(task_name) else load_ner_split,
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
                "dataset_revision": NCHLT_DATASETS["nchlt_ner"][1] if is_nchlt(task_name) else NER_REVISION,
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
    prompts = chosen_prompts(args.task, args.phase, selection)
    source_root = Path(protocol["source_snapshot"]["path"])
    manager, tasks, evidence = build_ner_tasks(source_root, args.phase, prompts, args.task)
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
            pretrained, work_root / "tokenizer", False
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
            batch_size=1 if BATCH1 else "auto:4",
            max_batch_size=1 if BATCH1 else 16,
            device="cuda:0",
            limit=args.limit,
            bootstrap_iters=0,
            write_out=True,
            log_samples=True,
            apply_chat_template=False,
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
        "task": args.task,
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


POS_CACHED_CHECK_WORDS = int(os.environ.get("FFT_POS_CACHED_CHECK", "200"))
_POS_CACHED_CHECKED = 0


def decode_pos_row(
    model: Any,
    tokenizer: Any,
    tokens: list[str],
    prompt: str,
    device: torch.device,
    labels: list[str] = UPOS_LABELS,  # noqa: B006 - read-only; NCHLT passes the language's own tags
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
    # Prefix-cached scoring (28 Sep 2026, runners/pos_cached.py): same tags, far fewer forward tokens. The first
    # POS_CACHED_CHECK_WORDS words of each process are also scored the reference way and must agree.
    scorer = pos_cached.CachedScorer(model, device) if pos_cached.enabled(model, len(labels)) else None

    for index, token in enumerate(tokens):
        prefix = ("[" if index == 0 else ", ") + f"({token!r}, '"
        context_text, context_ids = append_text(
            tokenizer, context_text, context_ids, prefix
        )
        label_ids = {
            label: continuation_ids(tokenizer, context_text, context_ids, label)
            for label in labels
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
        reference = partial(
            score_labels,
            model=model,
            context_ids=scoring_context_ids,
            label_ids=label_ids,
            labels=list(labels),
            score_mode="mean",
            pad_token_id=int(pad_token_id),
            pad_to_multiple_of=64,
            device=device,
        )
        if scorer is not None and prefix_tokens_removed == 0:
            scorer.feed(context_ids)
            label, score = scorer.score(label_ids, list(labels))
            global _POS_CACHED_CHECKED
            if _POS_CACHED_CHECKED < POS_CACHED_CHECK_WORDS:
                _POS_CACHED_CHECKED += 1
                if reference()[0] != label:
                    raise AssertionError(f"cached POS scoring disagrees with the reference at word {index}")
        else:
            label, score, _ = reference()
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
    prompts = chosen_prompts(args.task, args.phase, selection)
    split = "validation" if args.phase == "validation" else "test"
    datasets = {
        language: load_nchlt_split(args.task, language, split) if is_nchlt(args.task) else _load_masakhapos_split(language, split)
        for language in LANGUAGES
    }
    evidence = {
        language: {
            "split": split,
            "rows": len(dataset),
            "tokens": sum(len(tokens) for tokens in dataset["tokens"]),
            "dataset_revision": NCHLT_DATASETS[args.task][1] if is_nchlt(args.task) else protocol["tasks"]["pos"]["revision"],
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
        labels = pos_labels(args.task, language)
        for prompt_number in prompts[language]:
            template_id = pos_template(args.task, prompt_number)
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
                ) = decode_pos_row(model, tokenizer, tokens, prompt, device, labels)
                if len(prediction) != len(gold) or any(
                    label not in labels for label in prediction
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
        "task": args.task,
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
        "label_set": ({language: pos_labels(args.task, language) for language in LANGUAGES}
                      if is_nchlt(args.task) else list(UPOS_LABELS)),
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
    assert serialize_ner_prompt({"text": "hello"}, template="{{text}}") == (
        "[BOS]        <|user|>\n        hello[EOS]<|assistant|>"
    )
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
    if args.task in ("ner", "nchlt_ner"):
        payload = run_ner(args, protocol, binding, selection, selection_hash)
    else:
        payload = run_pos(args, protocol, binding, selection, selection_hash)
    if not math.isfinite(float(payload["elapsed_seconds"])):
        raise ValueError("Non-finite elapsed time")
    atomic_json(args.output, payload)
    print(f"GENERAL_SEQUENCE_COMPLETE {args.output}")


if __name__ == "__main__":
    main()
