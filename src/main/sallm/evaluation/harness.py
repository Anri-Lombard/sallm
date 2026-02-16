from __future__ import annotations

import json
import logging
import random
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset
from peft import PeftModel
from tokenizers.decoders import ByteLevel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sallm.config import (
    FewshotTemplateMode,
    GenerationEvalTaskConfig,
    ModelEvalConfig,
)
from sallm.data.afrihg import load_afrihg_from_github
from sallm.data.factory import build_conversation_dataset
from sallm.data.t2x import load_t2x_from_github
from sallm.evaluation.generation_metrics import GenerationEvaluator

logger = logging.getLogger(__name__)


def _prepare_tokenizer(tokenizer: AutoTokenizer) -> AutoTokenizer:
    tokenizer.backend_tokenizer.decoder = ByteLevel()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for decoder-only model generation
    return tokenizer


def _infer_vocab_size_from_peft_error(exc: RuntimeError) -> int | None:
    message = str(exc)
    if "size mismatch" not in message:
        return None

    checkpoint_max = 0
    current_max = 0
    for line in message.splitlines():
        if "size mismatch for" not in line:
            continue
        if not any(
            key in line for key in ("embeddings", "lm_head", "lora_embedding_A")
        ):
            continue

        checkpoint_match = re.search(
            r"copying a param with shape torch\.Size\(\[([0-9,\s]+)\]\)",
            line,
        )
        current_match = re.search(
            r"the shape in current model is torch\.Size\(\[([0-9,\s]+)\]\)",
            line,
        )

        if checkpoint_match:
            dims = [
                int(part.strip())
                for part in checkpoint_match.group(1).split(",")
                if part.strip().isdigit()
            ]
            if dims:
                checkpoint_max = max(checkpoint_max, max(dims))

        if current_match:
            dims = [
                int(part.strip())
                for part in current_match.group(1).split(",")
                if part.strip().isdigit()
            ]
            if dims:
                current_max = max(current_max, max(dims))

    if checkpoint_max > current_max and checkpoint_max >= 50000:
        return checkpoint_max
    return None


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    try:
        return getattr(torch, dtype_str)
    except AttributeError as exc:
        raise ValueError(f"Unsupported dtype '{dtype_str}'") from exc


def _load_tokenizer_and_pretrained(
    checkpoint: str, trust_remote_code: bool = True, adapter_path: str | None = None
) -> tuple[AutoTokenizer, str]:
    checkpoint_path = Path(checkpoint)
    pretrained_id = checkpoint
    if checkpoint_path.exists():
        try:
            pretrained_id = str(checkpoint_path.resolve())
        except Exception:
            pretrained_id = str(checkpoint_path)

    def _is_hf_hub_path(path_str: str) -> bool:
        if path_str.startswith(("/", ".", "~")):
            return False
        parts = path_str.split("/")
        return len(parts) == 2 and all(p and not p.startswith(".") for p in parts)

    adapter_tokenizer = None
    if adapter_path:
        candidate = Path(adapter_path)
        if candidate.exists() and candidate.is_dir():
            tokenizer_files = (
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "tokenizer.model",
            )
            if any((candidate / name).exists() for name in tokenizer_files):
                adapter_tokenizer = candidate
        elif _is_hf_hub_path(adapter_path):
            # Try loading tokenizer from HF hub adapter
            try:
                logger.info("Loading tokenizer from HF hub adapter: %s", adapter_path)
                tok = AutoTokenizer.from_pretrained(
                    adapter_path, trust_remote_code=trust_remote_code
                )
                return tok, pretrained_id
            except Exception as exc:
                logger.warning(
                    (
                        "Adapter tokenizer load failed for '%s' (%s). "
                        "Falling back to base checkpoint."
                    ),
                    adapter_path,
                    exc,
                )

    tokenizer_root = adapter_tokenizer
    if tokenizer_root is None and checkpoint_path.exists():
        tokenizer_root = checkpoint_path

    if tokenizer_root is not None and tokenizer_root.exists():
        try:
            tokenizer_resolved = str(tokenizer_root.resolve())
        except Exception:
            tokenizer_resolved = str(tokenizer_root)
        source_label = (
            "adapter checkpoint"
            if adapter_tokenizer is not None and tokenizer_root == adapter_tokenizer
            else "local checkpoint"
        )
        logger.info("Loading tokenizer from %s: %s", source_label, tokenizer_resolved)
        tok = AutoTokenizer.from_pretrained(
            tokenizer_resolved,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        return tok, pretrained_id

    logger.info("Loading tokenizer from HF hub or identifier: %s", checkpoint)
    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
    return tok, pretrained_id


def load_model_and_tokenizer(
    model_cfg: ModelEvalConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer, pretrained_id = _load_tokenizer_and_pretrained(
        model_cfg.checkpoint,
        trust_remote_code=True,
        adapter_path=model_cfg.peft_adapter,
    )
    tokenizer = _prepare_tokenizer(tokenizer)

    torch_dtype = _resolve_dtype(model_cfg.dtype)
    logger.info("Loading model weights from %s", pretrained_id)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    vocab_size = len(tokenizer)
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    if vocab_size > model_vocab_size:
        logger.info(
            "Resizing model embeddings from %d to %d to match tokenizer vocabulary.",
            model_vocab_size,
            vocab_size,
        )
        model.resize_token_embeddings(vocab_size)

    if model_cfg.peft_adapter:
        logger.info("Loading PEFT adapter from %s", model_cfg.peft_adapter)
        try:
            model = PeftModel.from_pretrained(model, model_cfg.peft_adapter)
        except RuntimeError as exc:
            target_vocab = _infer_vocab_size_from_peft_error(exc)
            adapter_tokenizer = None
            try:
                adapter_tokenizer = AutoTokenizer.from_pretrained(
                    model_cfg.peft_adapter, trust_remote_code=True
                )
                target_vocab = max(target_vocab or 0, len(adapter_tokenizer))
            except Exception as tok_exc:
                logger.warning(
                    (
                        "Unable to reload adapter tokenizer from '%s' during "
                        "PEFT retry: %s"
                    ),
                    model_cfg.peft_adapter,
                    tok_exc,
                )

            if target_vocab is None:
                raise

            current_vocab = model.get_input_embeddings().weight.shape[0]
            if target_vocab != current_vocab:
                logger.warning(
                    (
                        "Retrying PEFT adapter load after resizing embeddings "
                        "from %d to %d."
                    ),
                    current_vocab,
                    target_vocab,
                )
                # A failed initial PEFT load can partially wrap embeddings with LoRA
                # modules, which breaks resize_token_embeddings. Reload a clean base.
                model = AutoModelForCausalLM.from_pretrained(
                    pretrained_id,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                model.resize_token_embeddings(target_vocab)

            if adapter_tokenizer is not None:
                tokenizer = _prepare_tokenizer(adapter_tokenizer)

            model = PeftModel.from_pretrained(model, model_cfg.peft_adapter)
        if model_cfg.merge_lora:
            logger.info("Merging LoRA weights into the base model for evaluation.")
            model = model.merge_and_unload()

    device = torch.device(model_cfg.device)
    model.to(device)
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = (
        tokenizer.pad_token_id or tokenizer.eos_token_id
    )
    return model, tokenizer


def _load_raw_split(task_cfg: GenerationEvalTaskConfig, split_key: str) -> Dataset:
    ds_cfg = task_cfg.dataset
    split_name = ds_cfg.splits.get(split_key, split_key)
    lang_list = set(ds_cfg.languages or [])

    if isinstance(ds_cfg.hf_name, str) and ds_cfg.hf_name.startswith("github:"):
        gh_ref = ds_cfg.hf_name[len("github:") :]
        if "francois-meyer/t2x" in gh_ref or gh_ref.strip().endswith("/t2x"):
            dataset_dict = load_t2x_from_github()
        else:
            langs = [ds_cfg.subset] if ds_cfg.subset else ds_cfg.languages
            dataset_dict = load_afrihg_from_github(languages=langs)

        if not isinstance(dataset_dict, DatasetDict):
            raise TypeError(
                "Expected DatasetDict from GitHub loader; received "
                f"{type(dataset_dict)}"
            )
        if split_name not in dataset_dict:
            available = ", ".join(dataset_dict.keys())
            raise ValueError(
                f"Requested split '{split_name}' not found. Available: {available}"
            )
        raw_ds = dataset_dict[split_name]
    else:
        available_configs = get_dataset_config_names(
            ds_cfg.hf_name, trust_remote_code=True
        )
        load_name = ds_cfg.subset
        filter_after_load = False
        if ds_cfg.subset and ds_cfg.subset not in available_configs:
            load_name = None
            filter_after_load = True

        raw_ds = load_dataset(
            ds_cfg.hf_name,
            name=load_name,
            split=split_name,
            trust_remote_code=True,
        )

        if filter_after_load and ds_cfg.subset:

            def _matches_lang(ex: dict[str, Any]) -> bool:
                if "lang" in ex:
                    return ex["lang"] == ds_cfg.subset
                if "language" in ex:
                    return ex["language"] == ds_cfg.subset
                if "language_code" in ex:
                    return ex["language_code"] == ds_cfg.subset
                return False

            raw_ds = raw_ds.filter(_matches_lang)

    if lang_list:

        def _in_lang_list(ex: dict[str, Any]) -> bool:
            code = ex.get("lang") or ex.get("language_code") or ex.get("language")
            return code in lang_list

        raw_ds = raw_ds.filter(_in_lang_list)

    return raw_ds


def build_evaluation_dataset(task_cfg: GenerationEvalTaskConfig) -> Dataset:
    raw_ds = _load_raw_split(task_cfg, task_cfg.split)
    cfg_stub = SimpleNamespace(dataset=task_cfg.dataset)
    processed = build_conversation_dataset(raw_ds, cfg_stub)

    if task_cfg.fewshot <= 0:
        if task_cfg.system_prompt and len(processed) > 0:
            system_column = [task_cfg.system_prompt] * len(processed)
            processed = processed.add_column("system_message", system_column)
        return processed

    demo_raw = _load_raw_split(task_cfg, task_cfg.fewshot_split)
    demo_ds = build_conversation_dataset(demo_raw, cfg_stub)

    if len(demo_ds) == 0:
        raise RuntimeError(
            f"Few-shot evaluation requested with split '{task_cfg.fewshot_split}' "
            "but no examples were found."
        )

    seed = (
        task_cfg.fewshot_seed
        if task_cfg.fewshot_seed is not None
        else (task_cfg.sample_seed if task_cfg.sample_seed is not None else 13)
    )
    rng = random.Random(seed)

    demo_pool = [demo_ds[i] for i in range(len(demo_ds))]

    effective_budget = task_cfg.fewshot_token_budget
    if effective_budget is not None and task_cfg.prompt_headroom_tokens is not None:
        effective_budget = max(0, effective_budget - task_cfg.prompt_headroom_tokens)

    def _approx_prompt_tokens(messages: list[dict[str, str]]) -> int:
        return sum(len(msg.get("content", "").split()) for msg in messages)

    def _copy_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages]

    def _filtered_pool(
        target_lang: str | None, target_template: str | None
    ) -> list[dict[str, Any]]:
        candidates = demo_pool
        if task_cfg.fewshot_lang_match and target_lang:
            lang_filtered = [
                rec for rec in candidates if rec.get("lang") == target_lang
            ]
            if lang_filtered:
                candidates = lang_filtered
        if (
            task_cfg.fewshot_template_mode == FewshotTemplateMode.SAME
            and target_template
        ):
            template_filtered = [
                rec for rec in candidates if rec.get("template_id") == target_template
            ]
            if template_filtered:
                candidates = template_filtered
        return candidates

    def _select_demos(pool: list[dict[str, Any]], idx: int) -> list[dict[str, Any]]:
        if not pool:
            return []
        k = task_cfg.fewshot
        mode = task_cfg.fewshot_template_mode
        if mode == FewshotTemplateMode.CYCLE:
            total = len(pool)
            return [pool[(idx + offset) % total] for offset in range(k)]
        if mode == FewshotTemplateMode.RANDOM:
            if len(pool) >= k:
                return rng.sample(pool, k)
            return [rng.choice(pool) for _ in range(k)]
        # SAME mode or fallback deterministic selection
        if len(pool) >= k:
            return pool[:k]
        out = list(pool)
        while len(out) < k:
            out.append(pool[len(out) % len(pool)])
        return out

    records: list[dict[str, Any]] = []
    for idx in range(len(processed)):
        example = processed[idx]
        target_lang = example.get("lang")
        target_template = example.get("template_id")
        pool = _filtered_pool(target_lang, target_template)
        if not pool and task_cfg.fewshot_template_mode == FewshotTemplateMode.SAME:
            pool = _filtered_pool(target_lang, None)
        if not pool:
            pool = demo_pool

        selected = _select_demos(pool, idx)

        if effective_budget is not None:
            base_prompt_tokens = _approx_prompt_tokens(example["messages"][:-1])
            remaining = max(effective_budget - base_prompt_tokens, 0)
            trimmed: list[dict[str, Any]] = []
            consumed = 0
            for demo in selected:
                demo_tokens = _approx_prompt_tokens(demo["messages"])
                if demo_tokens <= 0:
                    trimmed.append(demo)
                    continue
                if consumed + demo_tokens > remaining:
                    if not trimmed:
                        trimmed.append(demo)
                    break
                trimmed.append(demo)
                consumed += demo_tokens
            selected = trimmed

        final_messages: list[dict[str, str]] = []
        for demo in selected:
            final_messages.extend(_copy_messages(demo["messages"]))
        final_messages.extend(_copy_messages(example["messages"]))

        record = {key: deepcopy(example[key]) for key in example.keys()}
        record["messages"] = final_messages
        record["fewshot_k"] = len(selected)
        record["fewshot_template_ids"] = [
            demo.get("template_id")
            for demo in selected
            if demo.get("template_id") is not None
        ]
        if task_cfg.system_prompt:
            record["system_message"] = task_cfg.system_prompt
        records.append(record)

    return Dataset.from_list(records)


def run_generation_task(
    task_cfg: GenerationEvalTaskConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_evaluation_dataset(task_cfg)
    logger.info(
        "Prepared %d evaluation examples for task '%s'.", len(dataset), task_cfg.id
    )

    evaluator = GenerationEvaluator(
        tokenizer,
        max_new_tokens=task_cfg.max_new_tokens,
        max_samples_per_lang=task_cfg.max_samples_per_lang,
        sample_seed=task_cfg.sample_seed,
        decoding=task_cfg.decoding,
    )

    result = evaluator.evaluate(
        model,
        dataset,
        world_size=1,
        metric_prefix=f"eval/{task_cfg.id}",
        collect_examples=True,
    )

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(result.metrics, handle, indent=2)
    logger.info("Saved metrics for task '%s' to %s", task_cfg.id, metrics_path)

    examples_path = out_dir / "examples.jsonl"
    with examples_path.open("w") as handle:
        for lang_key, lang_result in result.per_language.items():
            for example in lang_result.examples:
                record = {
                    "task": task_cfg.id,
                    "language": lang_key,
                    "prompt": example.prompt_text,
                    "prediction": example.prediction,
                    "reference": example.reference,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "task": task_cfg.id,
        "metrics": result.metrics,
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    return summary
