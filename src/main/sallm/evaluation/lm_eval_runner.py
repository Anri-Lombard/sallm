from __future__ import annotations

import json
import logging
import textwrap
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from lm_eval import evaluator
from transformers import AutoTokenizer

from sallm.config import ModelEvalConfig
from sallm.evaluation.config import TaskPack
from sallm.evaluation.registry import load_task_pack

logger = logging.getLogger(__name__)


def _format_model_args(
    pretrained_path: str,
    dtype: str | None,
    peft_adapter: str | None,
    tokenizer_override: str | None = None,
    extra_args: dict[str, Any] | None = None,
) -> str:
    args: list[str] = [
        f"pretrained={pretrained_path}",
        "trust_remote_code=true",
    ]
    if dtype:
        args.append(f"dtype={dtype}")
    if peft_adapter:
        args.append(f"peft={peft_adapter}")
    if tokenizer_override:
        args.append(f"tokenizer={tokenizer_override}")
    if extra_args:
        for k, v in extra_args.items():
            args.append(f"{k}={v}")
    return ",".join(args)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if type(value).__name__ == "dtype":
        return str(value)
    return value


def _prepare_model_for_harness(
    model_cfg: ModelEvalConfig, cache_root: Path
) -> tuple[str, str | None]:
    if not model_cfg.peft_adapter:
        return model_cfg.checkpoint, None

    logger.info(
        "Using PEFT adapter directly: base=%s, adapter=%s",
        model_cfg.checkpoint,
        model_cfg.peft_adapter,
    )
    return model_cfg.checkpoint, model_cfg.peft_adapter


def _fallback_chat_template() -> str:
    return textwrap.dedent(
        """
        {%- if system_message %}
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
        {%- if add_generation_prompt %}<|assistant|>{%- endif %}
        """
    ).lstrip()


def _prepare_tokenizer_for_lm_eval(
    pretrained_path: str,
    cache_root: Path,
    require_chat_template: bool,
) -> str | None:
    cache_root.mkdir(parents=True, exist_ok=True)
    tok_out = cache_root / "tokenizer"
    if tok_out.exists():
        return str(tok_out)

    try:
        tok = AutoTokenizer.from_pretrained(
            pretrained_path, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        try:
            tok = AutoTokenizer.from_pretrained(pretrained_path, trust_remote_code=True)
        except Exception:
            return None

    needs_template = (
        require_chat_template and getattr(tok, "chat_template", None) is None
    )
    if needs_template:
        logger.info("Injecting fallback chat template for lm-eval tokenizer.")
        try:
            tok.chat_template = _fallback_chat_template()  # type: ignore[attr-defined]
        except Exception:
            pass

    try:
        tok_out.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(tok_out)
        if needs_template and getattr(tok, "chat_template", None) is None:
            import json as _json

            cfg_path = tok_out / "tokenizer_config.json"
            data = {}
            if cfg_path.exists():
                try:
                    data = _json.loads(cfg_path.read_text())
                except Exception:
                    data = {}
            data["chat_template"] = _fallback_chat_template()
            cfg_path.write_text(_json.dumps(data, indent=2))
        return str(tok_out)
    except Exception:
        return None


def _run_pack(
    pack_name: str,
    model_cfg: ModelEvalConfig,
    output_dir: Path,
    pack_overrides: dict[str, Any] | None,
    pretrained_path: str,
    peft_adapter: str | None,
) -> dict[str, Any]:
    pack: TaskPack = load_task_pack(pack_name)
    pack_out = output_dir / pack_name
    pack_out.mkdir(parents=True, exist_ok=True)

    tokenizer_source = peft_adapter if peft_adapter else pretrained_path
    tokenizer_override = _prepare_tokenizer_for_lm_eval(
        tokenizer_source, output_dir / "_tokenizer", pack.apply_chat_template
    )
    model_args = _format_model_args(
        pretrained_path,
        model_cfg.dtype,
        peft_adapter,
        tokenizer_override,
        model_cfg.extra_model_args,
    )

    if model_cfg.peft_adapter and peft_adapter is None:
        logger.info("Using merged checkpoint for lm-eval: %s", pretrained_path)

    eval_kwargs: dict[str, Any] = {
        "model": "hf",
        "model_args": model_args,
        "device": model_cfg.device,
    }

    pack_kwargs = pack.to_lm_eval_kwargs()
    eval_kwargs.update(pack_kwargs)
    eval_kwargs["apply_chat_template"] = pack.apply_chat_template

    if pack_overrides:
        eval_kwargs.update(pack_overrides)

    logger.info(
        "Running lm-eval task pack '%s' with tasks=%s, fewshot=%s, batch_size=%s, "
        "apply_chat_template=%s",
        pack_name,
        ",".join(pack.tasks),
        pack.fewshot,
        pack.batch_size,
        pack.apply_chat_template,
    )

    raw_result = evaluator.simple_evaluate(**eval_kwargs)

    result = _to_serializable(raw_result)

    result_path = pack_out / "results.json"
    with result_path.open("w") as handle:
        json.dump(result, handle, indent=2)

    logger.info(json.dumps(result.get("results", {}), indent=2))

    logger.info("Saved lm-eval results for '%s' to %s", pack_name, result_path)

    return {
        "type": "lm_eval",
        "task_pack": pack_name,
        "tasks": pack.tasks,
        "fewshot": pack.fewshot,
        "batch_size": pack.batch_size,
        "apply_chat_template": pack.apply_chat_template,
        "results": result.get("results", {}),
        "metrics": result.get("metrics", {}),
        "result_path": str(result_path),
    }


def run_task_pack_evaluations(
    pack_names: list[str],
    model_cfg: ModelEvalConfig,
    output_dir: Path,
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not pack_names:
        return []

    pretrained_path, peft_adapter = _prepare_model_for_harness(
        model_cfg, output_dir / "_lm_eval"
    )

    summaries: list[dict[str, Any]] = []
    for pack_name in pack_names:
        pack_overrides = None
        if overrides and pack_name in overrides:
            raw_override = overrides[pack_name]
            if raw_override is not None:
                if not isinstance(raw_override, dict):
                    raise TypeError(
                        "evaluation.overrides values must be mappings keyed by "
                        "task-pack"
                    )
                pack_overrides = raw_override

        summary = _run_pack(
            pack_name,
            model_cfg,
            output_dir,
            pack_overrides,
            pretrained_path,
            peft_adapter,
        )
        summaries.append(summary)
    return summaries
