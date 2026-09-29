from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from tokenizers.processors import TemplateProcessing
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sallm.config import ModelEvalConfig
from sallm.evaluation.config import TASK_MANAGER_KWARG_KEYS, TaskPack
from sallm.evaluation.harness import (
    prepare_model_for_evaluation,
    use_exact_xlstm_head_dims,
)
from sallm.evaluation.registry import (
    CONF_DIR,
    RERANK_LM_EVAL_TASK_DIR,
    load_rerank_task_pack,
    load_task_pack,
)
from sallm.models.optional import register_fla_gated_deltanet
from sallm.templates.chat import DEFAULT_CHAT_TEMPLATE

logger = logging.getLogger(__name__)
PROJECT_ROOT = CONF_DIR.parent.parent


def _resolve_include_path(raw_path: str | Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [Path.cwd() / path, PROJECT_ROOT / path]

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            if not candidate.is_dir():
                raise NotADirectoryError(
                    f"lm-eval include path is not a directory: {candidate}"
                )
            return candidate

    attempted = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        f"lm-eval include path '{raw_path}' was not found. Tried: {attempted}"
    )


def _resolve_include_paths(include_path: str | Path | list[str | Path]) -> list[str]:
    raw_paths = include_path if isinstance(include_path, list) else [include_path]
    return [str(_resolve_include_path(raw_path)) for raw_path in raw_paths]


def _append_include_path(
    include_path: str | Path | list[str | Path] | None,
    extra_path: str | Path,
) -> list[str | Path]:
    include_paths = include_path if isinstance(include_path, list) else []
    if include_path and not isinstance(include_path, list):
        include_paths = [include_path]
    return [*include_paths, extra_path]


def _split_task_manager_kwargs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    evaluator_kwargs = dict(kwargs)
    task_manager_kwargs: dict[str, Any] = {}
    for key in TASK_MANAGER_KWARG_KEYS:
        if key in evaluator_kwargs:
            task_manager_kwargs[key] = evaluator_kwargs.pop(key)
    return evaluator_kwargs, task_manager_kwargs


TASK_PACK_SCOPES = {"eval", "rerank"}


def _format_model_args(
    *,
    pretrained_path: str,
    dtype: str | None,
    tokenizer_override: str | None = None,
    tie_word_embeddings: bool | None = None,
    extra_model_args: dict[str, Any] | None = None,
    default_add_bos_token: bool = False,
) -> str:
    args: list[str] = [
        f"pretrained={pretrained_path}",
        "trust_remote_code=true",
    ]
    if "add_bos_token" not in (extra_model_args or {}):
        args.append(f"add_bos_token={str(default_add_bos_token).lower()}")
    if dtype:
        args.append(f"dtype={dtype}")
    if tokenizer_override:
        args.append(f"tokenizer={tokenizer_override}")
    if tie_word_embeddings is not None:
        args.append(f"tie_word_embeddings={str(tie_word_embeddings).lower()}")
    for key, value in (extra_model_args or {}).items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = str(value).lower()
        args.append(f"{key}={value}")
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
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}.{name}"
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if type(value).__name__ == "dtype":
        return str(value)
    return value


def _materialize_model_for_lm_eval(model_cfg: ModelEvalConfig, cache_root: Path) -> str:
    # This precedes the AutoConfig and every AutoModel retry/materialization below.
    register_fla_gated_deltanet()
    config = AutoConfig.from_pretrained(
        model_cfg.checkpoint,
        trust_remote_code=True,
    )
    _use_exact_xlstm_head_dims(config)
    if (
        getattr(config, "model_type", None) != "xlstm"
        or getattr(config, "mode", None) != "train"
    ):
        return model_cfg.checkpoint

    base_dir = cache_root / "eval_safe_base_model"
    if not base_dir.exists():
        logger.info(
            "Materializing eval-safe xLSTM base checkpoint at %s",
            base_dir,
        )
        dtype = getattr(torch, str(model_cfg.dtype), None)
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.checkpoint,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = cast(
            Any,
            AutoTokenizer.from_pretrained(
                model_cfg.checkpoint,
                trust_remote_code=True,
            ),
        )
        _set_eval_safe_model_config(model)
        _sync_weight_tying_flag(model)
        base_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(base_dir)
        try:
            model.save_pretrained(base_dir)
        except RuntimeError as exc:
            if "shared tensors" not in str(exc):
                raise
            logger.warning(
                "Retrying save_pretrained with safe_serialization=False due to "
                "shared tensors."
            )
            model.save_pretrained(base_dir, safe_serialization=False)
    return str(base_dir)


def _set_eval_safe_model_config(model) -> None:
    prepare_model_for_evaluation(model)


def _use_exact_xlstm_head_dims(config) -> None:
    use_exact_xlstm_head_dims(config)


def _sync_weight_tying_flag(model) -> None:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "tie_word_embeddings"):
        return
    get_input = getattr(model, "get_input_embeddings", None)
    get_output = getattr(model, "get_output_embeddings", None)
    if not callable(get_input) or not callable(get_output):
        return
    input_emb = get_input()
    output_emb = get_output()
    if input_emb is None or output_emb is None:
        return
    input_weight = getattr(input_emb, "weight", None)
    output_weight = getattr(output_emb, "weight", None)
    if input_weight is None or output_weight is None:
        return
    shared = input_weight is output_weight
    if not shared:
        shared = input_weight.data_ptr() == output_weight.data_ptr()
    if shared and not bool(config.tie_word_embeddings):
        logger.info(
            "Detected tied embeddings; updating config.tie_word_embeddings to "
            "True before saving."
        )
        config.tie_word_embeddings = True
    if not shared and bool(config.tie_word_embeddings):
        logger.info(
            "Detected untied embeddings; updating config.tie_word_embeddings to "
            "False before saving."
        )
        config.tie_word_embeddings = False


def _resolve_ephemeral_eval_root() -> Path:
    candidates = [
        os.environ.get("SLURM_TMPDIR"),
        os.environ.get("TMPDIR"),
        f"/tmp/{os.environ.get('USER', 'sallm')}",
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        return path
    return Path(tempfile.gettempdir())


def _fallback_chat_template() -> str:
    return DEFAULT_CHAT_TEMPLATE


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
        except Exception as exc:
            if not require_chat_template:
                raise RuntimeError(
                    "Unable to prepare the required raw lm-eval tokenizer."
                ) from exc
            return None

    if tok is None:
        return None

    needs_template = (
        require_chat_template and getattr(tok, "chat_template", None) is None
    )
    if needs_template:
        logger.info("Injecting fallback chat template for lm-eval tokenizer.")
        try:
            cast(Any, tok).chat_template = _fallback_chat_template()
        except Exception:
            pass

    if not require_chat_template:
        bos_token = tok.bos_token
        bos_token_id = tok.bos_token_id
        backend = getattr(tok, "backend_tokenizer", None)
        if bos_token is None or bos_token_id is None or backend is None:
            raise ValueError("Raw lm-eval requires a fast tokenizer with a BOS token.")
        backend.post_processor = TemplateProcessing(
            single=f"{bos_token} $A",
            pair=f"{bos_token} $A $B",
            special_tokens=[(bos_token, bos_token_id)],
        )

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
    except Exception as exc:
        if not require_chat_template:
            raise RuntimeError(
                "Unable to persist the required raw lm-eval tokenizer."
            ) from exc
        return None


def _is_xlstm_checkpoint(pretrained_path: str) -> bool:
    config = AutoConfig.from_pretrained(pretrained_path, trust_remote_code=True)
    return getattr(config, "model_type", None) == "xlstm"


def _load_pack(pack_name: str, task_pack_scope: str) -> TaskPack:
    if task_pack_scope == "eval":
        return load_task_pack(pack_name)
    if task_pack_scope == "rerank":
        return load_rerank_task_pack(pack_name)
    supported = ", ".join(sorted(TASK_PACK_SCOPES))
    raise ValueError(
        f"Unsupported task_pack_scope '{task_pack_scope}'. "
        f"Supported scopes: {supported}."
    )


def _run_pack(
    pack_name: str,
    model_cfg: ModelEvalConfig,
    output_dir: Path,
    work_root: Path,
    pack_overrides: dict[str, Any] | None,
    pretrained_path: str,
    task_pack_scope: str,
    is_xlstm: bool = False,
) -> dict[str, Any]:
    pack: TaskPack = _load_pack(pack_name, task_pack_scope)
    pack_out = output_dir / pack_name
    pack_out.mkdir(parents=True, exist_ok=True)

    evaluator_overrides: dict[str, Any] = {}
    task_manager_overrides: dict[str, Any] = {}
    if pack_overrides:
        evaluator_overrides, task_manager_overrides = _split_task_manager_kwargs(
            pack_overrides
        )
    effective_apply_chat_template = evaluator_overrides.get(
        "apply_chat_template", pack.apply_chat_template
    )
    if not isinstance(effective_apply_chat_template, bool):
        raise TypeError("apply_chat_template override must be a boolean")
    add_bos_token = not effective_apply_chat_template
    tokenizer_cache = (
        "_tokenizer_chat" if effective_apply_chat_template else "_tokenizer_raw"
    )

    tokenizer_override = _prepare_tokenizer_for_lm_eval(
        pretrained_path,
        work_root / tokenizer_cache,
        effective_apply_chat_template,
    )
    model_args = _format_model_args(
        pretrained_path=pretrained_path,
        dtype=model_cfg.dtype,
        tokenizer_override=tokenizer_override,
        tie_word_embeddings=model_cfg.tie_word_embeddings,
        extra_model_args=model_cfg.lm_eval_model_args,
        default_add_bos_token=add_bos_token,
    )

    eval_kwargs: dict[str, Any] = {
        "model": "hf",
        "model_args": model_args,
        "device": model_cfg.device,
    }

    pack_kwargs = pack.to_evaluator_kwargs()
    task_manager_kwargs = pack.to_task_manager_kwargs()
    task_manager: TaskManager | None = None
    eval_kwargs.update(pack_kwargs)
    eval_kwargs["apply_chat_template"] = effective_apply_chat_template
    task_manager_kwargs.update(task_manager_overrides)
    eval_kwargs.update(evaluator_overrides)
    if is_xlstm and eval_kwargs.get("batch_size") != 1:
        # xLSTM ignores the attention mask, so padded batches leak into its state.
        logger.info("Using batch_size=1 for xLSTM lm-eval.")
        eval_kwargs["batch_size"] = 1

    if eval_kwargs.get("use_cache") is not None:
        raise ValueError("lm-eval response caching is disabled")

    effective_fewshot = int(eval_kwargs.get("num_fewshot", pack.fewshot))

    if task_pack_scope == "rerank":
        task_manager_kwargs["include_path"] = _append_include_path(
            task_manager_kwargs.get("include_path"),
            RERANK_LM_EVAL_TASK_DIR,
        )

    if task_manager_kwargs:
        include_path = task_manager_kwargs.get("include_path")
        if include_path:
            task_manager_kwargs["include_path"] = _resolve_include_paths(include_path)
        if "include_defaults" in task_manager_kwargs:
            task_manager_kwargs["include_defaults"] = bool(
                task_manager_kwargs["include_defaults"]
            )
        task_manager = TaskManager(**task_manager_kwargs)
        eval_kwargs["task_manager"] = task_manager

    logger.info(
        "Running lm-eval %s task pack '%s' with tasks=%s, fewshot=%s, batch_size=%s, "
        "apply_chat_template=%s, add_bos_token=%s",
        task_pack_scope,
        pack_name,
        ",".join(pack.tasks),
        effective_fewshot,
        eval_kwargs.get("batch_size", pack.batch_size),
        effective_apply_chat_template,
        add_bos_token,
    )
    if task_manager is not None:
        logger.info(
            "Using extra lm-eval task search paths: %s", task_manager.include_path
        )

    raw_result = evaluator.simple_evaluate(**eval_kwargs)

    result = _to_serializable(raw_result)
    result_config = result.get("config", {})
    if isinstance(result_config, dict) and result_config.get("num_fewshot") is not None:
        effective_fewshot = int(result_config["num_fewshot"])

    result_path = pack_out / "results.json"
    with result_path.open("w") as handle:
        json.dump(result, handle, indent=2)

    logger.info(json.dumps(result.get("results", {}), indent=2))

    logger.info("Saved lm-eval results for '%s' to %s", pack_name, result_path)

    return {
        "type": "lm_eval",
        "task_pack": pack_name,
        "task_pack_scope": task_pack_scope,
        "tasks": pack.tasks,
        "fewshot": effective_fewshot,
        "batch_size": eval_kwargs.get("batch_size", pack.batch_size),
        "apply_chat_template": effective_apply_chat_template,
        "add_bos_token": add_bos_token,
        "results": result.get("results", {}),
        "metrics": result.get("metrics", {}),
        "result_path": str(result_path),
    }


def run_task_pack_evaluations(
    pack_names: list[str],
    model_cfg: ModelEvalConfig,
    output_dir: Path,
    overrides: dict[str, Any] | None = None,
    task_pack_scope: str = "eval",
) -> list[dict[str, Any]]:
    if not pack_names:
        return []
    if task_pack_scope not in TASK_PACK_SCOPES:
        supported = ", ".join(sorted(TASK_PACK_SCOPES))
        raise ValueError(
            f"Unsupported task_pack_scope '{task_pack_scope}'. "
            f"Supported scopes: {supported}."
        )

    temp_root_parent = _resolve_ephemeral_eval_root()
    with tempfile.TemporaryDirectory(
        prefix="sallm_lm_eval_", dir=temp_root_parent
    ) as temp_root:
        work_root = Path(temp_root)
        pretrained_path = _materialize_model_for_lm_eval(
            model_cfg, work_root / "_lm_eval"
        )
        is_xlstm = _is_xlstm_checkpoint(model_cfg.checkpoint)

        summaries: list[dict[str, Any]] = []
        try:
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
                    work_root,
                    pack_overrides,
                    pretrained_path,
                    task_pack_scope,
                    is_xlstm=is_xlstm,
                )
                summaries.append(summary)
            return summaries
        finally:
            shutil.rmtree(work_root, ignore_errors=True)
