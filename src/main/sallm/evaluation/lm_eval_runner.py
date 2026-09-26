from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from copy import deepcopy
from datetime import date, datetime
from hashlib import sha1
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from lm_eval import evaluator
from lm_eval import tasks as lm_eval_tasks
from lm_eval.tasks import TaskManager
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sallm.chat_template import CANONICAL_CHAT_TEMPLATE
from sallm.config import ModelEvalConfig
from sallm.evaluation.config import TaskPack
from sallm.evaluation.harness import (
    load_model_and_tokenizer,
    prepare_model_for_evaluation,
    use_exact_xlstm_head_dims,
)
from sallm.evaluation.registry import (
    RERANK_LM_EVAL_TASK_DIR,
    load_rerank_task_pack,
    load_task_pack,
)
from sallm.models.optional import register_fla_gated_deltanet

logger = logging.getLogger(__name__)


def _lexical_lm_eval_tasks_root(module_file: str) -> Path:
    return Path(module_file).parent


LM_EVAL_TASKS_ROOT = _lexical_lm_eval_tasks_root(lm_eval_tasks.__file__)


def _prepare_include_paths(include_path: str | list[str]) -> list[str]:
    raw_paths = include_path if isinstance(include_path, list) else [include_path]
    prepared_paths: list[str] = []

    for raw_path in raw_paths:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()

        try:
            path.relative_to(LM_EVAL_TASKS_ROOT)
            prepared_paths.append(str(path))
            continue
        except ValueError:
            pass

        link_root = LM_EVAL_TASKS_ROOT / "_sallm_repo_overrides"
        link_root.mkdir(parents=True, exist_ok=True)
        link_name = f"{path.name}-{sha1(str(path).encode('utf-8')).hexdigest()[:8]}"
        link_path = link_root / link_name

        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink() and link_path.resolve() == path:
                prepared_paths.append(str(link_path))
                continue
            if link_path.is_dir() and link_path.resolve() == path:
                prepared_paths.append(str(link_path))
                continue
            if link_path.is_dir() and not link_path.is_symlink():
                raise FileExistsError(
                    "lm-eval include-path shim already exists and is not a "
                    f"symlink: {link_path}"
                )
            link_path.unlink()

        try:
            link_path.symlink_to(path, target_is_directory=True)
        except FileExistsError:
            if not link_path.exists() and not link_path.is_symlink():
                raise
            if link_path.resolve() != path:
                raise
        prepared_paths.append(str(link_path))

    return prepared_paths


TASK_PACK_SCOPES = {"eval", "rerank"}


def _format_model_args(
    *,
    pretrained_path: str,
    dtype: str | None,
    peft_adapter: str | None,
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
    if peft_adapter:
        args.append(f"peft={peft_adapter}")
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


def _materialize_model_for_lm_eval(
    model_cfg: ModelEvalConfig, cache_root: Path
) -> tuple[str, str | None]:
    # This precedes the AutoConfig and every AutoModel retry/materialization below.
    register_fla_gated_deltanet()
    if not model_cfg.peft_adapter:
        config = AutoConfig.from_pretrained(
            model_cfg.checkpoint,
            trust_remote_code=True,
        )
        _use_exact_xlstm_head_dims(config)
        if (
            getattr(config, "model_type", None) != "xlstm"
            or getattr(config, "mode", None) != "train"
        ):
            return model_cfg.checkpoint, None

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
            tokenizer = AutoTokenizer.from_pretrained(
                model_cfg.checkpoint,
                trust_remote_code=True,
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
        return str(base_dir), None
    if not model_cfg.merge_lora:
        base_dir = cache_root / "resized_base_model"
        if not base_dir.exists():
            tokenizer = AutoTokenizer.from_pretrained(
                model_cfg.peft_adapter,
                trust_remote_code=True,
            )
            dtype = getattr(torch, str(model_cfg.dtype), None)
            model = AutoModelForCausalLM.from_pretrained(
                model_cfg.checkpoint,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model.resize_token_embeddings(len(tokenizer))
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
        return str(base_dir), model_cfg.peft_adapter

    cache_root.mkdir(parents=True, exist_ok=True)
    merged_dir = cache_root / "merged_model"
    if merged_dir.exists():
        return str(merged_dir), None

    logger.info(
        "Merging PEFT adapter into temporary checkpoint for lm-eval at %s",
        merged_dir,
    )

    cfg_copy = deepcopy(model_cfg)
    cfg_copy.merge_lora = True

    model, tokenizer = load_model_and_tokenizer(cfg_copy)

    try:
        model = cast(Any, model).to("cpu")
    except Exception:
        pass

    merged_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(merged_dir)
    _sync_weight_tying_flag(model)
    _set_eval_safe_model_config(model)
    try:
        model.save_pretrained(merged_dir)
    except RuntimeError as exc:
        if "shared tensors" not in str(exc):
            raise
        logger.warning(
            "Retrying save_pretrained with safe_serialization=False due to "
            "shared tensors."
        )
        model.save_pretrained(merged_dir, safe_serialization=False)

    del model
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return str(merged_dir), None


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
    return CANONICAL_CHAT_TEMPLATE


def _prepare_tokenizer_for_lm_eval(
    pretrained_path: str,
    cache_root: Path,
    require_chat_template: bool,
) -> str | None:
    cache_root.mkdir(parents=True, exist_ok=True)
    tok_out = cache_root / (
        "tokenizer_chat" if require_chat_template else "tokenizer_raw"
    )
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
    peft_adapter: str | None,
    task_pack_scope: str,
) -> dict[str, Any]:
    pack: TaskPack = _load_pack(pack_name, task_pack_scope)
    pack_out = output_dir / pack_name
    pack_out.mkdir(parents=True, exist_ok=True)

    effective_apply_chat_template = pack.apply_chat_template
    if pack_overrides and "apply_chat_template" in pack_overrides:
        effective_apply_chat_template = bool(pack_overrides["apply_chat_template"])

    tokenizer_override = _prepare_tokenizer_for_lm_eval(
        pretrained_path,
        work_root / "_tokenizer",
        effective_apply_chat_template,
    )
    model_args = _format_model_args(
        pretrained_path=pretrained_path,
        dtype=model_cfg.dtype,
        peft_adapter=peft_adapter,
        tokenizer_override=tokenizer_override,
        tie_word_embeddings=model_cfg.tie_word_embeddings,
        extra_model_args=model_cfg.lm_eval_model_args,
        default_add_bos_token=not effective_apply_chat_template,
    )

    if model_cfg.peft_adapter and peft_adapter is None:
        logger.info("Using merged checkpoint for lm-eval: %s", pretrained_path)

    eval_kwargs: dict[str, Any] = {
        "model": "hf",
        "model_args": model_args,
        "device": model_cfg.device,
    }

    pack_kwargs = pack.to_lm_eval_kwargs()
    task_manager: TaskManager | None = None
    include_path = pack_kwargs.pop("include_path", None)
    include_defaults = bool(pack_kwargs.pop("include_defaults", True))
    eval_kwargs.update(pack_kwargs)
    eval_kwargs["apply_chat_template"] = effective_apply_chat_template

    if pack_overrides:
        override_kwargs = dict(pack_overrides)
        if "include_path" in override_kwargs:
            include_path = override_kwargs.pop("include_path")
        if "include_defaults" in override_kwargs:
            include_defaults = bool(override_kwargs.pop("include_defaults"))
        eval_kwargs.update(override_kwargs)

    effective_fewshot = int(eval_kwargs.get("num_fewshot", pack.fewshot))

    if task_pack_scope == "rerank":
        include_paths = include_path if isinstance(include_path, list) else []
        if include_path and not isinstance(include_path, list):
            include_paths = [include_path]
        include_paths.append(str(RERANK_LM_EVAL_TASK_DIR))
        include_path = include_paths

    if include_path:
        resolved_paths = _prepare_include_paths(include_path)
        task_manager = TaskManager(
            include_path=resolved_paths,
            include_defaults=include_defaults,
        )
        eval_kwargs["task_manager"] = task_manager

    logger.info(
        "Running lm-eval %s task pack '%s' with tasks=%s, fewshot=%s, batch_size=%s, "
        "apply_chat_template=%s",
        task_pack_scope,
        pack_name,
        ",".join(pack.tasks),
        effective_fewshot,
        eval_kwargs.get("batch_size", pack.batch_size),
        effective_apply_chat_template,
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
        "batch_size": pack.batch_size,
        "apply_chat_template": effective_apply_chat_template,
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
        pretrained_path, peft_adapter = _materialize_model_for_lm_eval(
            model_cfg, work_root / "_lm_eval"
        )

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
                    peft_adapter,
                    task_pack_scope,
                )
                summaries.append(summary)
            return summaries
        finally:
            shutil.rmtree(work_root, ignore_errors=True)
