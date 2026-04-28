from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import textwrap
from copy import deepcopy
from datetime import date, datetime
from hashlib import sha1
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from transformers import AutoTokenizer

from sallm.config import ModelEvalConfig
from sallm.evaluation.config import TaskPack
from sallm.evaluation.harness import load_model_and_tokenizer
from sallm.evaluation.registry import load_task_pack

logger = logging.getLogger(__name__)
LM_EVAL_TASKS_ROOT = (
    Path(__file__).resolve().parents[4]
    / ".venv"
    / "lib"
    / "python3.12"
    / "site-packages"
    / "lm_eval"
    / "tasks"
)


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


def _format_model_args(
    pretrained_path: str,
    dtype: str | None,
    peft_adapter: str | None,
    tokenizer_override: str | None = None,
    tie_word_embeddings: bool | None = None,
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
    if tie_word_embeddings is not None:
        args.append(f"tie_word_embeddings={str(tie_word_embeddings).lower()}")
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


def _materialize_model_for_lm_eval(
    model_cfg: ModelEvalConfig, cache_root: Path
) -> tuple[str, str | None]:
    if not model_cfg.peft_adapter:
        return model_cfg.checkpoint, None

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
    work_root: Path,
    pack_overrides: dict[str, Any] | None,
    pretrained_path: str,
    peft_adapter: str | None,
) -> dict[str, Any]:
    pack: TaskPack = load_task_pack(pack_name)
    pack_out = output_dir / pack_name
    pack_out.mkdir(parents=True, exist_ok=True)

    tokenizer_override = _prepare_tokenizer_for_lm_eval(
        pretrained_path, work_root / "_tokenizer", pack.apply_chat_template
    )
    model_args = _format_model_args(
        pretrained_path,
        model_cfg.dtype,
        peft_adapter,
        tokenizer_override,
        model_cfg.tie_word_embeddings,
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
    eval_kwargs["apply_chat_template"] = pack.apply_chat_template

    if pack_overrides:
        override_kwargs = dict(pack_overrides)
        if "include_path" in override_kwargs:
            include_path = override_kwargs.pop("include_path")
        if "include_defaults" in override_kwargs:
            include_defaults = bool(override_kwargs.pop("include_defaults"))
        eval_kwargs.update(override_kwargs)

    if include_path:
        resolved_paths = _prepare_include_paths(include_path)
        task_manager = TaskManager(
            include_path=resolved_paths,
            include_defaults=include_defaults,
        )
        eval_kwargs["task_manager"] = task_manager

    logger.info(
        "Running lm-eval task pack '%s' with tasks=%s, fewshot=%s, batch_size=%s, "
        "apply_chat_template=%s",
        pack_name,
        ",".join(pack.tasks),
        pack.fewshot,
        pack.batch_size,
        pack.apply_chat_template,
    )
    if task_manager is not None:
        logger.info(
            "Using extra lm-eval task search paths: %s", task_manager.include_path
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
                )
                summaries.append(summary)
            return summaries
        finally:
            shutil.rmtree(work_root, ignore_errors=True)
