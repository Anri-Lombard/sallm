from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import lm_eval
import numpy as np
import torch
from lm_eval.models.huggingface import HFLM
from peft import PeftModel
from tokenizers.decoders import ByteLevel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sallm.config import ModelEvalConfig
from sallm.evaluation.config import TaskPack

logger = logging.getLogger(__name__)


def _safe_json_encoder(obj):
    if isinstance(obj, np.integer) or isinstance(obj, np.floating):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.ndim else obj.item()
    if isinstance(obj, torch.dtype):
        return str(obj)
    return str(obj)


def _load_tokenizer_and_pretrained(checkpoint: str, trust_remote_code: bool = True):
    path = Path(checkpoint)
    if path.exists():
        resolved = str(path.resolve())
        logger.info("Loading tokenizer from local checkpoint: %s", resolved)
        tok = AutoTokenizer.from_pretrained(
            resolved, trust_remote_code=trust_remote_code, local_files_only=True
        )
        return tok, resolved
    logger.info("Loading tokenizer from HF hub or identifier: %s", checkpoint)
    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
    return tok, checkpoint


def _torch_dtype_from_name(name: str) -> torch.dtype | None:
    if not name:
        return None
    if name == "auto":
        return None
    if hasattr(torch, name):
        candidate = getattr(torch, name)
        if isinstance(candidate, torch.dtype):
            return candidate
    raise ValueError(f"Unsupported torch dtype '{name}'.")


def _merge_lora_checkpoint(
    checkpoint: str, adapter: str, tok, dtype_name: str | None
) -> tempfile.TemporaryDirectory:
    dtype = None
    if dtype_name:
        dtype = _torch_dtype_from_name(dtype_name)
    base = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, torch_dtype=dtype
    )
    base.resize_token_embeddings(len(tok))
    merged = PeftModel.from_pretrained(base, adapter)
    merged = merged.merge_and_unload()
    merged = merged.to("cpu")
    temp_dir = tempfile.TemporaryDirectory()
    merged.save_pretrained(temp_dir.name)
    tok.save_pretrained(temp_dir.name)
    return temp_dir


def _build_lm(
    checkpoint: str,
    adapter: str | None,
    tok,
    device: str,
    dtype_name: str | None,
    merge: bool,
) -> tuple[HFLM, tempfile.TemporaryDirectory | None]:
    try:
        model = HFLM(
            pretrained=checkpoint,
            tokenizer=tok,
            peft=adapter,
            device=device,
            dtype=dtype_name,
            trust_remote_code=True,
        )
        return model, None
    except RuntimeError as exc:
        if not adapter or not merge:
            raise
        message = str(exc)
        if "size mismatch" not in message:
            raise
        temp_dir = _merge_lora_checkpoint(checkpoint, adapter, tok, dtype_name)
        model = HFLM(
            pretrained=temp_dir.name,
            tokenizer=tok,
            peft=None,
            device=device,
            dtype=dtype_name,
            trust_remote_code=True,
        )
        return model, temp_dir


def evaluate_pack(
    pack: TaskPack,
    model_cfg: ModelEvalConfig,
    out_dir: Path,
    overrides: dict[str, dict],
) -> dict:
    pack_over = overrides.get(pack.name, {})
    task_list = pack_over.get("tasks", pack.tasks)
    fewshot = int(pack_over.get("fewshot", pack.fewshot))
    batch_size = int(pack_over.get("batch_size", pack.batch_size))
    apply_chat_template = pack_over.get("apply_chat_template", pack.apply_chat_template)

    tok, pretrained_id = _load_tokenizer_and_pretrained(
        model_cfg.checkpoint, trust_remote_code=True
    )
    tok.backend_tokenizer.decoder = ByteLevel()

    hf_model, temp_dir = _build_lm(
        pretrained_id,
        model_cfg.peft_adapter,
        tok,
        model_cfg.device,
        model_cfg.dtype,
        bool(model_cfg.merge_lora),
    )

    genconf = hf_model.model.generation_config
    genconf.eos_token_id = tok.eos_token_id
    genconf.pad_token_id = tok.pad_token_id or tok.eos_token_id

    if model_cfg.merge_lora and isinstance(hf_model.model, PeftModel):
        hf_model._model = hf_model.model.merge_and_unload()

    try:
        results = lm_eval.simple_evaluate(
            model=hf_model,
            tasks=task_list,
            num_fewshot=fewshot,
            batch_size=batch_size,
            apply_chat_template=apply_chat_template,
            log_samples=True,
            write_out=True,
        )
    except TypeError:
        raise
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pack.name}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=_safe_json_encoder)

    return results
