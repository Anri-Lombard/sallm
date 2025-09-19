from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import lm_eval
import numpy as np
import torch
from lm_eval.models.huggingface import HFLM
from peft import PeftModel
from tokenizers.decoders import ByteLevel
from transformers import AutoTokenizer

from sallm.config import ModelEvalConfig
from sallm.evaluation.config import TaskPack


def _safe_json_encoder(obj: Any) -> Any:
    if isinstance(obj, np.integer) or isinstance(obj, np.floating):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.ndim else obj.item()
    if isinstance(obj, torch.dtype):
        return str(obj)
    return str(obj)


def evaluate_pack(
    pack: TaskPack,
    model_cfg: ModelEvalConfig,
    out_dir: Path,
    overrides: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pack_over = overrides.get(pack.name, {})
    task_list = pack_over.get("tasks", pack.tasks)
    fewshot = int(pack_over.get("fewshot", pack.fewshot))
    batch_size = int(pack_over.get("batch_size", pack.batch_size))
    apply_chat_template = pack_over.get("apply_chat_template", pack.apply_chat_template)

    tok = cast(
        AutoTokenizer,
        AutoTokenizer.from_pretrained(model_cfg.checkpoint, trust_remote_code=True),
    )
    tok_any = cast(Any, tok)
    tok_any.backend_tokenizer.decoder = ByteLevel()

    hf_model = HFLM(
        pretrained=model_cfg.checkpoint,
        tokenizer=tok,
        peft=model_cfg.peft_adapter,
        device=model_cfg.device,
        dtype=model_cfg.dtype,
        trust_remote_code=True,
    )

    genconf = hf_model.model.generation_config
    eos_token_id = getattr(tok_any, "eos_token_id", None)
    pad_token_id = getattr(tok_any, "pad_token_id", None)
    genconf.eos_token_id = eos_token_id
    genconf.pad_token_id = pad_token_id or eos_token_id

    if model_cfg.merge_lora and isinstance(hf_model.model, PeftModel):
        hf_model._model = hf_model.model.merge_and_unload()

    try:
        results_any = lm_eval.simple_evaluate(
            model=hf_model,
            tasks=task_list,
            num_fewshot=fewshot,
            batch_size=batch_size,
            apply_chat_template=apply_chat_template,
            log_samples=True,
            write_out=True,
        )
        results = cast(dict[str, Any], results_any)
    except TypeError:
        raise

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pack.name}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=_safe_json_encoder)

    return results
