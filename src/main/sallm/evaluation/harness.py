from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from peft import PeftModel
from transformers import AutoTokenizer
from tokenizers.decoders import ByteLevel

from sallm.config import ModelEvalConfig
from sallm.evaluation.config import TaskPack


def _safe_json_encoder(obj):
    if isinstance(obj, (np.integer, np.floating)):
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
    overrides: Dict[str, Dict],
) -> Dict:
    pack_over = overrides.get(pack.name, {})
    task_list = pack_over.get("tasks", pack.tasks)
    fewshot = pack_over.get("fewshot", pack.fewshot)
    batch_size = pack_over.get("batch_size", pack.batch_size)
    apply_chat_template = pack_over.get("apply_chat_template", pack.apply_chat_template)

    tok = AutoTokenizer.from_pretrained(model_cfg.checkpoint, trust_remote_code=True)
    tok.backend_tokenizer.decoder = ByteLevel()

    hf_model = HFLM(
        pretrained=model_cfg.checkpoint,
        tokenizer=tok,
        peft=model_cfg.peft_adapter,
        device=model_cfg.device,
        dtype=model_cfg.dtype,
        trust_remote_code=True,
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
    except TypeError as e:
        raise ValueError(
            f"{e}. Did you pass a dict instead of a list/str for `tasks`?"
        ) from None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pack.name}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=_safe_json_encoder)

    return results
