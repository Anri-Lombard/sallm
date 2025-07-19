from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

import lm_eval
from lm_eval.models.huggingface import HFLM
from peft import PeftModel

from sallm.config import ModelEvalConfig
from sallm.evaluation.config import TaskPack


def safe_json_encoder(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, torch.Tensor):
        return obj.tolist() if obj.ndim else obj.item()
    if isinstance(obj, torch.dtype):
        return str(obj)

    return str(obj)


SUPPORTED_LANGS: List[str] = [
    "afr",
    "xho",
    "zul",
    "nso",
    "sot",
    "ssw",
    "tsn",
    "tso",
    "ven",
    "eng",
    "nbl",
]


def _filter_tasks_by_lang(task_names: list[str]) -> list[str]:
    def has_lang(tokens):
        return any(tok in SUPPORTED_LANGS for tok in tokens)

    return [t for t in task_names if has_lang(t.split("_"))]


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

    hf_model = HFLM(
        pretrained=model_cfg.checkpoint,
        peft=model_cfg.peft_adapter,
        device=model_cfg.device,
        dtype=model_cfg.dtype,
        trust_remote_code=True,
    )

    if model_cfg.merge_lora and isinstance(hf_model.model, PeftModel):
        hf_model._model = hf_model.model.merge_and_unload()

    try:
        results = lm_eval.simple_evaluate(
            model=hf_model,
            tasks=task_list,
            num_fewshot=fewshot,
            batch_size=batch_size,
        )
    except TypeError as e:
        raise ValueError(
            f"{e}. Did you pass a dict instead of a list/str for `tasks`?"
        ) from None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pack.name}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=safe_json_encoder)

    return results
