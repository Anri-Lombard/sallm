from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from peft import PeftModel
from transformers import AutoTokenizer

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


_SUPPORTED_LANGS: List[str] = [
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
        return any(tok in _SUPPORTED_LANGS for tok in tokens)

    return [t for t in task_names if has_lang(t.split("_"))]


class ChatHFLM(HFLM):
    def _wrap_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def generate(
        self,
        requests,
        max_length=512,
        temperature=0.0,
        top_p=1.0,
        **gen_kwargs,
    ):
        if isinstance(requests, str):
            requests = [requests]

        wrapped = [self._wrap_prompt(p) for p in requests]
        return super().generate(
            wrapped,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            **gen_kwargs,
        )


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

    _ = AutoTokenizer.from_pretrained(model_cfg.checkpoint, trust_remote_code=True)

    hf_model = ChatHFLM(
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
        json.dump(results, f, indent=2, default=_safe_json_encoder)

    return results
