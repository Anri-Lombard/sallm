from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import lm_eval

from sallm.config import ModelEvalConfig
from sallm.evaluation.config import TaskPack

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


def _filter_tasks_by_lang(task_names: List[str]) -> List[str]:
    return [t for t in task_names if t.split("_")[-1] in SUPPORTED_LANGS]


def evaluate_pack(
    pack: TaskPack,
    model_cfg: ModelEvalConfig,
    out_dir: Path,
    overrides: Dict[str, Dict],
) -> Dict:
    pack_over = overrides.get(pack.name, {})
    task_list = pack_over.get("tasks", pack.tasks)
    task_list = _filter_tasks_by_lang(task_list)

    fewshot = pack_over.get("fewshot", pack.fewshot)
    batch_size = pack_over.get("batch_size", pack.batch_size)

    model_args = f"pretrained={model_cfg.checkpoint},dtype={model_cfg.dtype},device={model_cfg.device}"

    eval_kwargs = {
        "model": model_cfg.adapter,
        "model_args": model_args,
        "tasks": task_list,
        "batch_size": batch_size,
        "num_fewshot": fewshot,
        "verbosity": "ERROR",
    }
    eval_kwargs.update(pack.lm_eval_kwargs)

    results = lm_eval.evaluate(**eval_kwargs)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pack.name}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    return results
