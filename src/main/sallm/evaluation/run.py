from __future__ import annotations
import logging
import json
from pathlib import Path
from typing import Dict

from sallm.config import ExperimentConfig
from sallm.evaluation.harness import evaluate_pack
from sallm.evaluation.registry import load_task_pack

logger = logging.getLogger(__name__)


def run(config: ExperimentConfig) -> None:
    assert (
        config.evaluation and config.eval_model
    ), "`evaluation` and `eval_model` blocks required."

    eval_cfg = config.evaluation
    model_cfg = config.eval_model
    out_root = Path(eval_cfg.output_dir)
    overrides = eval_cfg.overrides or {}

    for pack_name in eval_cfg.task_packs:
        pack = load_task_pack(pack_name)
        logger.info(
            f"Evaluating task-pack '{pack_name}' with {len(pack.tasks)} tasks …"
        )
        results: Dict = evaluate_pack(pack, model_cfg, out_root, overrides)
        logger.info(json.dumps(results["results"], indent=2))

    logger.info("Evaluation done.")
