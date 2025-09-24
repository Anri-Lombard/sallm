from __future__ import annotations

import json
import logging
from pathlib import Path

from omegaconf import OmegaConf

from sallm.config import ExperimentConfig, ModelEvalConfig
from sallm.evaluation.harness import evaluate_pack
from sallm.evaluation.registry import load_task_pack

logger = logging.getLogger(__name__)


def run(config: ExperimentConfig) -> None:
    assert (
        config.evaluation and config.eval_model
    ), "`evaluation` and `eval_model` blocks required."

    eval_cfg = config.evaluation
    eval_model_cfg = config.eval_model
    if isinstance(eval_model_cfg, ModelEvalConfig):
        model_cfg = eval_model_cfg
    else:
        cfg_dict = OmegaConf.to_container(eval_model_cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise TypeError("eval_model configuration must resolve to a mapping")
        model_cfg = ModelEvalConfig(**cfg_dict)
    out_root = Path(eval_cfg.output_dir)
    overrides = eval_cfg.overrides or {}

    for pack_name in eval_cfg.task_packs:
        pack = load_task_pack(pack_name)
        logger.info(
            "Evaluating task-pack '%s' with %d tasks ...",
            pack_name,
            len(pack.tasks),
        )
        results: dict = evaluate_pack(pack, model_cfg, out_root, overrides)
        logger.info(json.dumps(results["results"], indent=2))

    logger.info("Evaluation done.")
