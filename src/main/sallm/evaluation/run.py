from __future__ import annotations

import json
import logging
from pathlib import Path

from sallm.config import ExperimentConfig, ModelEvalConfig, to_resolved_dict
from sallm.evaluation.harness import load_model_and_tokenizer, run_generation_task
from sallm.evaluation.lm_eval_runner import run_task_pack_evaluations

logger = logging.getLogger(__name__)


def _resolve_model_config(eval_model_cfg: ModelEvalConfig | dict) -> ModelEvalConfig:
    if isinstance(eval_model_cfg, ModelEvalConfig):
        return eval_model_cfg
    cfg_dict = to_resolved_dict(eval_model_cfg, name="eval_model")
    return ModelEvalConfig(**cfg_dict)


def run(config: ExperimentConfig) -> None:
    assert config.evaluation and config.eval_model, (
        "`evaluation` and `eval_model` blocks required."
    )

    eval_cfg = config.evaluation
    model_cfg = _resolve_model_config(config.eval_model)

    logger.info(
        "Using checkpoint: %s (peft_adapter=%s, merge_lora=%s)",
        model_cfg.checkpoint,
        model_cfg.peft_adapter,
        model_cfg.merge_lora,
    )

    has_generation = bool(eval_cfg.generation_tasks)
    has_task_packs = bool(eval_cfg.task_packs)

    if not has_generation and not has_task_packs:
        raise ValueError(
            "No evaluation tasks configured. Provide generation_tasks or task_packs."
        )

    out_root = Path(eval_cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []

    if has_generation:
        model, tokenizer = load_model_and_tokenizer(model_cfg)
        for task_cfg in eval_cfg.generation_tasks:
            logger.info(
                "Evaluating task '%s' on split '%s' (max_new_tokens=%s).",
                task_cfg.id,
                task_cfg.split,
                task_cfg.max_new_tokens,
            )
            task_out_dir = out_root / task_cfg.id
            summary = run_generation_task(task_cfg, model, tokenizer, task_out_dir)
            summaries.append(summary)
            logger.info(json.dumps(summary["metrics"], indent=2))

    if has_task_packs:
        overrides = eval_cfg.overrides
        if overrides:
            overrides = to_resolved_dict(overrides, name="evaluation.overrides")
        pack_summaries = run_task_pack_evaluations(
            list(eval_cfg.task_packs),
            model_cfg,
            out_root,
            overrides or None,
        )
        summaries.extend(pack_summaries)

    summary_path = out_root / "evaluation_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summaries, handle, indent=2)  # type: ignore[arg-type]

    logger.info("Evaluation done.")
