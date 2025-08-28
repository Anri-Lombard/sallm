from __future__ import annotations
import logging
import json
from pathlib import Path
from typing import Dict

from sallm.config import ExperimentConfig
from sallm.evaluation.harness import evaluate_pack
from sallm.evaluation.registry import load_task_pack
from sallm.evaluation.config import TaskPack

logger = logging.getLogger(__name__)


def run(config: ExperimentConfig) -> None:
    assert (
        config.evaluation and config.eval_model
    ), "`evaluation` and `eval_model` blocks required."

    eval_cfg = config.evaluation
    model_cfg = config.eval_model
    out_root = Path(eval_cfg.output_dir)
    overrides = eval_cfg.overrides or {}

    def _evaluate_pack(pack: TaskPack):
        logger.info(
            f"Evaluating task-pack '{pack.name}' with {len(pack.tasks)} tasks …"
        )
        results: Dict = evaluate_pack(pack, model_cfg, out_root, overrides)
        logger.info(json.dumps(results["results"], indent=2))

    # 1) Static task packs
    if eval_cfg.task_packs:
        for pack_name in eval_cfg.task_packs:
            pack = load_task_pack(pack_name)
            _evaluate_pack(pack)

    # 2) Single dynamic generator
    elif eval_cfg.generator:
        gen = eval_cfg.generator
        lang = overrides.get("lang") or gen.get("lang")
        if not lang:
            raise ValueError(
                "Evaluation language not provided. Set evaluation.overrides.lang or generator.lang"
            )
        lang_map = gen.get("lang_map", {})
        lang_token = lang_map.get(lang, lang)
        name_template = gen["name_template"]
        prompt_ids = gen.get("prompt_ids") or list(
            range(1, int(gen.get("n_prompts", 5)) + 1)
        )
        tasks = [name_template.format(lang=lang_token, i=i) for i in prompt_ids]
        pack = TaskPack(
            name=f"{gen.get('name', 'pack')}_{lang}",
            tasks=tasks,
            fewshot=int(gen.get("fewshot", 0)),
            batch_size=int(gen.get("batch_size", 8)),
            apply_chat_template=bool(gen.get("apply_chat_template", True)),
            lm_eval_kwargs=gen.get("lm_eval_kwargs", {}),
        )
        _evaluate_pack(pack)

    # 3) Multiple dynamic generators (e.g., SA suite)
    elif eval_cfg.generators:
        for gen in eval_cfg.generators:
            lang = overrides.get("lang") or gen.get("lang")
            if not lang:
                raise ValueError(
                    "Evaluation language not provided. Set evaluation.overrides.lang or generators[i].lang"
                )
            lang_map = gen.get("lang_map", {})
            lang_token = lang_map.get(lang, lang)
            name_template = gen["name_template"]
            prompt_ids = gen.get("prompt_ids") or list(
                range(1, int(gen.get("n_prompts", 5)) + 1)
            )
            tasks = [name_template.format(lang=lang_token, i=i) for i in prompt_ids]
            pack = TaskPack(
                name=f"{gen.get('name', 'pack')}_{lang}",
                tasks=tasks,
                fewshot=int(gen.get("fewshot", 0)),
                batch_size=int(gen.get("batch_size", 8)),
                apply_chat_template=bool(gen.get("apply_chat_template", True)),
                lm_eval_kwargs=gen.get("lm_eval_kwargs", {}),
            )
            _evaluate_pack(pack)

    else:
        raise ValueError(
            "No evaluation.task_packs provided and no generator(s) configured."
        )

    logger.info("Evaluation done.")
