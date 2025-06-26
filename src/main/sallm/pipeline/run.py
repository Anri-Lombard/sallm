from __future__ import annotations
import hydra

# from omegaconf import DictConfig

from sallm.config import ExperimentConfig
from sallm.fine_tune.run import run as run_ft
from sallm.evaluation.run import run as run_ev


def run(cfg: ExperimentConfig):
    pipe = cfg.pipeline
    for lang in pipe.languages:
        # TODO make absolute
        with hydra.initialize(config_path="../../conf"):
            ft_cfg = hydra.compose(
                config_name=pipe.finetune_base_cfg,
                overrides=[
                    f"model.init_checkpoint={pipe.base_checkpoint}",
                    f"dataset.subset={lang}",
                    f"wandb.name=ft-{lang}",
                ],
            )
            run_ft(hydra.utils.instantiate(ft_cfg))

        # TODO make absolute
        with hydra.initialize(config_path="../../conf"):
            ev_cfg = hydra.compose(
                config_name=pipe.eval_stub_cfg,
                overrides=[
                    "eval_model.checkpoint=???",  # TODO: get from ft_cfg
                    f"evaluation.task_packs=[masakhanews_{lang}]",
                ],
            )
            run_ev(hydra.utils.instantiate(ev_cfg))
