from __future__ import annotations
import logging

import hydra
from omegaconf import DictConfig

from sallm.config import ExperimentConfig
from sallm.utils import RunMode

from sallm.training.run import run as run_train
from sallm.fine_tune.run import run as run_fine_tune
from sallm.evaluation.run import run as run_eval
from sallm.pipeline.run import run as run_orch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


# TODO: make path absolute
@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config: ExperimentConfig = hydra.utils.instantiate(cfg)

    logger.info(f"Run mode: {config.mode.value}")

    if config.mode == RunMode.TRAIN:
        run_train(config)
    elif config.mode == RunMode.FINETUNE:
        run_fine_tune(config)
    elif config.mode == RunMode.EVALUATE:
        run_eval(config)
    elif config.mode == RunMode.ORCHESTRATE:
        run_orch(config)
    else:
        raise ValueError(f"Unsupported mode {config.mode!r}")


if __name__ == "__main__":
    main()
