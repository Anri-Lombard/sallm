from __future__ import annotations
import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

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


def setup_logging(config: DictConfig) -> None:
    log_handlers = [logging.StreamHandler(sys.stdout)]

    log_file_path = None
    if config.training:
        log_file_path = config.training.get("log_file")

    if log_file_path:
        log_handlers.append(logging.FileHandler(log_file_path))
        print(f"Logging text output to: {log_file_path}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=log_handlers,
    )


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    unwrapped_cfg = cfg

    keys_in_cfg = list(cfg.keys())
    if (
        len(keys_in_cfg) == 1
        and keys_in_cfg[0] not in ExperimentConfig.__dataclass_fields__
    ):
        group_name = keys_in_cfg[0]
        logger.info(
            f"Detected nested config group '{group_name}'. Unwrapping configuration."
        )
        unwrapped_cfg = cfg[group_name]

    schema = OmegaConf.structured(ExperimentConfig)
    config = OmegaConf.merge(schema, unwrapped_cfg)

    setup_logging(config)

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
