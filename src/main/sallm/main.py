from __future__ import annotations
import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from sallm.config import ExperimentConfig
from sallm.utils import RunMode

from sallm.training.run import run as run_train
from sallm.fine_tune.run import run as run_fine_tune
from sallm.evaluation.run import run as run_eval
from sallm.pipeline.run import run as run_orch

import os
from transformers.trainer_utils import is_main_process
from transformers.utils import logging as hf_logging

logger = logging.getLogger(__name__)


def _is_main_process() -> bool:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    return (local_rank == -1) or is_main_process(local_rank)


def setup_logging(config: DictConfig) -> None:
    is_main = bool(OmegaConf.select(config, "runtime.is_main"))

    if is_main:
        handlers = [logging.StreamHandler(sys.stdout)]

        log_file_path = None
        if config.training:
            log_file_path = config.training.get("log_file")
        if log_file_path:
            handlers.append(logging.FileHandler(log_file_path))

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s — %(levelname)s — %(message)s",
            handlers=handlers,
            force=True,
        )
    else:
        logging.basicConfig(handlers=[logging.NullHandler()], force=True)
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.disable(logging.CRITICAL)

        hf_logging.set_verbosity_error()
        os.environ.setdefault("TQDM_DISABLE", "1")


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    unwrapped_cfg = cfg

    keys_in_cfg = list(cfg.keys())
    if (
        len(keys_in_cfg) == 1
        and keys_in_cfg[0] not in ExperimentConfig.__dataclass_fields__
    ):
        group_name = keys_in_cfg[0]
        unwrapped_cfg = cfg[group_name]

    schema = OmegaConf.structured(ExperimentConfig)
    config = OmegaConf.merge(schema, unwrapped_cfg)

    is_main = _is_main_process()
    with open_dict(config):
        config["runtime"] = {"is_main": is_main}

    setup_logging(config)

    if (
        len(keys_in_cfg) == 1
        and keys_in_cfg[0] not in ExperimentConfig.__dataclass_fields__
    ):
        logger.info(
            f"Detected nested config group '{keys_in_cfg[0]}'. Unwrapping configuration."
        )

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
