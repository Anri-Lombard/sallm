from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers.trainer_utils import is_main_process
from transformers.utils import logging as hf_logging

from sallm.config import ExperimentConfig
from sallm.evaluation.run import run as run_eval
from sallm.fine_tune.run import run as run_fine_tune
from sallm.training.run import run as run_train
from sallm.utils import RunMode

# Suppress noisy torch.compile warnings
warnings.filterwarnings(
    "ignore",
    message="Skipping serialization of skipfiles_inline_module_allowlist",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Graph break due to unsupported builtin causal_conv1d_cuda",
    category=UserWarning,
)

logger = logging.getLogger(__name__)


def _is_main_process() -> bool:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    return (local_rank == -1) or is_main_process(local_rank)


def setup_logging(config: DictConfig) -> None:
    is_main = bool(OmegaConf.select(config, "runtime.is_main"))

    if is_main:
        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

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
    if not isinstance(config, DictConfig):
        raise TypeError("Hydra config must resolve to a mapping.")

    is_main = _is_main_process()
    with open_dict(config):
        config["runtime"] = {"is_main": is_main}

    setup_logging(config)

    if (
        len(keys_in_cfg) == 1
        and keys_in_cfg[0] not in ExperimentConfig.__dataclass_fields__
    ):
        logger.info(
            "Detected nested config group '%s'. Unwrapping configuration.",
            keys_in_cfg[0],
        )

    logger.info(f"Run mode: {config.mode.value}")

    if config.mode == RunMode.TRAIN:
        run_train(cast(ExperimentConfig, config))
    elif config.mode == RunMode.FINETUNE:
        run_fine_tune(cast(ExperimentConfig, config))
    elif config.mode == RunMode.EVALUATE:
        run_eval(cast(ExperimentConfig, config))
    else:
        raise ValueError(f"Unsupported mode {config.mode!r}")


if __name__ == "__main__":
    main()
