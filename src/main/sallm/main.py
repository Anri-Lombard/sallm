from __future__ import annotations
import logging

from transformers import HfArgumentParser, TrainingArguments

from sallm.config import ScriptArguments, load_experiment_config
from sallm.utils import RunMode

from sallm.training.run import run as run_train
from sallm.fine_tune.run import run as run_fine_tune
from sallm.evaluation.run import run as run_eval
from sallm.pipeline.run import run as run_orch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_cli = parser.parse_args_into_dataclasses()

    cfg_path = script_args.config_path
    logger.info(f"Loading config: {cfg_path}")
    config = load_experiment_config(cfg_path)

    if config.training is None:
        config.training = {}
    default_args = TrainingArguments(output_dir=".")
    cli_overrides = {
        k: v
        for k, v in training_cli.to_dict().items()
        if getattr(training_cli, k) != getattr(default_args, k)
    }
    if cli_overrides:
        logger.info("CLI overrides → config.training:")
        for k, v in cli_overrides.items():
            logger.info(f"  • {k}: {v}")
        config.training.update(cli_overrides)

    if script_args.wandb_run_id:
        logger.info(f"Resuming WANDB run id {script_args.wandb_run_id}")
        if config.wandb is None:
            raise ValueError("`wandb` block missing in config.")
        config.wandb.id = script_args.wandb_run_id
        config.training["resume_from_checkpoint"] = True

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
