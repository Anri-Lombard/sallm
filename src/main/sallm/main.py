import logging
import os
from transformers import HfArgumentParser, TrainingArguments
from sallm.config import ScriptArguments, load_experiment_config
from sallm.training.run import run as run_train
from sallm.utils import RunMode

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args_cli = parser.parse_args_into_dataclasses()

    logger.info(
        f"Loading base experiment configuration from: {script_args.config_path}"
    )
    config = load_experiment_config(script_args.config_path)

    final_train_args_dict = config.training.copy()

    default_args = TrainingArguments(output_dir=".")
    cli_overrides = {
        key: value
        for key, value in training_args_cli.to_dict().items()
        if getattr(training_args_cli, key) != getattr(default_args, key)
    }

    if cli_overrides:
        logger.info("Overriding base config with CLI arguments:")
        for key, value in cli_overrides.items():
            logger.info(f"  - {key}: {value}")
        final_train_args_dict.update(cli_overrides)

    is_sweep = "WANDB_SWEEP_ID" in os.environ
    if not is_sweep and config.wandb.name:
        final_train_args_dict["run_name"] = config.wandb.name

    config.training = final_train_args_dict

    if script_args.wandb_run_id:
        logger.info(f"Received wandb run ID for resumption: {script_args.wandb_run_id}")
        config.wandb.id = script_args.wandb_run_id
        config.training["resume_from_checkpoint"] = True

    logger.info(f"Starting run in '{config.mode.value}' mode.")

    if config.mode == RunMode.TRAIN:
        run_train(config)
    elif config.mode == RunMode.FINETUNE:
        raise NotImplementedError("Finetune mode is not yet implemented.")
    elif config.mode == RunMode.EVALUATE:
        raise NotImplementedError("Evaluate mode is not yet implemented.")


if __name__ == "__main__":
    main()
