import argparse
import logging

from sallm.config import load_experiment_config
from sallm.training.run import run as run_train
from sallm.utils import RunMode

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Modular Language Model Training Framework."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the main YAML experiment config file.",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="Wandb run ID to resume a specific crashed trial.",
    )
    cli_args = parser.parse_args()

    logger.info(f"Loading experiment configuration from: {cli_args.config_path}")
    config = load_experiment_config(cli_args.config_path)

    if cli_args.wandb_run_id:
        logger.info(f"Received wandb run ID for resumption: {cli_args.wandb_run_id}")
        config.wandb.id = cli_args.wandb_run_id

    logger.info(f"Starting run in '{config.mode.value}' mode.")

    if config.mode == RunMode.TRAIN:
        run_train(config)
    elif config.mode == RunMode.FINETUNE:
        # TODO implement
        raise NotImplementedError("Finetune mode is not yet implemented.")
    elif config.mode == RunMode.EVALUATE:
        # TODO implement
        raise NotImplementedError("Evaluate mode is not yet implemented.")


if __name__ == "__main__":
    main()