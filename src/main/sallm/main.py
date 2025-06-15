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
        "--mode",
        type=RunMode,
        required=True,
        choices=list(RunMode),
        help="The operational mode: 'train' for training/HPO, or future modes.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        nargs="?",
        const=True,
        default=False,
        help="Resume from the last checkpoint. Pass a path to resume from a specific one.",
    )
    cli_args = parser.parse_args()

    logger.info(f"Loading experiment configuration from: {cli_args.config_path}")
    config = load_experiment_config(cli_args.config_path)

    logger.info(f"Starting run in '{cli_args.mode.value}' mode.")

    if cli_args.mode == RunMode.TRAIN:
        run_train(config, cli_args)
    elif cli_args.mode == RunMode.FINETUNE:
        # TODO implement
        raise NotImplementedError("Finetune mode is not yet implemented.")
    elif cli_args.mode == RunMode.EVALUATE:
        # TODO implement
        raise NotImplementedError("Evaluate mode is not yet implemented.")


if __name__ == "__main__":
    main()
