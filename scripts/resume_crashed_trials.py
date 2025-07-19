# TODO make this script part of src/?
import argparse
import logging
import subprocess
import sys
from pathlib import Path

import wandb

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def resume_crashed_trials(sweep_id: str, config_path: str) -> None:
    try:
        api = wandb.Api()
        sweep = api.sweep(sweep_id)
        sweep_entity = sweep.entity
        sweep_project = sweep.project
    except wandb.errors.CommError as e:
        logger.error(
            f"Could not connect to wandb API or find sweep '{sweep_id}'. Error: {e}"
        )
        sys.exit(1)

    logger.info(
        f"Scanning sweep '{sweep.name}' in project '{sweep_entity}/{sweep_project}' for crashed runs."
    )

    for run in sweep.runs:
        if run.state in ["crashed", "failed"]:
            logger.warning(f"Found a '{run.state}' trial: {run.name} (ID: {run.id}).")
            logger.info(
                f"Submitting a new SLURM job to resume this trial from its last checkpoint."
            )

            resume_script_path = Path(__file__).parent / "train_resumed_trial.sh"

            try:
                subprocess.run(
                    [
                        "sbatch",
                        str(resume_script_path),
                        config_path,
                        run.id,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info(
                    f"Successfully submitted resumption job for run ID: {run.id}."
                )
                sys.exit(0)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to submit SLURM job for run {run.id}.")
                logger.error(f"sbatch stderr: {e.stderr}")
                sys.exit(1)

    logger.info("No crashed trials found in the sweep.")
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wandb sweep crash recovery helper.")
    parser.add_argument(
        "--sweep_id",
        type=str,
        required=True,
        help="The wandb sweep ID (e.g., 'entity/project/sweep_id').",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the main experiment config file.",
    )
    args = parser.parse_args()

    resume_crashed_trials(args.sweep_id, args.config_path)
