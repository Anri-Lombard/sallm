import logging
import math
from typing import Any

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

# TODO remove vebose comments
class PerLanguageLossCallback(TrainerCallback):

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        trainer: Trainer = kwargs.get("trainer")
        if not trainer or not trainer.eval_dataset:
            logger.warning(
                "Callback is exiting: Trainer or eval_dataset not found in kwargs."
            )
            return

        eval_dataset = trainer.eval_dataset

        if "lang" not in eval_dataset.features:
            logger.warning(
                "Callback is exiting: 'lang' column not found in the eval dataset."
            )
            return

        if not args.prediction_loss_only:
            logger.warning(
                "Callback is exiting: `prediction_loss_only` is not True. "
                "This callback is only designed to run in that mode."
            )
            return

        unique_languages = sorted(list(set(eval_dataset["lang"])))
        logger.info(
            f"Found {len(unique_languages)} unique languages for evaluation: {unique_languages}"
        )

        per_language_metrics = {}
        model = trainer.model
        model.eval()

        if trainer._signature_columns is None:
            logger.error(
                "Callback is exiting: Could not determine model signature columns from Trainer."
            )
            return
        model_args = trainer._signature_columns

        for lang in unique_languages:
            logger.info(f"--- Starting evaluation for language: '{lang}' ---")
            lang_dataset = eval_dataset.filter(
                lambda example: example["lang"] == lang, load_from_cache_file=False
            )

            if len(lang_dataset) == 0:
                logger.warning(
                    f"Skipping language '{lang}' as it has no samples in the eval set."
                )
                continue

            dataloader: DataLoader = trainer.get_eval_dataloader(lang_dataset)
            total_loss = 0.0
            num_samples = 0

            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
                    model_inputs = {k: v for k, v in batch.items() if k in model_args}

                    outputs = model(**model_inputs)
                    loss = outputs.loss
                    batch_size = batch["input_ids"].size(0)
                    total_loss += loss.item() * batch_size
                    num_samples += batch_size

            all_losses = trainer.accelerator.gather(
                torch.tensor(total_loss, device=args.device)
            )
            all_samples = trainer.accelerator.gather(
                torch.tensor(num_samples, device=args.device)
            )

            total_loss_agg = torch.sum(all_losses).item()
            total_samples_agg = torch.sum(all_samples).item()

            if state.is_world_process_zero:
                logger.info(
                    f"Aggregated stats for '{lang}': total_loss={total_loss_agg:.4f}, num_samples={total_samples_agg}"
                )

            if total_samples_agg > 0:
                avg_loss = total_loss_agg / total_samples_agg
                try:
                    perplexity = math.exp(avg_loss)
                except OverflowError:
                    perplexity = float("inf")

                if state.is_world_process_zero:
                    logger.info(
                        f"Calculated final metrics for '{lang}': loss={avg_loss:.4f}, perplexity={perplexity:.4f}"
                    )

                per_language_metrics[f"eval/{lang}_loss"] = avg_loss
                per_language_metrics[f"eval/{lang}_perplexity"] = perplexity
            else:
                logger.warning(
                    f"No samples found for language '{lang}' after distributed aggregation. Skipping metric calculation."
                )

        if state.is_world_process_zero:
            if per_language_metrics:
                logger.info(
                    "--- Final per-language metrics dictionary to be logged to wandb ---"
                )
                for key, value in per_language_metrics.items():
                    logger.info(f"  - {key}: {value:.4f}")
                wandb.log(per_language_metrics, step=state.global_step)
                logger.info("Successfully logged per-language metrics to wandb.")
            else:
                logger.warning(
                    "No per-language metrics were generated. Nothing to log to wandb."
                )

