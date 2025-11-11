import logging
import math
import time
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.utils import is_datasets_available
from trl import SFTTrainer

if is_datasets_available():
    import datasets

logger = logging.getLogger(__name__)


# TODO rename to something more appropriate since different models will use
# different trainers
class CustomTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: datasets.Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        if "lang" not in eval_dataset.features:
            logger.warning("Custom `evaluate` is exiting: 'lang' column not found.")
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        start_time = time.time()

        unique_languages = sorted(list(set(eval_dataset["lang"])))
        per_language_metrics_for_wandb = {}
        # TODO remove samples?
        grand_total_loss = 0.0
        grand_total_samples = 0

        model = self._wrap_model(self.model, training=False, dataloader=None)
        model.eval()

        if self._signature_columns is None:
            raise RuntimeError("Could not find model signature columns.")
        model_args = self._signature_columns

        for lang in unique_languages:
            lang_dataset = eval_dataset.filter(
                lambda x, _lang=lang: x["lang"] == _lang, load_from_cache_file=False
            )
            if len(lang_dataset) == 0:
                continue

            dataloader: DataLoader = self.get_eval_dataloader(lang_dataset)
            total_loss, num_samples = 0.0, 0

            with torch.no_grad():
                for batch in dataloader:
                    batch = self._prepare_inputs(batch)
                    model_inputs = {k: v for k, v in batch.items() if k in model_args}
                    outputs = model(**model_inputs)
                    loss = outputs.loss
                    batch_size = batch["input_ids"].size(0)
                    total_loss += loss.item() * batch_size
                    num_samples += batch_size

            all_losses = self.accelerator.gather(
                torch.tensor(total_loss, device=self.args.device)
            )
            all_samples = self.accelerator.gather(
                torch.tensor(num_samples, device=self.args.device)
            )

            total_loss_agg = torch.sum(all_losses).item()
            total_samples_agg = torch.sum(all_samples).item()

            grand_total_loss += total_loss_agg
            grand_total_samples += total_samples_agg

            if total_samples_agg > 0:
                avg_loss = total_loss_agg / total_samples_agg
                try:
                    perplexity = math.exp(avg_loss)
                except OverflowError:
                    perplexity = float("inf")

                per_language_metrics_for_wandb[f"eval/{lang}_loss"] = avg_loss
                per_language_metrics_for_wandb[f"eval/{lang}_perplexity"] = perplexity

        if self.is_world_process_zero() and per_language_metrics_for_wandb:
            wandb.log(per_language_metrics_for_wandb, step=self.state.global_step)

        metrics_to_return = {}
        if grand_total_samples > 0:
            overall_loss = grand_total_loss / grand_total_samples
            metrics_to_return[f"{metric_key_prefix}_loss"] = overall_loss

        runtime = time.time() - start_time
        metrics_to_return[f"{metric_key_prefix}_runtime"] = runtime
        metrics_to_return[f"{metric_key_prefix}_samples_per_second"] = (
            grand_total_samples / runtime
        )

        self.log(metrics_to_return)

        return metrics_to_return

    def save_model(self, output_dir=None, _internal_call=False):
        out = output_dir or self.args.output_dir
        Path(out).mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(out, safe_serialization=False)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(out)


class CustomSFTTrainer(SFTTrainer):
    """SFTTrainer with per-language evaluation metrics for foundation model training."""

    def evaluate(
        self,
        eval_dataset: datasets.Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        if "lang" not in eval_dataset.features:
            logger.warning("Custom `evaluate` is exiting: 'lang' column not found.")
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        start_time = time.time()

        unique_languages = sorted(list(set(eval_dataset["lang"])))
        per_language_metrics_for_wandb = {}
        grand_total_loss = 0.0
        grand_total_samples = 0

        model = self._wrap_model(self.model, training=False, dataloader=None)
        model.eval()

        if self._signature_columns is None:
            raise RuntimeError("Could not find model signature columns.")
        model_args = self._signature_columns

        for lang in unique_languages:
            lang_dataset = eval_dataset.filter(
                lambda x, _lang=lang: x["lang"] == _lang, load_from_cache_file=False
            )
            if len(lang_dataset) == 0:
                continue

            dataloader: DataLoader = self.get_eval_dataloader(lang_dataset)
            total_loss, num_samples = 0.0, 0

            with torch.no_grad():
                for batch in dataloader:
                    batch = self._prepare_inputs(batch)
                    model_inputs = {k: v for k, v in batch.items() if k in model_args}
                    outputs = model(**model_inputs)
                    loss = outputs.loss
                    batch_size = batch["input_ids"].size(0)
                    total_loss += loss.item() * batch_size
                    num_samples += batch_size

            all_losses = self.accelerator.gather(
                torch.tensor(total_loss, device=self.args.device)
            )
            all_samples = self.accelerator.gather(
                torch.tensor(num_samples, device=self.args.device)
            )

            total_loss_agg = torch.sum(all_losses).item()
            total_samples_agg = torch.sum(all_samples).item()

            grand_total_loss += total_loss_agg
            grand_total_samples += total_samples_agg

            if total_samples_agg > 0:
                avg_loss = total_loss_agg / total_samples_agg
                try:
                    perplexity = math.exp(avg_loss)
                except OverflowError:
                    perplexity = float("inf")

                per_language_metrics_for_wandb[f"eval/{lang}_loss"] = avg_loss
                per_language_metrics_for_wandb[f"eval/{lang}_perplexity"] = perplexity

        if self.is_world_process_zero() and per_language_metrics_for_wandb:
            wandb.log(per_language_metrics_for_wandb, step=self.state.global_step)

        metrics_to_return = {}
        if grand_total_samples > 0:
            overall_loss = grand_total_loss / grand_total_samples
            metrics_to_return[f"{metric_key_prefix}_loss"] = overall_loss

        runtime = time.time() - start_time
        metrics_to_return[f"{metric_key_prefix}_runtime"] = runtime
        metrics_to_return[f"{metric_key_prefix}_samples_per_second"] = (
            grand_total_samples / runtime
        )

        self.log(metrics_to_return)

        return metrics_to_return

    def save_model(self, output_dir=None, _internal_call=False):
        out = output_dir or self.args.output_dir
        Path(out).mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(out, safe_serialization=False)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(out)
