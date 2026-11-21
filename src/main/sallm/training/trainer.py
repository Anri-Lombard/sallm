from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import datasets
import torch
import wandb
from torch.utils.data import DataLoader
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from trl import SFTTrainer

from sallm.training.general_validation import (
    GENERAL_PROCESSED_ROWS,
    GENERAL_SELECTION_PROTOCOL,
    GENERAL_TASKS,
    compute_equal_family_token_nll,
    validate_general_coverage,
)
from sallm.training.validation_artifacts import write_validation_artifact

logger = logging.getLogger(__name__)

# Transformers' Trainer.compute_loss inspects this mapping to apply causal label shift.
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.setdefault(
    "gated_deltanet",
    "GatedDeltaNetForCausalLM",
)

_NO_DECAY_SUFFIXES = ("A_log", "D")


def _filter_decay_parameter_names(
    model: Any,
    decay_parameter_names: Iterable[str],
) -> list[str]:
    """Exclude recurrent state parameters (A_log, D, _no_weight_decay) from decay."""
    decay_set = set(decay_parameter_names)
    excluded = set()
    for name, parameter in model.named_parameters():
        if name not in decay_set:
            continue
        if getattr(parameter, "_no_weight_decay", False):
            excluded.add(name)
            continue
        if any(
            name == suffix or name.endswith(f".{suffix}")
            for suffix in _NO_DECAY_SUFFIXES
        ):
            excluded.add(name)

    filtered = sorted(decay_set - excluded)
    if excluded:
        logger.info(
            "Excluded %d recurrent parameters from weight decay.", len(excluded)
        )
    return filtered


class CustomSFTTrainer(SFTTrainer):
    """SFTTrainer with per-language evaluation and General selection metrics."""

    def __init__(self, *args: Any, general_selection: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.general_selection = general_selection

    def get_decay_parameter_names(self, model) -> list[str]:
        decay_parameter_names = super().get_decay_parameter_names(model)
        return _filter_decay_parameter_names(model, decay_parameter_names)

    def _evaluate_general_selection(
        self,
        eval_dataset: datasets.Dataset,
        metric_key_prefix: str,
    ) -> dict[str, Any]:
        if int(getattr(self.args, "world_size", 1)) != 1:
            raise ValueError(
                "The frozen General selection contract currently requires one GPU."
            )
        coverage = validate_general_coverage(eval_dataset)
        model = self.model
        if model is None:
            raise ValueError("Trainer: General evaluation requires a model.")
        model_for_eval = cast(Any, model)
        model_for_eval.eval()
        if self._signature_columns is None:
            raise RuntimeError("Could not find model signature columns.")
        model_args = self._signature_columns

        start_time = time.time()
        summed_nll: dict[str, float] = {}
        valid_token_counts: dict[str, int] = {}
        evaluated_rows: dict[str, int] = {}
        for task_name in GENERAL_TASKS:
            task_dataset = eval_dataset.filter(
                lambda row, _task=task_name: row["task_name"] == _task,
                load_from_cache_file=False,
            )
            dataloader: DataLoader = self.get_eval_dataloader(cast(Any, task_dataset))
            task_nll = 0.0
            task_tokens = 0
            task_rows = 0
            with torch.no_grad():
                for batch_index, batch in enumerate(dataloader):
                    batch = self._prepare_inputs(batch)
                    labels = batch.get("labels")
                    if labels is None:
                        raise ValueError(
                            f"General {task_name} batch {batch_index} has no labels."
                        )
                    valid_tokens = int((labels[:, 1:] != -100).sum().item())
                    if valid_tokens <= 0:
                        raise ValueError(
                            f"General {task_name} batch {batch_index} is masked-only."
                        )
                    model_inputs = {
                        key: value for key, value in batch.items() if key in model_args
                    }
                    model_inputs["use_cache"] = False
                    loss = model_for_eval(**model_inputs).loss
                    loss_value = float(loss.item())
                    if not math.isfinite(loss_value):
                        raise ValueError(
                            f"General {task_name} batch {batch_index} has "
                            "non-finite loss."
                        )
                    task_nll += loss_value * valid_tokens
                    task_tokens += valid_tokens
                    task_rows += int(batch["input_ids"].shape[0])

            expected_rows = GENERAL_PROCESSED_ROWS[task_name]
            if task_rows != expected_rows:
                raise ValueError(
                    f"General {task_name} evaluated {task_rows} rows; "
                    f"expected {expected_rows}."
                )
            summed_nll[task_name] = task_nll
            valid_token_counts[task_name] = task_tokens
            evaluated_rows[task_name] = task_rows

        macro_nll, family_nll = compute_equal_family_token_nll(
            summed_nll,
            valid_token_counts,
        )
        runtime = time.time() - start_time
        metrics: dict[str, Any] = {
            f"{metric_key_prefix}_loss": macro_nll,
            f"{metric_key_prefix}_runtime": runtime,
            f"{metric_key_prefix}_samples_per_second": len(eval_dataset) / runtime,
        }
        for task_name in GENERAL_TASKS:
            metrics[f"{metric_key_prefix}/{task_name}_nll"] = family_nll[task_name]
            metrics[f"{metric_key_prefix}/{task_name}_valid_tokens"] = (
                valid_token_counts[task_name]
            )
            metrics[f"{metric_key_prefix}/{task_name}_rows"] = evaluated_rows[task_name]

        payload = {
            "protocol": GENERAL_SELECTION_PROTOCOL,
            "global_step": int(self.state.global_step),
            "epoch": self.state.epoch,
            "coverage": coverage,
            "summed_assistant_token_nll": summed_nll,
            "valid_assistant_token_counts": valid_token_counts,
            "family_nll": family_nll,
            "macro_nll": macro_nll,
        }
        write_validation_artifact(
            output_dir=str(self.args.output_dir),
            subdir="general",
            step=int(self.state.global_step),
            payload=payload,
            label="General",
            save_to_wandb=self.is_world_process_zero(),
        )
        if self.is_world_process_zero() and getattr(wandb, "run", None) is not None:
            wandb.log(metrics, step=self.state.global_step)

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics,
        )
        return metrics

    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, Any]:
        resolved_eval_dataset: Any = (
            eval_dataset if eval_dataset is not None else self.eval_dataset
        )
        if resolved_eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        if self.general_selection:
            if not isinstance(resolved_eval_dataset, datasets.Dataset):
                raise TypeError("General selection requires a Hugging Face Dataset.")
            return self._evaluate_general_selection(
                resolved_eval_dataset,
                metric_key_prefix,
            )

        if not isinstance(resolved_eval_dataset, datasets.Dataset):
            return super().evaluate(
                eval_dataset=resolved_eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        eval_dataset = resolved_eval_dataset

        if "lang" not in eval_dataset.features:
            logger.warning("Custom `evaluate` is exiting: 'lang' column not found.")
            return super().evaluate(
                eval_dataset=cast(Any, eval_dataset),
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        start_time = time.time()
        unique_languages = sorted(
            [lang for lang in set(eval_dataset["lang"]) if lang is not None]
        )
        per_language_metrics_for_wandb: dict[str, float] = {}
        grand_total_loss = 0.0
        grand_total_samples = 0

        model = self.model
        if model is None:
            raise ValueError("Trainer: evaluation requires a model.")
        model_for_eval = cast(Any, model)
        model_for_eval.eval()

        if self._signature_columns is None:
            raise RuntimeError("Could not find model signature columns.")
        model_args = self._signature_columns

        for lang in unique_languages:
            lang_dataset = eval_dataset.filter(
                lambda x, _lang=lang: x["lang"] == _lang, load_from_cache_file=False
            )
            if len(lang_dataset) == 0:
                continue

            dataloader: DataLoader = self.get_eval_dataloader(cast(Any, lang_dataset))
            total_loss, num_samples = 0.0, 0
            nan_count = 0
            last_batch_idx = -1

            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    last_batch_idx = batch_idx
                    batch = self._prepare_inputs(batch)

                    if "labels" in batch and not (batch["labels"] != -100).any():
                        logger.debug(
                            "lang=%s batch=%d: all labels masked, skipping.",
                            lang,
                            batch_idx,
                        )
                        continue

                    model_inputs = {k: v for k, v in batch.items() if k in model_args}
                    model_inputs["use_cache"] = False
                    loss_val = float(model_for_eval(**model_inputs).loss.item())
                    batch_size = int(batch["input_ids"].size(0))

                    if math.isnan(loss_val):
                        nan_count += 1
                        logger.warning(
                            "lang=%s batch=%d: NaN loss despite valid labels; "
                            "skipping.",
                            lang,
                            batch_idx,
                        )
                        continue

                    total_loss += loss_val * batch_size
                    num_samples += batch_size

            if nan_count > 0:
                logger.warning(
                    "lang=%s: %d/%d batches had NaN loss.",
                    lang,
                    nan_count,
                    last_batch_idx + 1,
                )

            all_losses = self.accelerator.gather(
                torch.tensor(total_loss, device=self.args.device)
            )
            all_samples = self.accelerator.gather(
                torch.tensor(num_samples, device=self.args.device)
            )

            total_loss_agg = float(torch.sum(all_losses).item())
            total_samples_agg = float(torch.sum(all_samples).item())

            grand_total_loss += total_loss_agg
            grand_total_samples += int(total_samples_agg)

            if total_samples_agg > 0:
                avg_loss = total_loss_agg / total_samples_agg
                try:
                    perplexity = math.exp(avg_loss)
                except OverflowError:
                    perplexity = float("inf")

                per_language_metrics_for_wandb[f"eval/{lang}_loss"] = avg_loss
                per_language_metrics_for_wandb[f"eval/{lang}_perplexity"] = perplexity
                logger.debug(
                    "lang=%s avg_loss=%.4f samples=%d",
                    lang,
                    avg_loss,
                    int(total_samples_agg),
                )

        if self.is_world_process_zero() and per_language_metrics_for_wandb:
            wandb.log(per_language_metrics_for_wandb, step=self.state.global_step)

        metrics_to_return: dict[str, Any] = {}
        if grand_total_samples > 0:
            metrics_to_return[f"{metric_key_prefix}_loss"] = (
                grand_total_loss / grand_total_samples
            )
        else:
            logger.warning("No valid eval samples found; omitting eval_loss.")

        runtime = time.time() - start_time
        metrics_to_return[f"{metric_key_prefix}_runtime"] = runtime
        metrics_to_return[f"{metric_key_prefix}_samples_per_second"] = (
            grand_total_samples / runtime
        )

        self.log(metrics_to_return)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics_to_return
        )
        return metrics_to_return

    def save_model(self, output_dir=None, _internal_call=False):
        if not self.args.should_save:
            return
        out = str(output_dir or self.args.output_dir)
        Path(out).mkdir(parents=True, exist_ok=True)

        cast(Any, self.model).save_pretrained(out, safe_serialization=False)

        processing_class = getattr(self, "processing_class", None)
        if processing_class is not None:
            cast(Any, processing_class).save_pretrained(out)
