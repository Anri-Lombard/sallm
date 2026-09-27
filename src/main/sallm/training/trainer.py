import hashlib
import json
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
from transformers import Trainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from trl import SFTTrainer

from sallm.training.general_validation import (
    GENERAL_PROCESSED_ROWS,
    GENERAL_SELECTION_PROTOCOL,
    GENERAL_TASKS,
    compute_equal_family_token_nll,
    validate_general_coverage,
)

logger = logging.getLogger(__name__)


MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.setdefault(
    "gated_deltanet",
    "GatedDeltaNetForCausalLM",
)


_MAMBA_NO_DECAY_SUFFIXES = ("A_log", "D")


def _filter_decay_parameter_names(
    model,
    decay_parameter_names: Iterable[str],
) -> list[str]:
    """Exclude parameters that should not receive weight decay in Mamba blocks."""
    decay_set = set(decay_parameter_names)
    mamba_no_decay = set()
    for name, parameter in model.named_parameters():
        if name not in decay_set:
            continue
        if getattr(parameter, "_no_weight_decay", False):
            mamba_no_decay.add(name)
            continue
        if any(
            name == suffix or name.endswith(f".{suffix}")
            for suffix in _MAMBA_NO_DECAY_SUFFIXES
        ):
            mamba_no_decay.add(name)

    filtered = sorted(decay_set - mamba_no_decay)
    if mamba_no_decay:
        logger.info(
            "Excluded %d Mamba parameters from weight decay.",
            len(mamba_no_decay),
        )
    return filtered


# TODO rename to something more appropriate since different models will use
# different trainers
class CustomTrainer(Trainer):
    def get_decay_parameter_names(self, model) -> list[str]:
        decay_parameter_names = super().get_decay_parameter_names(model)
        return _filter_decay_parameter_names(model, decay_parameter_names)

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
        per_language_metrics_for_wandb = {}
        # TODO remove samples?
        grand_total_loss = 0.0
        grand_total_samples = 0

        # Use the prepared model directly (same as super().evaluate())
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

            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    batch = self._prepare_inputs(batch)

                    # Skip batches where all labels are masked (ignore_index = -100)
                    if "labels" in batch:
                        labels = batch["labels"]
                        valid_labels = (labels != -100).any()
                        if not valid_labels:
                            # All labels are masked, skip this batch
                            if self.is_world_process_zero():
                                print(
                                    f"DEBUG: lang={lang}, batch {batch_idx}: "
                                    "all labels masked, skipping",
                                    flush=True,
                                )
                            continue

                    model_inputs = {k: v for k, v in batch.items() if k in model_args}
                    model_inputs["use_cache"] = False  # Prevent Mamba2Cache in outputs
                    outputs = model_for_eval(**model_inputs)
                    loss = outputs.loss
                    loss_val = loss.item()
                    batch_size = batch["input_ids"].size(0)

                    if math.isnan(loss_val):
                        # This shouldn't happen now, but log if it does
                        nan_count += 1
                        if self.is_world_process_zero():
                            print(
                                f"WARNING: lang={lang}, batch {batch_idx}: "
                                "NAN loss despite valid labels! Skipping.",
                                flush=True,
                            )
                        continue

                    total_loss += loss_val * batch_size
                    num_samples += batch_size

            if self.is_world_process_zero() and nan_count > 0:
                msg = f"DEBUG: lang={lang}: {nan_count}/{batch_idx + 1} "
                msg += f"batches had nan loss, total_loss={total_loss}, "
                msg += f"num_samples={num_samples}"
                print(msg, flush=True)

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
                # Debug logging
                if self.is_world_process_zero():
                    msg = f"DEBUG: lang={lang}, avg_loss={avg_loss:.4f}, "
                    msg += f"samples={total_samples_agg}, "
                    msg += f"grand_total_loss_so_far={grand_total_loss:.4f}, "
                    msg += f"grand_total_samples_so_far={grand_total_samples}"
                    print(msg, flush=True)

        if self.is_world_process_zero() and per_language_metrics_for_wandb:
            wandb.log(per_language_metrics_for_wandb, step=self.state.global_step)
            msg = f"DEBUG: Logged {len(per_language_metrics_for_wandb)} "
            msg += "per-language metrics to wandb"
            print(msg, flush=True)

        metrics_to_return = {}
        if grand_total_samples > 0:
            overall_loss = grand_total_loss / grand_total_samples
            if self.is_world_process_zero():
                msg = "DEBUG: Computing overall eval_loss: "
                msg += f"{grand_total_loss:.4f} / {grand_total_samples} = "
                msg += f"{overall_loss}"
                print(msg, flush=True)
            metrics_to_return[f"{metric_key_prefix}_loss"] = overall_loss
        else:
            if self.is_world_process_zero():
                print(
                    "DEBUG: grand_total_samples is 0! Cannot compute eval_loss",
                    flush=True,
                )

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


class CustomSFTTrainer(SFTTrainer):
    """SFTTrainer with per-language evaluation metrics for foundation model training."""

    def __init__(self, *args: Any, general_selection: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Select checkpoints by equal-family assistant-token NLL on General data.
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
        encoded = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        artifact_dir = (
            Path(str(self.args.output_dir)) / "validation_artifacts" / "general"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"step-{int(self.state.global_step):08d}.json"
        path.write_bytes(encoded)
        digest = hashlib.sha256(encoded).hexdigest()
        path.with_suffix(".json.sha256").write_text(
            f"{digest}  {path.name}\n",
            encoding="utf-8",
        )
        logger.info("Saved General validation artifact %s (%s)", path, digest)
        if self.is_world_process_zero() and getattr(wandb, "run", None) is not None:
            wandb.log(metrics, step=self.state.global_step)
            wandb.save(str(path), policy="now")

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
        per_language_metrics_for_wandb = {}
        grand_total_loss = 0.0
        grand_total_samples = 0

        # Use the prepared model directly (same as super().evaluate())
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

            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    batch = self._prepare_inputs(batch)

                    # Skip batches where all labels are masked (ignore_index = -100)
                    if "labels" in batch:
                        labels = batch["labels"]
                        valid_labels = (labels != -100).any()
                        if not valid_labels:
                            # All labels are masked, skip this batch
                            if self.is_world_process_zero():
                                print(
                                    f"DEBUG: lang={lang}, batch {batch_idx}: "
                                    "all labels masked, skipping",
                                    flush=True,
                                )
                            continue

                    model_inputs = {k: v for k, v in batch.items() if k in model_args}
                    model_inputs["use_cache"] = False  # Prevent Mamba2Cache in outputs
                    outputs = model_for_eval(**model_inputs)
                    loss = outputs.loss
                    loss_val = loss.item()
                    batch_size = batch["input_ids"].size(0)

                    if math.isnan(loss_val):
                        # This shouldn't happen now, but log if it does
                        nan_count += 1
                        if self.is_world_process_zero():
                            print(
                                f"WARNING: lang={lang}, batch {batch_idx}: "
                                "NAN loss despite valid labels! Skipping.",
                                flush=True,
                            )
                        continue

                    total_loss += loss_val * batch_size
                    num_samples += batch_size

            if self.is_world_process_zero() and nan_count > 0:
                msg = f"DEBUG: lang={lang}: {nan_count}/{batch_idx + 1} "
                msg += f"batches had nan loss, total_loss={total_loss}, "
                msg += f"num_samples={num_samples}"
                print(msg, flush=True)

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
                # Debug logging
                if self.is_world_process_zero():
                    msg = f"DEBUG: lang={lang}, avg_loss={avg_loss:.4f}, "
                    msg += f"samples={total_samples_agg}, "
                    msg += f"grand_total_loss_so_far={grand_total_loss:.4f}, "
                    msg += f"grand_total_samples_so_far={grand_total_samples}"
                    print(msg, flush=True)

        if self.is_world_process_zero() and per_language_metrics_for_wandb:
            wandb.log(per_language_metrics_for_wandb, step=self.state.global_step)
            msg = f"DEBUG: Logged {len(per_language_metrics_for_wandb)} "
            msg += "per-language metrics to wandb"
            print(msg, flush=True)

        metrics_to_return = {}
        if grand_total_samples > 0:
            overall_loss = grand_total_loss / grand_total_samples
            if self.is_world_process_zero():
                msg = "DEBUG: Computing overall eval_loss: "
                msg += f"{grand_total_loss:.4f} / {grand_total_samples} = "
                msg += f"{overall_loss}"
                print(msg, flush=True)
            metrics_to_return[f"{metric_key_prefix}_loss"] = overall_loss
        else:
            if self.is_world_process_zero():
                print(
                    "DEBUG: grand_total_samples is 0! Cannot compute eval_loss",
                    flush=True,
                )

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
