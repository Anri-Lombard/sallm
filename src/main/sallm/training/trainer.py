import logging
import math
import time
from pathlib import Path
from typing import Any, cast

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.utils import is_datasets_available

if is_datasets_available():
    pass

logger = logging.getLogger(__name__)


# TODO rename to something more appropriate since different models will use
# different trainers
class CustomTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Any | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        base_trainer = cast(Trainer, self)
        trainer_any = cast(Any, self)
        ds_candidate = (
            eval_dataset
            if eval_dataset is not None
            else getattr(self, "eval_dataset", None)
        )
        if ds_candidate is None:
            return cast(
                dict[str, float],
                Trainer.evaluate(
                    base_trainer,
                    eval_dataset=eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix,
                ),
            )

        if not hasattr(ds_candidate, "features") or "lang" not in ds_candidate.features:
            return cast(
                dict[str, float],
                Trainer.evaluate(
                    base_trainer,
                    eval_dataset=ds_candidate,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix,
                ),
            )

        try:
            start_time = time.time()
            unique_languages = sorted(list(set(ds_candidate["lang"])))
            per_language_metrics: dict[str, float] = {}
            total_loss_all = 0.0
            total_samples_all = 0

            model_obj = getattr(trainer_any, "model", None)
            if model_obj is None:
                return cast(
                    dict[str, float],
                    Trainer.evaluate(
                        base_trainer,
                        eval_dataset=ds_candidate,
                        ignore_keys=ignore_keys,
                        metric_key_prefix=metric_key_prefix,
                    ),
                )
            model: Any = model_obj
            if hasattr(model, "eval"):
                model.eval()

            signature_columns = getattr(trainer_any, "_signature_columns", None)
            if signature_columns is None:
                return cast(
                    dict[str, float],
                    Trainer.evaluate(
                        base_trainer,
                        eval_dataset=ds_candidate,
                        ignore_keys=ignore_keys,
                        metric_key_prefix=metric_key_prefix,
                    ),
                )

            for lang in unique_languages:
                lang_ds = ds_candidate.filter(
                    lambda x, _l=lang: x["lang"] == _l, load_from_cache_file=False
                )
                if len(lang_ds) == 0:
                    continue
                dataloader: DataLoader[Any] = trainer_any.get_eval_dataloader(lang_ds)
                lang_total_loss = 0.0
                lang_samples = 0
                with torch.no_grad():
                    for batch in dataloader:
                        batch_inputs = trainer_any._prepare_inputs(batch)
                        inputs_for_model = {
                            k: v
                            for k, v in batch_inputs.items()
                            if k in signature_columns
                        }
                        outputs = model(**inputs_for_model)
                        loss_val = outputs.loss
                        bsz = batch_inputs["input_ids"].size(0)
                        lang_total_loss += loss_val.item() * bsz
                        lang_samples += bsz
                total_loss_all += lang_total_loss
                total_samples_all += lang_samples
                if lang_samples > 0:
                    avg_loss = lang_total_loss / lang_samples
                    try:
                        perplexity = math.exp(avg_loss)
                    except OverflowError:
                        perplexity = float("inf")
                    per_language_metrics[f"eval/{lang}_loss"] = avg_loss
                    per_language_metrics[f"eval/{lang}_perplexity"] = perplexity

            state_obj = getattr(trainer_any, "state", None)
            if state_obj is not None and hasattr(trainer_any, "is_world_process_zero"):
                if trainer_any.is_world_process_zero() and per_language_metrics:
                    wandb.log(per_language_metrics, step=state_obj.global_step)

            result: dict[str, float] = {}
            if total_samples_all > 0:
                overall_loss = total_loss_all / total_samples_all
                result[f"{metric_key_prefix}_loss"] = overall_loss
                elapsed = time.time() - start_time
                result[f"{metric_key_prefix}_runtime"] = elapsed
                if elapsed > 0:
                    result[f"{metric_key_prefix}_samples_per_second"] = (
                        total_samples_all / elapsed
                    )
            Trainer.log(base_trainer, result)
            return result
        except Exception:
            return cast(
                dict[str, float],
                Trainer.evaluate(
                    base_trainer,
                    eval_dataset=ds_candidate,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix,
                ),
            )

    def save_model(
        self, output_dir: str | None = None, _internal_call: bool = False
    ) -> None:
        trainer_any = cast(Any, self)
        out = output_dir or trainer_any.args.output_dir
        Path(out).mkdir(parents=True, exist_ok=True)

        model = getattr(trainer_any, "model", None)
        if model is not None:
            model.save_pretrained(out, safe_serialization=False)

        tokenizer = getattr(trainer_any, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(out)
