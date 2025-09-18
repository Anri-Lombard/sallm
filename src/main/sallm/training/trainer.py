import logging
import math
import time
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.utils import is_datasets_available

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
        ds = (
            eval_dataset
            if eval_dataset is not None
            else getattr(self, "eval_dataset", None)
        )
        if ds is None:
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        if "lang" not in ds.features:
            return super().evaluate(
                eval_dataset=ds,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        try:
            start_time = time.time()
            unique_languages = sorted(list(set(ds["lang"])))
            per_language_metrics: dict[str, float] = {}
            total_loss_all = 0.0
            total_samples_all = 0

            if not hasattr(self, "model"):
                return super().evaluate(
                    eval_dataset=ds,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix,
                )
            model = self.model
            model.eval()

            if (
                not hasattr(self, "_signature_columns")
                or self._signature_columns is None
            ):
                return super().evaluate(
                    eval_dataset=ds,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix,
                )
            model_args = self._signature_columns

            for lang in unique_languages:
                lang_ds = ds.filter(
                    lambda x, _l=lang: x["lang"] == _l, load_from_cache_file=False
                )
                if len(lang_ds) == 0:
                    continue
                dataloader: DataLoader = self.get_eval_dataloader(lang_ds)
                lang_total_loss = 0.0
                lang_samples = 0
                with torch.no_grad():
                    for batch in dataloader:
                        batch_inputs = self._prepare_inputs(batch)
                        inputs_for_model = {
                            k: v for k, v in batch_inputs.items() if k in model_args
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

            if hasattr(self, "is_world_process_zero") and self.is_world_process_zero():
                if per_language_metrics:
                    wandb.log(per_language_metrics, step=self.state.global_step)

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
            self.log(result)
            return result
        except Exception:  # fall back to default behavior
            return super().evaluate(
                eval_dataset=ds,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

    def save_model(self, output_dir=None, _internal_call=False):
        out = output_dir or self.args.output_dir
        Path(out).mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(out, safe_serialization=False)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(out)
