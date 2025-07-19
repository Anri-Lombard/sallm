from typing import Any, Dict

import numpy as np
import wandb
from transformers import (
    EvalPrediction,
    TrainerCallback,
    TrainerControl,
    TrainingArguments,
    TrainerState,
)

from sallm.utils import compute_metrics


class PerLanguageEvalCallback(TrainerCallback):
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        trainer = kwargs.get("trainer")
        if not trainer:
            return

        eval_dataset = trainer.eval_dataset
        if "lang" not in eval_dataset.features:
            return

        predictions = trainer.predict(eval_dataset)
        logits, labels = predictions.predictions, predictions.label_ids

        languages = eval_dataset["lang"]
        unique_languages = set(languages)

        per_language_metrics: Dict[str, float] = {}

        for lang in unique_languages:
            mask = np.array(languages) == lang

            lang_logits = logits[mask]
            lang_labels = labels[mask]

            metrics = compute_metrics(
                EvalPrediction(predictions=lang_logits, label_ids=lang_labels)
            )

            for key, value in metrics.items():
                metric_name = f"eval/{lang}_{key}"
                per_language_metrics[metric_name] = value

        if state.is_world_process_zero:
            wandb.log(per_language_metrics, step=state.global_step)
