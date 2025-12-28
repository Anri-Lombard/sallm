import logging
import random

import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from sallm.config import DecodingConfig
from sallm.evaluation.classification_metrics import ClassificationEvaluator
from sallm.evaluation.generation_metrics import GenerationEvaluator

logger = logging.getLogger(__name__)


# TODO: add models to model.py


class ShowCompletionsCallback(TrainerCallback):
    def __init__(
        self,
        eval_dataset: Dataset,
        tokenizer: AutoTokenizer,
        num_samples: int = 5,
        max_new_tokens: int = 100,
        decoding: DecodingConfig | None = None,
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.decoding_config = DecodingConfig.from_any(decoding)
        if (
            self.decoding_config.num_return_sequences is not None
            and self.decoding_config.num_return_sequences != 1
        ):
            raise ValueError(
                "ShowCompletionsCallback requires a single return sequence."
            )

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        model: AutoModelForCausalLM | None = kwargs.get("model")
        if model is None:
            logger.warning(
                "ShowCompletionsCallback: `model` not found in kwargs. Skipping."
            )
            return

        ds_len = len(self.eval_dataset)
        indices: list[int] = (
            random.sample(range(ds_len), self.num_samples)
            if ds_len > self.num_samples
            else list(range(ds_len))
        )
        samples = self.eval_dataset.select(indices)

        logger.info(
            f"\n--- Showing {len(indices)} Generated Examples "
            f"after Epoch {int(state.epoch):d} ---"
        )

        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        device = model.device

        for i, sample in enumerate(samples, start=1):
            messages: list[dict[str, str]] = sample["messages"]

            prompt_messages = messages[:-1]
            gold_completion = messages[-1]["content"].lstrip()

            inputs = self.tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                generate_kwargs = self.decoding_config.to_generate_kwargs()
                generate_kwargs["max_new_tokens"] = self.max_new_tokens
                generate_kwargs["pad_token_id"] = pad_id
                generate_kwargs["eos_token_id"] = eos_id
                generate_kwargs.setdefault("use_cache", False)
                gen_ids = model.generate(
                    inputs,
                    **generate_kwargs,
                )

            generated_ids = gen_ids[0][inputs.shape[-1] :]
            generated_completion = self.tokenizer.decode(
                generated_ids,
                # skip_special_tokens=True,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            prompt_text_for_log = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            logger.info(f"\n--- Sample {i}/{len(indices)} ---")
            logger.info(f"Prompt (as seen by model):\n{prompt_text_for_log}")
            logger.info(f"\n--> Generated Completion: '{generated_completion}'")
            logger.info(f"--> Gold Completion:      '{gold_completion}'")
            logger.info("-" * 40)

        logger.info("--- End of Generated Examples ---\n")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        if not logs:
            logger.info("ShowCompletionsCallback: no evaluation logs provided.")
            return

        eval_items = {k: v for k, v in logs.items()}
        if "eval_loss" in eval_items:
            logger.info(f"Evaluation loss: {eval_items.get('eval_loss')}")
        if "eval_perplexity" in eval_items:
            logger.info(f"Evaluation perplexity: {eval_items.get('eval_perplexity')}")

        logger.info(
            "Evaluation metrics:\n"
            + "\n".join(f"{k}: {v}" for k, v in eval_items.items())
        )


class GenerationMetricsCallback(TrainerCallback):
    def __init__(
        self,
        eval_dataset: Dataset,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 64,
        max_samples_per_lang: int | None = 64,
        decoding: DecodingConfig | None = None,
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.decoding_config = DecodingConfig.from_any(decoding)
        self.evaluator = GenerationEvaluator(
            tokenizer,
            max_new_tokens=max_new_tokens,
            max_samples_per_lang=max_samples_per_lang,
            decoding=self.decoding_config,
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        model = kwargs.get("model")
        if model is None:
            logger.warning(
                "GenerationMetricsCallback: `model` not found in kwargs. Skipping."
            )
            return

        world_size = getattr(args, "world_size", 1)
        result = self.evaluator.evaluate(
            model,
            self.eval_dataset,
            world_size=world_size,
            metric_prefix="eval",
            collect_examples=False,
        )

        if result.metrics:
            wandb.log(result.metrics)


class ClassificationMetricsCallback(TrainerCallback):
    """Callback that computes accuracy metrics for classification tasks."""

    def __init__(
        self,
        eval_dataset: Dataset,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 32,
        max_samples_per_lang: int | None = 256,
        decoding: DecodingConfig | None = None,
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.decoding_config = DecodingConfig.from_any(decoding)
        self.evaluator = ClassificationEvaluator(
            tokenizer,
            max_new_tokens=max_new_tokens,
            max_samples_per_lang=max_samples_per_lang,
            decoding=self.decoding_config,
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        model = kwargs.get("model")
        if model is None:
            logger.warning(
                "ClassificationMetricsCallback: `model` not found in kwargs. Skipping."
            )
            return

        metrics = self.evaluator.evaluate(
            model,
            self.eval_dataset,
            metric_prefix="classification",
        )

        if metrics:
            wandb.log(metrics)


class MultiTaskMetricsCallback(TrainerCallback):
    """Callback that evaluates mixed datasets with task-appropriate metrics.

    Expects the eval_dataset to have '_task' and '_task_type' columns added by
    the mix loader. Applies different evaluators based on task type:
    - classification/named_entity_recognition/pos_tagging → accuracy
    - instruction (afrihg) → ROUGE metrics
    - instruction (t2x) → chrF/BLEU metrics
    """

    CLASSIFICATION_TASKS = {"classification", "named_entity_recognition", "pos_tagging"}
    GENERATION_HEADLINE_TASKS = {"afrihg"}
    GENERATION_TRANSLATION_TASKS = {"t2x"}

    def __init__(
        self,
        eval_dataset: Dataset,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 64,
        max_samples_per_task: int | None = 128,
        decoding: DecodingConfig | None = None,
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_samples_per_task = max_samples_per_task
        self.decoding_config = DecodingConfig.from_any(decoding)

        self.classification_evaluator = ClassificationEvaluator(
            tokenizer,
            max_new_tokens=32,
            max_samples_per_lang=max_samples_per_task,
            decoding=self.decoding_config,
        )
        self.generation_evaluator = GenerationEvaluator(
            tokenizer,
            max_new_tokens=max_new_tokens,
            max_samples_per_lang=max_samples_per_task,
            decoding=self.decoding_config,
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        model = kwargs.get("model")
        if model is None:
            logger.warning(
                "MultiTaskMetricsCallback: `model` not found in kwargs. Skipping."
            )
            return

        if "_task" not in self.eval_dataset.column_names:
            logger.warning(
                "MultiTaskMetricsCallback: eval_dataset missing '_task' column. "
                "Skipping per-task evaluation."
            )
            return

        unique_tasks = sorted(set(self.eval_dataset["_task"]))
        all_metrics: dict[str, float] = {}
        task_scores: dict[str, float] = {}

        for task_name in unique_tasks:
            task_subset = self.eval_dataset.filter(
                lambda ex, t=task_name: ex.get("_task") == t,
                load_from_cache_file=False,
            )

            if len(task_subset) == 0:
                continue

            task_type = task_subset[0].get("_task_type", "instruction")

            if task_type in self.CLASSIFICATION_TASKS:
                metrics = self.classification_evaluator.evaluate(
                    model,
                    task_subset,
                    metric_prefix=f"multitask/{task_name}",
                )
                all_metrics.update(metrics)
                accuracy_key = f"multitask/{task_name}/all_accuracy"
                if accuracy_key in metrics:
                    task_scores[task_name] = metrics[accuracy_key]

            elif task_name in self.GENERATION_HEADLINE_TASKS:
                result = self.generation_evaluator.evaluate(
                    model,
                    task_subset,
                    metric_prefix=f"multitask/{task_name}",
                    collect_examples=False,
                )
                all_metrics.update(result.metrics)
                rougeL_key = f"multitask/{task_name}/all_rougeL"
                if rougeL_key in result.metrics:
                    task_scores[task_name] = result.metrics[rougeL_key]

            elif task_name in self.GENERATION_TRANSLATION_TASKS:
                result = self.generation_evaluator.evaluate(
                    model,
                    task_subset,
                    metric_prefix=f"multitask/{task_name}",
                    collect_examples=False,
                )
                all_metrics.update(result.metrics)
                chrf_key = f"multitask/{task_name}/all_chrf"
                if chrf_key in result.metrics:
                    task_scores[task_name] = result.metrics[chrf_key] / 100.0

            else:
                result = self.generation_evaluator.evaluate(
                    model,
                    task_subset,
                    metric_prefix=f"multitask/{task_name}",
                    collect_examples=False,
                )
                all_metrics.update(result.metrics)

        if task_scores:
            all_metrics["multitask/aggregate_score"] = sum(task_scores.values()) / len(
                task_scores
            )

        if all_metrics:
            wandb.log(all_metrics)
            logger.info(
                f"MultiTask Evaluation | tasks={len(unique_tasks)} | "
                f"aggregate={all_metrics.get('multitask/aggregate_score', 'N/A'):.4f}"
            )


class EnsureStaticGraphCallback(TrainerCallback):
    def __init__(self) -> None:
        self._applied = False

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._applied or not args.gradient_checkpointing or args.world_size <= 1:
            return control

        model = kwargs.get("model")
        if model is None or not hasattr(model, "_set_static_graph"):
            return control

        model._set_static_graph()
        self._applied = True
        if state.is_world_process_zero:
            logger.info(
                "Enabled DDP static graph mode to stabilize gradient "
                "checkpointing with PEFT."
            )
            find_unused = getattr(model, "find_unused_parameters", None)
            if find_unused:
                logger.info(
                    "DDP `find_unused_parameters` is still enabled; expect "
                    "an informational warning from PyTorch while static-graph "
                    "mode remains in effect."
                )

        return control
