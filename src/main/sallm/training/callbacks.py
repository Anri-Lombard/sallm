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
def _broadcast_metrics_from_rank0(
    *,
    local_metrics: dict[str, float],
    is_world_process_zero: bool,
) -> dict[str, float]:
    """Share metrics computed on rank 0 with all distributed workers."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return local_metrics

    payload = local_metrics if is_world_process_zero else {}
    container = [payload]
    torch.distributed.broadcast_object_list(container, src=0)

    broadcasted = container[0]
    if isinstance(broadcasted, dict):
        return broadcasted
    return {}


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
        trainer_metrics: dict[str, float] = {}
        wandb_metrics: dict[str, float] = {}
        model = kwargs.get("model")
        if model is None:
            logger.warning(
                "GenerationMetricsCallback: `model` not found in kwargs. Skipping."
            )
        elif state.is_world_process_zero:
            world_size = getattr(args, "world_size", 1)
            result = self.evaluator.evaluate(
                model,
                self.eval_dataset,
                world_size=world_size,
                metric_prefix="eval",
                collect_examples=False,
            )

            if result.metrics:
                trainer_metrics = dict(result.metrics)
                for key, value in list(result.metrics.items()):
                    # Expose underscore aliases for Trainer metric selection.
                    if "/" in key:
                        trainer_metrics[key.replace("/", "_")] = value
                wandb_metrics = dict(result.metrics)

        trainer_metrics = _broadcast_metrics_from_rank0(
            local_metrics=trainer_metrics,
            is_world_process_zero=state.is_world_process_zero,
        )
        callback_metrics = kwargs.get("metrics")
        if trainer_metrics and isinstance(callback_metrics, dict):
            callback_metrics.update(trainer_metrics)
        if state.is_world_process_zero and wandb_metrics:
            wandb.log(wandb_metrics)


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
        trainer_metrics: dict[str, float] = {}
        wandb_metrics: dict[str, float] = {}
        model = kwargs.get("model")
        if model is None:
            logger.warning(
                "ClassificationMetricsCallback: `model` not found in kwargs. Skipping."
            )
        elif state.is_world_process_zero:
            metrics = self.evaluator.evaluate(
                model,
                self.eval_dataset,
                metric_prefix="classification",
            )

            if metrics:
                trainer_metrics = dict(metrics)
                for key, value in list(metrics.items()):
                    # Trainer expects eval_* when selecting best model metrics.
                    trainer_metrics[f"eval_{key}"] = value
                    if "/" in key:
                        trainer_metrics[f"eval_{key.replace('/', '_')}"] = value
                wandb_metrics = dict(metrics)

        trainer_metrics = _broadcast_metrics_from_rank0(
            local_metrics=trainer_metrics,
            is_world_process_zero=state.is_world_process_zero,
        )
        callback_metrics = kwargs.get("metrics")
        if trainer_metrics and isinstance(callback_metrics, dict):
            callback_metrics.update(trainer_metrics)
        if state.is_world_process_zero and wandb_metrics:
            wandb.log(wandb_metrics)


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
