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
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens

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
                gen_ids = model.generate(
                    inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    use_cache=False,
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
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.evaluator = GenerationEvaluator(
            tokenizer,
            max_new_tokens=max_new_tokens,
            max_samples_per_lang=max_samples_per_lang,
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
            wandb.log(result.metrics, step=state.global_step)


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
