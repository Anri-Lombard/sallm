import logging
import random

import torch
import wandb
from datasets import Dataset
from evaluate import load as _eval_load
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

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
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

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

        rouge = _eval_load("rouge")
        bleu = _eval_load("bleu")
        chrf = _eval_load("chrf")

        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        device = model.device

        if "lang" in self.eval_dataset.features:
            unique_languages = sorted(list(set(self.eval_dataset["lang"])))
        else:
            unique_languages = [None]

        log_dict: dict[str, float] = {}

        for lang in unique_languages:
            if lang is None:
                lang_ds = self.eval_dataset
                lang_key = "all"
            else:
                lang_ds = self.eval_dataset.filter(
                    lambda x, _lang=lang: x["lang"] == _lang, load_from_cache_file=False
                )
                lang_key = lang

            if len(lang_ds) == 0:
                continue

            preds: list[str] = []
            refs: list[str] = []

            model.eval()
            with torch.no_grad():
                for sample in lang_ds:
                    messages = sample["messages"]
                    prompt_messages = messages[:-1]
                    gold_completion = messages[-1]["content"].lstrip()

                    inputs = self.tokenizer.apply_chat_template(
                        prompt_messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(device)

                    gen_ids = model.generate(
                        inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                        eos_token_id=eos_id,
                    )

                    generated_ids = gen_ids[0][inputs.shape[-1] :]
                    generated_completion = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    )

                    preds.append(generated_completion)
                    refs.append(gold_completion)

            if preds and refs:
                metrics = rouge.compute(predictions=preds, references=refs)
                r1 = metrics.get("rouge1")
                r2 = metrics.get("rouge2")
                rl = metrics.get("rougeL")

                if r1 is not None:
                    log_dict[f"eval/{lang_key}_rouge1"] = float(r1)
                if r2 is not None:
                    log_dict[f"eval/{lang_key}_rouge2"] = float(r2)
                if rl is not None:
                    log_dict[f"eval/{lang_key}_rougeL"] = float(rl)

                bleu_metrics = bleu.compute(
                    predictions=preds, references=[[r] for r in refs]
                )
                bleu_score = bleu_metrics.get("bleu") or bleu_metrics.get("score")
                if bleu_score is not None:
                    log_dict[f"eval/{lang_key}_bleu"] = float(bleu_score)

                chrf_metrics = chrf.compute(predictions=preds, references=refs)
                chrf_score = chrf_metrics.get("score") or chrf_metrics.get("chrf")
                if chrf_score is not None:
                    log_dict[f"eval/{lang_key}_chrf"] = float(chrf_score)

        if log_dict:
            wandb.log(log_dict, step=state.global_step)
