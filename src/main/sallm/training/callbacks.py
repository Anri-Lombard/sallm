import logging
import random
from typing import List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class ShowCompletionsCallback(TrainerCallback):
    def __init__(
        self,
        eval_dataset: Dataset,
        tokenizer: AutoTokenizer,
        num_samples: int = 5,
        max_new_tokens: int = 6,
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens

    def on_epoch_end(  # noqa: D401
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
        indices: List[int] = (
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
        pad_tok = (
            self.tokenizer.convert_ids_to_tokens(pad_id) if pad_id is not None else None
        )
        eos_tok = self.tokenizer.convert_ids_to_tokens(eos_id)

        device = model.device

        for i, sample in enumerate(samples, start=1):
            prompt_text: str = sample["prompt"]
            gold_completion: str = sample["completion"].lstrip()

            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(device)
            inputs.pop("token_type_ids", None)

            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )

            generated_ids = gen_ids[0][inputs["input_ids"].shape[-1] :]
            generated_completion = self.tokenizer.decode(
                generated_ids
                # generated_ids, skip_special_tokens=True
            ).strip()

            gold_tokens = self.tokenizer.tokenize(" " + gold_completion)
            gold_ids = self.tokenizer.convert_tokens_to_ids(gold_tokens)

            gen_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)
            gen_ids_list = generated_ids.tolist()

            contains_pad = pad_id is not None and pad_id in gen_ids_list
            contains_eos = eos_id in gen_ids_list

            logger.info(f"\n--- Sample {i}/{len(indices)} ---")
            logger.info(f"Prompt:\n{prompt_text}")

            logger.info(f"\n--> Generated Completion: '{generated_completion}'")
            logger.info(f"    Generated tokens: {gen_tokens}")
            logger.info(f"    Generated IDs:    {gen_ids_list}")
            logger.info(f"    Contains PAD ({pad_tok}): {contains_pad}")
            logger.info(f"    Contains EOS ({eos_tok}): {contains_eos}")

            logger.info(f"\n--> Gold Completion:      '{gold_completion}'")
            logger.info(f"    Gold tokens: {gold_tokens}")
            logger.info(f"    Gold IDs:    {gold_ids}")
            logger.info("-" * 40)

        logger.info("--- End of Generated Examples ---\n")
