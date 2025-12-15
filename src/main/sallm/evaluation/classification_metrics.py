from __future__ import annotations

import logging
import textwrap
from collections import defaultdict

import torch
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedModel

from sallm.config import DecodingConfig

logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """Evaluator for classification tasks that computes accuracy metrics."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 32,
        max_samples_per_lang: int | None = 256,
        decoding: DecodingConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_samples_per_lang = max_samples_per_lang
        self.decoding_config = DecodingConfig.from_any(decoding)

    def evaluate(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        metric_prefix: str = "classification",
    ) -> dict[str, float]:
        """Evaluate classification accuracy on the dataset.

        Args:
            model: The model to evaluate
            dataset: Dataset with 'messages' column in conversation format
            metric_prefix: Prefix for metric names

        Returns:
            Dictionary of metrics including accuracy and per-class stats
        """
        fallback_template = self._get_fallback_template()
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        if hasattr(model, "eval"):
            model.eval()
        device = getattr(model, "device", torch.device("cpu"))

        lang_column_present = "lang" in dataset.features
        unique_languages: list[str | None]
        if lang_column_present:
            raw_langs = set(dataset["lang"])
            str_langs = sorted({x for x in raw_langs if isinstance(x, str) and x})
            unique_languages = str_langs if str_langs else [None]
        else:
            unique_languages = [None]

        metrics: dict[str, float] = {}

        for lang in unique_languages:
            if lang is None:
                lang_dataset = dataset
                lang_key = "all"
            else:
                lang_key = lang
                lang_dataset = dataset.filter(
                    lambda ex, _lang=lang: ex.get("lang") == _lang,
                    load_from_cache_file=False,
                )

            if len(lang_dataset) == 0:
                continue

            capped_dataset = self._cap_dataset(lang_dataset, lang_key)
            lang_metrics = self._evaluate_subset(
                model, capped_dataset, device, pad_id, eos_id, fallback_template
            )

            for name, value in lang_metrics.items():
                metrics[f"{metric_prefix}/{lang_key}_{name}"] = value

        return metrics

    def _evaluate_subset(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        device: torch.device,
        pad_id: int | None,
        eos_id: int | None,
        fallback_template: str | None,
    ) -> dict[str, float]:
        """Evaluate a single language subset."""
        correct = 0
        total = 0
        class_correct: dict[str, int] = defaultdict(int)
        class_total: dict[str, int] = defaultdict(int)

        with torch.no_grad():
            for sample in dataset:
                messages: list[dict[str, str]] = sample["messages"]
                if not messages or len(messages) < 2:
                    continue

                prompt_messages = messages[:-1]
                gold_label = messages[-1]["content"].strip()

                pred_label = self._generate_prediction(
                    model,
                    prompt_messages,
                    device,
                    pad_id,
                    eos_id,
                    fallback_template,
                )

                is_correct = self._labels_match(pred_label, gold_label)
                if is_correct:
                    correct += 1
                    class_correct[gold_label] += 1
                total += 1
                class_total[gold_label] += 1

        if total == 0:
            return {"accuracy": 0.0}

        accuracy = correct / total
        metrics: dict[str, float] = {"accuracy": accuracy}

        if len(class_total) > 1:
            per_class_acc = []
            for cls in class_total:
                if class_total[cls] > 0:
                    cls_acc = class_correct[cls] / class_total[cls]
                    per_class_acc.append(cls_acc)
            if per_class_acc:
                metrics["macro_accuracy"] = sum(per_class_acc) / len(per_class_acc)

        return metrics

    def _generate_prediction(
        self,
        model: PreTrainedModel,
        prompt_messages: list[dict[str, str]],
        device: torch.device,
        pad_id: int | None,
        eos_id: int | None,
        fallback_template: str | None,
    ) -> str:
        """Generate a prediction for the given prompt."""
        template_kwargs: dict[str, str] = {}
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=fallback_template,
            **template_kwargs,
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
        ).to(device)

        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask")
        if attn is None:
            attn = (input_ids != pad_id).long()

        generate_kwargs = self.decoding_config.to_generate_kwargs()
        generate_kwargs["max_new_tokens"] = self.max_new_tokens
        generate_kwargs["pad_token_id"] = pad_id
        generate_kwargs["eos_token_id"] = eos_id
        generate_kwargs.setdefault("use_cache", True)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            **generate_kwargs,
        )

        prompt_len = input_ids.shape[1]
        gen_seq = outputs[0][prompt_len:]
        generated_text = self.tokenizer.decode(
            gen_seq,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return self._extract_label(generated_text)

    def _extract_label(self, text: str) -> str:
        """Extract the classification label from generated text."""
        cleaned = text.strip()
        if "\n" in cleaned:
            cleaned = cleaned.split("\n")[0].strip()
        trailing_markers = (
            "[EOS]",
            "[EOT]",
            "</s>",
            "<|endoftext|>",
            "<|im_end|>",
            "<|eot_id|>",
        )
        for marker in trailing_markers:
            if cleaned.endswith(marker):
                cleaned = cleaned[: -len(marker)].strip()
        return cleaned

    def _labels_match(self, pred: str, gold: str) -> bool:
        """Check if predicted and gold labels match (case-insensitive)."""
        return pred.lower().strip() == gold.lower().strip()

    def _cap_dataset(self, dataset: Dataset, lang_key: str) -> Dataset:
        """Cap dataset to max_samples_per_lang."""
        if self.max_samples_per_lang is None:
            return dataset
        if len(dataset) <= self.max_samples_per_lang:
            return dataset
        indices = list(range(self.max_samples_per_lang))
        return dataset.select(indices)

    def _get_fallback_template(self) -> str | None:
        """Get fallback chat template if tokenizer doesn't have one."""
        if getattr(self.tokenizer, "chat_template", None) is not None:
            return None
        return textwrap.dedent(
            """
            {%- if system_message %}
            <|system|>
            {{ system_message }}{{ eos_token }}
            {%- endif %}
            {%- for message in messages %}
                {%- if message['role'] == 'user' %}
                    <|user|>
                    {{ message['content'] }}{{ eos_token }}
                {%- elif message['role'] == 'assistant' %}
                    {%- generation -%}
                    <|assistant|>
                    {{ message['content'] }}{{ eos_token }}
                    {%- endgeneration -%}
                {%- endif %}
            {%- endfor %}
            {%- if add_generation_prompt %}<|assistant|>{%- endif %}
            """
        ).lstrip()
