from __future__ import annotations

import logging
import textwrap
from collections import defaultdict
from enum import Enum

import torch
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, PreTrainedModel

from sallm.config import DecodingConfig
from sallm.templates import registry as tmpl

logger = logging.getLogger(__name__)


class ChoiceScoreMode(str, Enum):
    SUM = "sum"
    MEAN = "mean"


class ClassificationEvaluator:
    """Evaluator for classification tasks that computes accuracy metrics."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 32,
        max_samples_per_lang: int | None = 256,
        decoding: DecodingConfig | None = None,
        choice_score_mode: ChoiceScoreMode | str = ChoiceScoreMode.SUM,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_samples_per_lang = max_samples_per_lang
        self.decoding_config = DecodingConfig.from_any(decoding)
        self.choice_score_mode = ChoiceScoreMode(choice_score_mode)

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
        all_accuracies: list[float] = []
        all_f1s: list[float] = []

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

            if "accuracy" in lang_metrics:
                all_accuracies.append(lang_metrics["accuracy"])
            if "f1" in lang_metrics:
                all_f1s.append(lang_metrics["f1"])

        # Add aggregate accuracy across all languages
        if all_accuracies:
            metrics[f"{metric_prefix}/all_accuracy"] = sum(all_accuracies) / len(
                all_accuracies
            )
        if all_f1s:
            metrics[f"{metric_prefix}/all_f1"] = sum(all_f1s) / len(all_f1s)

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
        gold_labels: list[str] = []
        pred_labels: list[str] = []
        examples_logged = 0
        max_examples_to_log = 3

        with torch.no_grad():
            for sample in dataset:
                messages: list[dict[str, str]] = sample["messages"]
                if not messages or len(messages) < 2:
                    continue

                prompt_messages = messages[:-1]
                gold_label = messages[-1]["content"].strip()
                template_id = sample.get("template_id")
                pred_label = self._predict_label(
                    model=model,
                    prompt_messages=prompt_messages,
                    device=device,
                    pad_id=pad_id,
                    eos_id=eos_id,
                    fallback_template=fallback_template,
                    template_id=str(template_id) if template_id else None,
                    system_message=sample.get("system_message"),
                )

                is_correct = self._labels_match(pred_label, gold_label)
                gold_labels.append(gold_label)
                pred_labels.append(pred_label)

                if examples_logged < max_examples_to_log:
                    logger.info(
                        f"[Classification] pred='{pred_label}' | "
                        f"gold='{gold_label}' | match={is_correct}"
                    )
                    examples_logged += 1

        total = len(gold_labels)
        if total == 0:
            return {"accuracy": 0.0}

        return self._compute_classification_metrics(gold_labels, pred_labels)

    def _predict_label(
        self,
        model: PreTrainedModel,
        prompt_messages: list[dict[str, str]],
        device: torch.device,
        pad_id: int | None,
        eos_id: int | None,
        fallback_template: str | None,
        template_id: str | None,
        system_message: object | None,
    ) -> str:
        label_choices = self._get_label_choices(template_id)
        if label_choices:
            return self._score_label_choices(
                model=model,
                prompt_messages=prompt_messages,
                label_choices=label_choices,
                device=device,
                pad_id=pad_id,
                eos_id=eos_id,
                fallback_template=fallback_template,
                system_message=system_message,
            )

        logger.warning(
            "Classification sample missing template choices for template_id=%s. "
            "Falling back to free-generation scoring.",
            template_id,
        )
        return self._generate_prediction(
            model=model,
            prompt_messages=prompt_messages,
            device=device,
            pad_id=pad_id,
            eos_id=eos_id,
            fallback_template=fallback_template,
            system_message=system_message,
        )

    def _get_label_choices(self, template_id: str | None) -> list[str]:
        if not template_id:
            return []
        spec = tmpl.get(template_id)
        if not spec.label_mapping:
            return []

        items = list(spec.label_mapping.items())
        if items and all(self._is_int_like(key) for key, _ in items):
            items.sort(key=lambda item: int(item[0]))
        return [str(value).strip() for _, value in items]

    def _score_label_choices(
        self,
        model: PreTrainedModel,
        prompt_messages: list[dict[str, str]],
        label_choices: list[str],
        device: torch.device,
        pad_id: int | None,
        eos_id: int | None,
        fallback_template: str | None,
        system_message: object | None,
    ) -> str:
        prompt_text = self._build_prompt_text(
            prompt_messages=prompt_messages,
            fallback_template=fallback_template,
            system_message=system_message,
        )
        pad_token_id = self._resolve_pad_id(pad_id, eos_id)
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
        prompt_len = int(prompt_ids.shape[-1])

        encoded_choices = self.tokenizer(
            [prompt_text + choice for choice in label_choices],
            return_tensors="pt",
            padding=True,
        )
        input_ids = encoded_choices["input_ids"].to(device)
        attn = encoded_choices.get("attention_mask")
        if attn is None:
            attn = (input_ids != pad_token_id).long()
        else:
            attn = attn.to(device)

        input_ids, attn, choice_starts = self._trim_choice_inputs(
            input_ids=input_ids,
            attention_mask=attn,
            prompt_len=prompt_len,
            model_ctx_limit=self._get_model_ctx_limit(model),
            pad_token_id=pad_token_id,
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        token_log_probs = torch.gather(
            torch.log_softmax(logits, dim=-1),
            2,
            target_ids.unsqueeze(-1),
        ).squeeze(-1)

        continuation_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        seq_lens = attn.sum(dim=1)
        for idx, start in enumerate(choice_starts):
            seq_len = int(seq_lens[idx].item())
            if seq_len > start:
                continuation_mask[idx, start:seq_len] = True

        continuation_mask = continuation_mask[:, 1:] & attn[:, 1:].bool()
        choice_scores = self._aggregate_choice_scores(
            token_log_probs=token_log_probs,
            continuation_mask=continuation_mask,
        )
        valid_choice = continuation_mask.sum(dim=1) > 0
        if not bool(valid_choice.any().item()):
            return label_choices[0]
        choice_scores = choice_scores.masked_fill(~valid_choice, float("-inf"))
        return label_choices[int(choice_scores.argmax().item())]

    def _generate_prediction(
        self,
        model: PreTrainedModel,
        prompt_messages: list[dict[str, str]],
        device: torch.device,
        pad_id: int | None,
        eos_id: int | None,
        fallback_template: str | None,
        system_message: object | None = None,
    ) -> str:
        """Generate a prediction for the given prompt."""
        prompt_text = self._build_prompt_text(
            prompt_messages=prompt_messages,
            fallback_template=fallback_template,
            system_message=system_message,
        )
        pad_token_id = self._resolve_pad_id(pad_id, eos_id)

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
        ).to(device)

        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask")
        if attn is None:
            attn = (input_ids != pad_token_id).long()

        model_ctx_limit = self._get_model_ctx_limit(model)
        input_len = int(attn.sum(dim=1).max().item())
        window_len = max(1, model_ctx_limit - 1)
        if input_len >= window_len:
            input_ids = input_ids[:, -window_len:]
            attn = (input_ids != pad_token_id).long()
            input_len = int(attn.sum(dim=1).max().item())
            logger.warning(
                "Classification prompt truncated to last %d tokens to respect "
                "context limit %d.",
                window_len,
                model_ctx_limit,
            )

        generate_kwargs = self.decoding_config.to_generate_kwargs()
        avail_for_gen = max(1, model_ctx_limit - input_len - 1)
        generate_kwargs["max_new_tokens"] = min(self.max_new_tokens, avail_for_gen)
        generate_kwargs["pad_token_id"] = pad_token_id
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

    def _build_prompt_text(
        self,
        prompt_messages: list[dict[str, str]],
        fallback_template: str | None,
        system_message: object | None,
    ) -> str:
        template_kwargs: dict[str, str] = {}
        if isinstance(system_message, str) and system_message.strip():
            template_kwargs["system_message"] = system_message
            template_kwargs["system_prompt"] = system_message
        return self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=fallback_template,
            **template_kwargs,
        )

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

    def _compute_classification_metrics(
        self,
        gold_labels: list[str],
        pred_labels: list[str],
    ) -> dict[str, float]:
        total = len(gold_labels)
        if total == 0:
            return {"accuracy": 0.0}

        correct = sum(
            1
            for gold_label, pred_label in zip(gold_labels, pred_labels, strict=False)
            if self._labels_match(pred_label, gold_label)
        )
        metrics: dict[str, float] = {"accuracy": correct / total}

        class_total: dict[str, int] = defaultdict(int)
        class_correct: dict[str, int] = defaultdict(int)
        pred_total: dict[str, int] = defaultdict(int)

        for gold_label, pred_label in zip(gold_labels, pred_labels, strict=False):
            class_total[gold_label] += 1
            pred_total[pred_label] += 1
            if self._labels_match(pred_label, gold_label):
                class_correct[gold_label] += 1

        labels = list(class_total.keys())
        per_class_acc: list[float] = []
        per_class_f1: list[float] = []
        weighted_f1 = 0.0

        for label in labels:
            support = class_total[label]
            tp = class_correct[label]
            fp = pred_total[label] - tp

            per_class_acc.append(tp / support)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / support if support > 0 else 0.0
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_class_f1.append(f1)
            weighted_f1 += f1 * support

        if per_class_acc:
            metrics["macro_accuracy"] = sum(per_class_acc) / len(per_class_acc)
        if per_class_f1:
            metrics["macro_f1"] = sum(per_class_f1) / len(per_class_f1)
            metrics["f1"] = weighted_f1 / total

        return metrics

    def _aggregate_choice_scores(
        self,
        token_log_probs: torch.Tensor,
        continuation_mask: torch.Tensor,
    ) -> torch.Tensor:
        raw_scores = token_log_probs.masked_fill(~continuation_mask, 0.0).sum(dim=1)
        if self.choice_score_mode == ChoiceScoreMode.SUM:
            return raw_scores
        if self.choice_score_mode == ChoiceScoreMode.MEAN:
            token_counts = continuation_mask.sum(dim=1).clamp_min(1)
            return raw_scores / token_counts
        raise ValueError(f"Unsupported choice score mode '{self.choice_score_mode}'")

    def _cap_dataset(self, dataset: Dataset, lang_key: str) -> Dataset:
        """Cap dataset to max_samples_per_lang."""
        if self.max_samples_per_lang is None:
            return dataset
        if len(dataset) <= self.max_samples_per_lang:
            return dataset
        indices = list(range(self.max_samples_per_lang))
        return dataset.select(indices)

    def _trim_choice_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        model_ctx_limit: int,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        max_seq_len = int(attention_mask.sum(dim=1).max().item())
        if max_seq_len <= model_ctx_limit:
            return input_ids, attention_mask, [prompt_len] * input_ids.shape[0]

        trimmed_sequences: list[torch.Tensor] = []
        choice_starts: list[int] = []
        label_truncated = False
        for row_ids, row_mask in zip(input_ids, attention_mask, strict=False):
            seq_len = int(row_mask.sum().item())
            seq = row_ids[:seq_len]
            removed = max(0, seq_len - model_ctx_limit)
            if removed:
                seq = seq[-model_ctx_limit:]
            choice_start = max(0, prompt_len - removed)
            if removed > prompt_len:
                label_truncated = True
            trimmed_sequences.append(seq)
            choice_starts.append(choice_start)

        if label_truncated:
            logger.warning(
                "Classification scoring truncated at least one label continuation "
                "because the prompt exceeded the model context limit %d.",
                model_ctx_limit,
            )

        padded = pad_sequence(
            trimmed_sequences,
            batch_first=True,
            padding_value=pad_token_id,
        ).to(input_ids.device)
        trimmed_mask = torch.zeros_like(padded, dtype=attention_mask.dtype)
        for idx, seq in enumerate(trimmed_sequences):
            trimmed_mask[idx, : seq.shape[0]] = 1
        return padded, trimmed_mask, choice_starts

    def _resolve_pad_id(self, pad_id: int | None, eos_id: int | None) -> int:
        if pad_id is not None:
            return pad_id
        if eos_id is not None:
            return eos_id
        return 0

    @staticmethod
    def _is_int_like(value: object) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        if isinstance(value, str):
            return value.isdigit()
        return False

    def _get_model_ctx_limit(self, model: PreTrainedModel) -> int:
        if hasattr(model, "config"):
            for attr in (
                "max_position_embeddings",
                "max_sequence_length",
                "n_positions",
            ):
                value = getattr(model.config, attr, None)
                if isinstance(value, int) and value > 0:
                    return value
        tok_max = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tok_max, int) and 0 < tok_max < 1_000_000:
            return tok_max
        gen_max = getattr(
            getattr(model, "generation_config", object()), "max_length", None
        )
        if isinstance(gen_max, int) and gen_max > 256:
            return gen_max
        return 2048

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
