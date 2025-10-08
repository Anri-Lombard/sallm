from __future__ import annotations

import logging
import os
import random
import textwrap

import torch
from datasets import Dataset
from evaluate import load as eval_load
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, PreTrainedModel

from sallm.config import (
    DecodingConfig,
    GeneratedExample,
    GenerationEvalResult,
    LanguageEvalResult,
)

logger = logging.getLogger(__name__)


class GenerationEvaluator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 64,
        max_samples_per_lang: int | None = 64,
        sample_seed: int | None = None,
        skip_special_tokens: bool = True,
        decoding: DecodingConfig | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_samples_per_lang = max_samples_per_lang
        self.sample_seed = sample_seed
        self.skip_special_tokens = skip_special_tokens
        self.decoding_config = DecodingConfig.from_any(decoding)
        if (
            self.decoding_config.num_return_sequences is not None
            and self.decoding_config.num_return_sequences != 1
        ):
            raise ValueError("GenerationEvaluator supports a single return sequence.")

        env_bs = os.getenv("SALLM_EVAL_BATCH_SIZE")
        default_bs = 4
        cfg_bs = getattr(self.decoding_config, "batch_size", None)
        resolved_bs = (
            batch_size
            if batch_size is not None
            else (
                cfg_bs
                if cfg_bs is not None
                else (int(env_bs) if env_bs and env_bs.isdigit() else default_bs)
            )
        )
        if resolved_bs < 1:
            resolved_bs = 1
        self.batch_size = resolved_bs

        # Lazily initialise evaluation metrics once
        self._bleu = eval_load("bleu")
        self._chrf = eval_load("chrf")
        self._rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def evaluate(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        world_size: int = 1,
        metric_prefix: str = "eval",
        collect_examples: bool = False,
    ) -> GenerationEvalResult:
        fallback_template = None
        if getattr(self.tokenizer, "chat_template", None) is None:
            fallback_template = textwrap.dedent(
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
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        if hasattr(model, "eval"):
            model.eval()
        device = getattr(model, "device", torch.device("cpu"))

        lang_column_present = "lang" in dataset.features
        unique_languages: list[str | None]
        if lang_column_present:
            raw_langs = set(dataset["lang"])  # type: ignore[index]
            str_langs = sorted({x for x in raw_langs if isinstance(x, str) and x})
            unique_languages = str_langs if str_langs else [None]
        else:
            unique_languages = [None]

        metrics: dict[str, float] = {}
        per_language: dict[str, LanguageEvalResult] = {}

        model_ctx_limit = None
        if hasattr(model, "config"):
            for attr in (
                "max_position_embeddings",
                "max_sequence_length",
                "n_positions",
            ):
                val = getattr(model.config, attr, None)
                if isinstance(val, int) and val > 0:
                    model_ctx_limit = val
                    break
        if model_ctx_limit is None:
            tok_max = getattr(self.tokenizer, "model_max_length", None)
            if isinstance(tok_max, int) and 0 < tok_max < 1_000_000:
                model_ctx_limit = tok_max
        if model_ctx_limit is None:
            gen_max = getattr(
                getattr(model, "generation_config", object()), "max_length", None
            )
            if isinstance(gen_max, int) and gen_max > 0:
                model_ctx_limit = gen_max
        if model_ctx_limit is None:
            model_ctx_limit = 2048

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

            capped_dataset = self._cap_dataset(lang_dataset, world_size, lang_key)

            preds: list[str] = []
            refs: list[str] = []
            examples: list[GeneratedExample] = []

            with torch.no_grad():
                total = len(capped_dataset)
                bs = max(1, self.batch_size)

                for start in range(0, total, bs):
                    end = min(start + bs, total)
                    batch_samples = [capped_dataset[i] for i in range(start, end)]

                    prompt_messages_list: list[list[dict[str, str]]] = []
                    reference_lists: list[list[str]] = []
                    prompt_texts: list[str] = []

                    for sample in batch_samples:
                        messages: list[dict[str, str]] = sample["messages"]
                        if not messages:
                            prompt_messages_list.append([])
                            reference_lists.append([""])
                            prompt_texts.append("")
                            continue
                        prompt_messages = messages[:-1]
                        reference_texts = self._prepare_references(
                            messages[-1]["content"]
                        )
                        prompt_text = self.tokenizer.apply_chat_template(
                            prompt_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            chat_template=fallback_template,
                        )
                        prompt_messages_list.append(prompt_messages)
                        reference_lists.append(reference_texts)
                        prompt_texts.append(prompt_text)

                    if not any(prompt_texts):
                        continue

                    tokenized = self.tokenizer(
                        prompt_texts,
                        return_tensors="pt",
                        padding=True,
                    ).to(device)

                    input_ids = tokenized["input_ids"]
                    attn = tokenized.get("attention_mask")
                    if attn is None:
                        attn = (input_ids != pad_id).long()
                    input_lengths = attn.sum(dim=1)

                    max_input_len = int(input_lengths.max().item())
                    window_len = max(1, model_ctx_limit - 1)
                    if max_input_len >= window_len:
                        input_ids = input_ids[:, -window_len:]
                        attn = (input_ids != pad_id).long()
                        input_lengths = attn.sum(dim=1)
                        logger.warning(
                            "Input truncated to last %d tokens "
                            "to respect context limit %d.",
                            window_len,
                            model_ctx_limit,
                        )

                    avail_for_gen = max(
                        1, model_ctx_limit - int(input_lengths.max().item()) - 1
                    )
                    eff_max_new = min(self.max_new_tokens, avail_for_gen)

                    generate_kwargs = self.decoding_config.to_generate_kwargs()
                    generate_kwargs["max_new_tokens"] = eff_max_new
                    generate_kwargs["pad_token_id"] = pad_id
                    generate_kwargs["eos_token_id"] = eos_id
                    generate_kwargs.setdefault("use_cache", True)

                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        **generate_kwargs,
                    )

                    for b_idx in range(outputs.shape[0]):
                        prompt_len = int(input_lengths[b_idx].item())
                        gen_seq = outputs[b_idx][prompt_len:]
                        generated_text = self.tokenizer.decode(
                            gen_seq,
                            skip_special_tokens=self.skip_special_tokens,
                            clean_up_tokenization_spaces=True,
                        )
                        cleaned_prediction = self._clean_text(generated_text)

                        preds.append(cleaned_prediction)
                        refs.append(reference_lists[b_idx])

                        if collect_examples:
                            examples.append(
                                GeneratedExample(
                                    prompt_messages=prompt_messages_list[b_idx],
                                    prompt_text=prompt_texts[b_idx],
                                    prediction=cleaned_prediction,
                                    reference=" | ".join(reference_lists[b_idx]),
                                )
                            )

            lang_metrics = self._compute_metrics(preds, refs)
            metric_entries = {
                f"{metric_prefix}/{lang_key}_{name}": value
                for name, value in lang_metrics.items()
            }
            metrics.update(metric_entries)
            per_language[lang_key] = LanguageEvalResult(
                key=lang_key,
                metrics=metric_entries,
                examples=examples,
            )

        return GenerationEvalResult(metrics=metrics, per_language=per_language)

    def _cap_dataset(
        self,
        dataset: Dataset,
        world_size: int,
        lang_key: str,
    ) -> Dataset:
        if self.max_samples_per_lang is None:
            return dataset

        if len(dataset) <= self.max_samples_per_lang:
            return dataset

        sample_cap = self.max_samples_per_lang
        if world_size > 1:
            sample_cap = max(1, sample_cap // world_size)

        indices = self._sample_indices(len(dataset), sample_cap, lang_key)
        return dataset.select(indices)

    def _sample_indices(
        self,
        population_size: int,
        sample_size: int,
        lang_key: str,
    ) -> list[int]:
        if sample_size >= population_size:
            return list(range(population_size))

        if self.sample_seed is not None:
            rng = random.Random(self.sample_seed + hash(lang_key))
            return sorted(rng.sample(range(population_size), sample_size))

        return sorted(random.sample(range(population_size), sample_size))

    def _compute_metrics(
        self,
        predictions: list[str],
        references: list[list[str]],
    ) -> dict[str, float]:
        if not predictions or not references:
            return {}

        cleaned_refs = [self._ensure_reference_list(refs) for refs in references]
        normalised_refs = self._normalize_reference_counts(cleaned_refs)

        rouge_totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for prediction, ref_list in zip(predictions, cleaned_refs, strict=False):
            best_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            for reference in ref_list:
                scores = self._rouge_scorer.score(reference, prediction)
                for key in best_scores:
                    best_scores[key] = max(best_scores[key], scores[key].fmeasure)
            for key in rouge_totals:
                rouge_totals[key] += best_scores[key]

        count = len(predictions)
        rouge_metrics = {k: rouge_totals[k] / count for k in rouge_totals}

        bleu_metrics = self._bleu.compute(
            predictions=predictions, references=cleaned_refs
        )
        bleu_score = bleu_metrics.get("bleu") or bleu_metrics.get("score")

        chrf_metrics = self._chrf.compute(
            predictions=predictions, references=normalised_refs
        )
        chrf_score = chrf_metrics.get("score") or chrf_metrics.get("chrf")

        out: dict[str, float] = {}
        if rouge_metrics.get("rouge1") is not None:
            out["rouge1"] = float(rouge_metrics["rouge1"])
        if rouge_metrics.get("rouge2") is not None:
            out["rouge2"] = float(rouge_metrics["rouge2"])
        if rouge_metrics.get("rougeL") is not None:
            out["rougeL"] = float(rouge_metrics["rougeL"])
        if bleu_score is not None:
            out["bleu"] = float(bleu_score)
        if chrf_score is not None:
            out["chrf"] = float(chrf_score)
        return out

    def _prepare_references(self, text: str) -> list[str]:
        parts = text.split("*#")
        cleaned = [self._clean_text(p) for p in parts]
        filtered = [p for p in cleaned if p]
        return filtered or [""]

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned = text.strip()
        trailing_markers = (
            "[EOS]",
            "[EOT]",
            "[eos]",
            "[eot]",
            "</s>",
            "<|endoftext|>",
            "<|im_end|>",
            "<|eot_id|>",
        )
        for marker in trailing_markers:
            if cleaned.endswith(marker):
                cleaned = cleaned[: -len(marker)].strip()
        return cleaned

    @staticmethod
    def _ensure_reference_list(refs: list[str]) -> list[str]:
        cleaned = [r for r in refs if r]
        return cleaned or [""]

    @staticmethod
    def _normalize_reference_counts(references: list[list[str]]) -> list[list[str]]:
        if not references:
            return references
        target = max(len(ref_list) for ref_list in references)
        if target <= 1:
            return references
        normalised: list[list[str]] = []
        for ref_list in references:
            if len(ref_list) == target:
                normalised.append(ref_list)
                continue
            padded = list(ref_list)
            padded.extend([ref_list[-1]] * (target - len(ref_list)))
            normalised.append(padded)
        return normalised
