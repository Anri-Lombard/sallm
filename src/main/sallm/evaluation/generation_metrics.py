from __future__ import annotations

import collections
import logging
import os
import random
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import torch
from datasets import Dataset
from evaluate import load as eval_load
from rouge_score import rouge_scorer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sallm.config import (
    DecodingConfig,
    FinetuneTaskType,
    GeneratedExample,
    GenerationEvalResult,
    LanguageEvalResult,
)
from sallm.evaluation.task_metrics import (
    build_ner_debug_record,
    build_pos_debug_record,
    compute_ner_span_f1,
    compute_pos_token_accuracy,
)

logger = logging.getLogger(__name__)


@contextmanager
def _temporary_padding_side(
    tokenizer: PreTrainedTokenizerBase, padding_side: str
) -> Iterator[None]:
    original = getattr(tokenizer, "padding_side", None)
    tokenizer.padding_side = padding_side
    try:
        yield
    finally:
        if original is not None:
            tokenizer.padding_side = original


class GenerationEvaluator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_new_tokens: int = 64,
        max_samples_per_lang: int | None = 64,
        sample_seed: int | None = None,
        skip_special_tokens: bool = True,
        decoding: DecodingConfig | None = None,
        batch_size: int | str | None = None,
        task_type: FinetuneTaskType | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_samples_per_lang = max_samples_per_lang
        self.sample_seed = sample_seed
        self.skip_special_tokens = skip_special_tokens
        self.decoding_config = DecodingConfig.from_any(decoding)
        self.task_type = task_type
        if (
            self.decoding_config.num_return_sequences is not None
            and self.decoding_config.num_return_sequences != 1
        ):
            raise ValueError("GenerationEvaluator supports a single return sequence.")

        env_bs = os.getenv("SALLM_EVAL_BATCH_SIZE")
        env_max_bs = os.getenv("SALLM_EVAL_MAX_BATCH_SIZE")
        default_bs: int | str = "auto:4"
        cfg_bs = getattr(self.decoding_config, "batch_size", None)
        resolved_bs = (
            batch_size
            if batch_size is not None
            else (cfg_bs if cfg_bs is not None else (env_bs if env_bs else default_bs))
        )
        cfg_max_bs = getattr(self.decoding_config, "max_batch_size", None)
        resolved_max_bs = (
            cfg_max_bs
            if cfg_max_bs is not None
            else (int(env_max_bs) if env_max_bs and env_max_bs.isdigit() else 64)
        )
        self.max_batch_size = max(1, int(resolved_max_bs))
        self.auto_batch_schedule = 1
        self._auto_batch_sizes: dict[int, int] = {}
        self._logged_mamba_batch_cap = False

        if isinstance(resolved_bs, str):
            raw_bs = resolved_bs.strip()
            if raw_bs.isdigit():
                resolved_bs = int(raw_bs)
            elif raw_bs.startswith("auto"):
                parts = raw_bs.split(":", 1)
                resolved_bs = "auto"
                if len(parts) == 2 and parts[1]:
                    self.auto_batch_schedule = max(1, int(float(parts[1])))
            else:
                raise ValueError(
                    "Unsupported generation batch_size "
                    f"'{resolved_bs}'. Use an integer, 'auto', or 'auto:N'."
                )

        if isinstance(resolved_bs, int) and resolved_bs < 1:
            resolved_bs = 1
        self.batch_size = resolved_bs

        # Lazily initialise evaluation metrics once
        self._bleu: Any = eval_load("bleu")
        self._chrf: Any = eval_load("chrf")
        self._rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    @staticmethod
    def _is_oom_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "out of memory" in message or "cublas_status_alloc_failed" in message

    @staticmethod
    def _is_mamba_model(model: PreTrainedModel) -> bool:
        candidates: list[object] = [getattr(model, "config", None)]

        base_model = getattr(model, "base_model", None)
        if base_model is not None:
            candidates.append(getattr(base_model, "config", None))
            wrapped_model = getattr(base_model, "model", None)
            if wrapped_model is not None:
                candidates.append(getattr(wrapped_model, "config", None))

        for config in candidates:
            if config is None:
                continue
            model_type = str(getattr(config, "model_type", "")).lower()
            if "mamba" in model_type:
                return True
            architectures = getattr(config, "architectures", None) or []
            for architecture in architectures:
                if "mamba" in str(architecture).lower():
                    return True
        return False

    def _effective_max_batch_size(self, model: PreTrainedModel) -> int:
        if not self._is_mamba_model(model):
            return self.max_batch_size

        raw_cap = os.getenv("SALLM_MAMBA_GENERATION_MAX_BATCH_SIZE", "1")
        try:
            cap = max(1, int(raw_cap))
        except ValueError:
            logger.warning(
                "Ignoring invalid SALLM_MAMBA_GENERATION_MAX_BATCH_SIZE=%r; using 1.",
                raw_cap,
            )
            cap = 1

        effective = min(self.max_batch_size, cap)
        if effective < self.max_batch_size and not self._logged_mamba_batch_cap:
            logger.info(
                "Clamping Mamba generation max batch size from %d to %d to avoid "
                "unstable CUDA auto-batch probing. Override with "
                "SALLM_MAMBA_GENERATION_MAX_BATCH_SIZE if needed.",
                self.max_batch_size,
                effective,
            )
            self._logged_mamba_batch_cap = True
        return effective

    @staticmethod
    def _clear_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _auto_batch_segment(self, start: int, total: int) -> int:
        segment_size = max(1, total // max(1, self.auto_batch_schedule))
        return start // segment_size

    def _prepare_generation_batch(
        self,
        batch_samples: list[dict[str, Any]],
        fallback_template: str | None,
        device: torch.device,
        model_ctx_limit: int,
        pad_id: int | None,
        eos_id: int | None,
    ) -> (
        tuple[
            list[list[dict[str, str]]],
            list[list[str]],
            list[str],
            torch.Tensor,
            torch.Tensor,
            dict[str, Any],
        ]
        | None
    ):
        prompt_messages_list: list[list[dict[str, str]]] = []
        reference_lists: list[list[str]] = []
        prompt_texts: list[str] = []

        for sample in batch_samples:
            messages = cast(list[dict[str, str]], sample["messages"])
            system_message = sample.get("system_message")
            if not messages:
                prompt_messages_list.append([])
                reference_lists.append([""])
                prompt_texts.append("")
                continue
            prompt_messages = messages[:-1]
            reference_texts = self._prepare_references(messages[-1]["content"])
            template_kwargs: dict[str, str] = {}
            if isinstance(system_message, str) and system_message.strip():
                template_kwargs["system_message"] = system_message
                template_kwargs["system_prompt"] = system_message
            prompt_text = cast(
                str,
                cast(Any, self.tokenizer).apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=fallback_template,
                    **template_kwargs,
                ),
            )
            prompt_messages_list.append(prompt_messages)
            reference_lists.append(reference_texts)
            prompt_texts.append(prompt_text)

        if not any(prompt_texts):
            return None

        with _temporary_padding_side(self.tokenizer, "left"):
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
                "Input truncated to last %d tokens to respect context limit %d.",
                window_len,
                model_ctx_limit,
            )

        avail_for_gen = max(1, model_ctx_limit - int(input_lengths.max().item()) - 1)
        eff_max_new = min(self.max_new_tokens, avail_for_gen)

        generate_kwargs = self.decoding_config.to_generate_kwargs()
        generate_kwargs["max_new_tokens"] = eff_max_new
        generate_kwargs["pad_token_id"] = pad_id
        generate_kwargs["eos_token_id"] = eos_id
        generate_kwargs.setdefault("use_cache", True)

        return (
            prompt_messages_list,
            reference_lists,
            prompt_texts,
            input_ids,
            attn,
            generate_kwargs,
        )

    def _detect_auto_batch_size(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        start: int,
        total: int,
        fallback_template: str | None,
        device: torch.device,
        model_ctx_limit: int,
        pad_id: int | None,
        eos_id: int | None,
    ) -> int:
        max_batch_size = self._effective_max_batch_size(model)
        candidate_bs = max(1, min(max_batch_size, total - start))
        while candidate_bs >= 1:
            end = min(start + candidate_bs, total)
            batch_samples = [dataset[i] for i in range(start, end)]
            prepared = self._prepare_generation_batch(
                batch_samples,
                fallback_template,
                device,
                model_ctx_limit,
                pad_id,
                eos_id,
            )
            if prepared is None:
                return 1
            _, _, _, input_ids, attn, generate_kwargs = prepared
            try:
                outputs = cast(Any, model).generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    **generate_kwargs,
                )
                del outputs, input_ids, attn
                self._clear_cuda_cache()
                return len(batch_samples)
            except RuntimeError as exc:
                if not self._is_oom_error(exc) or candidate_bs == 1:
                    raise
                next_bs = max(1, candidate_bs // 2)
                logger.warning(
                    "Auto batch detection hit OOM at batch size %d; retrying with %d.",
                    candidate_bs,
                    next_bs,
                )
                candidate_bs = next_bs
                self._clear_cuda_cache()
        return 1

    def _resolve_batch_size(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        start: int,
        total: int,
        fallback_template: str | None,
        device: torch.device,
        model_ctx_limit: int,
        pad_id: int | None,
        eos_id: int | None,
    ) -> tuple[int, int | None]:
        if self.batch_size != "auto":
            return max(1, int(self.batch_size)), None

        max_batch_size = self._effective_max_batch_size(model)
        segment = self._auto_batch_segment(start, total)
        if segment in self._auto_batch_sizes:
            return min(self._auto_batch_sizes[segment], total - start), segment

        if segment > 0 and self._auto_batch_sizes.get(segment - 1) == max_batch_size:
            self._auto_batch_sizes[segment] = min(max_batch_size, total - start)
            return self._auto_batch_sizes[segment], segment

        detected = self._detect_auto_batch_size(
            model,
            dataset,
            start,
            total,
            fallback_template,
            device,
            model_ctx_limit,
            pad_id,
            eos_id,
        )
        self._auto_batch_sizes[segment] = detected
        logger.info(
            "Determined automatic generation batch size %d for segment %d.",
            detected,
            segment + 1,
        )
        return detected, segment

    def evaluate(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        world_size: int = 1,
        metric_prefix: str = "eval",
        collect_examples: bool = False,
        example_limit_per_lang: int | None = None,
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
            unique_languages = cast(
                list[str | None], str_langs if str_langs else [None]
            )
        else:
            unique_languages = [None]

        metrics: dict[str, float] = {}
        per_language: dict[str, LanguageEvalResult] = {}
        aggregate_metrics: dict[str, list[float]] = collections.defaultdict(list)

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
            if isinstance(gen_max, int) and gen_max > 256:
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
            refs: list[list[str]] = []
            examples: list[GeneratedExample] = []

            with torch.no_grad():
                total = len(capped_dataset)
                if self.batch_size == "auto":
                    self._auto_batch_sizes = {}

                start = 0
                while start < total:
                    bs, segment = self._resolve_batch_size(
                        model,
                        capped_dataset,
                        start,
                        total,
                        fallback_template,
                        device,
                        model_ctx_limit,
                        pad_id,
                        eos_id,
                    )
                    end = min(start + bs, total)
                    batch_samples = [
                        cast(dict[str, Any], capped_dataset[i])
                        for i in range(start, end)
                    ]

                    prepared = self._prepare_generation_batch(
                        batch_samples,
                        fallback_template,
                        device,
                        model_ctx_limit,
                        pad_id,
                        eos_id,
                    )
                    if prepared is None:
                        start = end
                        continue

                    (
                        prompt_messages_list,
                        reference_lists,
                        prompt_texts,
                        input_ids,
                        attn,
                        generate_kwargs,
                    ) = prepared

                    try:
                        outputs = cast(Any, model).generate(
                            input_ids=input_ids,
                            attention_mask=attn,
                            **generate_kwargs,
                        )
                    except RuntimeError as exc:
                        if (
                            self.batch_size == "auto"
                            and bs > 1
                            and self._is_oom_error(exc)
                        ):
                            fallback_bs = max(1, bs // 2)
                            if segment is not None:
                                self._auto_batch_sizes[segment] = fallback_bs
                            logger.warning(
                                "Generation OOM at batch size %d; retrying with %d.",
                                bs,
                                fallback_bs,
                            )
                            self._clear_cuda_cache()
                            continue
                        raise

                    for b_idx in range(outputs.shape[0]):
                        # Left-padding: generated tokens start after input
                        gen_seq = outputs[b_idx][input_ids.shape[1] :]
                        generated_text = self.tokenizer.decode(
                            gen_seq,
                            skip_special_tokens=self.skip_special_tokens,
                            clean_up_tokenization_spaces=True,
                        )
                        cleaned_prediction = self._clean_text(generated_text)

                        preds.append(cleaned_prediction)
                        refs.append(reference_lists[b_idx])

                        if collect_examples and (
                            example_limit_per_lang is None
                            or len(examples) < example_limit_per_lang
                        ):
                            reference = (
                                reference_lists[b_idx][0]
                                if reference_lists[b_idx]
                                else ""
                            )
                            examples.append(
                                GeneratedExample(
                                    prompt_messages=prompt_messages_list[b_idx],
                                    prompt_text=prompt_texts[b_idx],
                                    prediction=cleaned_prediction,
                                    reference=" | ".join(reference_lists[b_idx]),
                                    raw_prediction=generated_text,
                                    debug=self._build_example_debug(
                                        reference=reference,
                                        prediction=cleaned_prediction,
                                    ),
                                )
                            )
                    start = end

            lang_metrics = self._compute_metrics(preds, refs)
            metric_entries = {
                f"{metric_prefix}/{lang_key}_{name}": value
                for name, value in lang_metrics.items()
            }
            metrics.update(metric_entries)
            for name, value in lang_metrics.items():
                aggregate_metrics[name].append(value)
            per_language[lang_key] = LanguageEvalResult(
                key=lang_key,
                metrics=metric_entries,
                examples=examples,
            )

        for name, values in aggregate_metrics.items():
            if values:
                metrics[f"{metric_prefix}/all_{name}"] = sum(values) / len(values)

        return GenerationEvalResult(metrics=metrics, per_language=per_language)

    def _build_example_debug(
        self, reference: str, prediction: str
    ) -> dict[str, object]:
        if self.task_type == FinetuneTaskType.NAMED_ENTITY_RECOGNITION:
            return build_ner_debug_record(reference, prediction)
        if self.task_type == FinetuneTaskType.POS_TAGGING:
            return build_pos_debug_record(reference, prediction)
        return {
            "empty_prediction": not bool(prediction.strip()),
            "exact_match": prediction.strip() == reference.strip(),
        }

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

        bleu_score = None
        non_empty_preds = [p for p in predictions if p.strip()]
        if non_empty_preds:
            try:
                bleu_metrics = self._bleu.compute(
                    predictions=predictions, references=cleaned_refs
                )
                bleu_score = (
                    bleu_metrics.get("bleu") if isinstance(bleu_metrics, dict) else None
                )
                if bleu_score is None and isinstance(bleu_metrics, dict):
                    bleu_score = bleu_metrics.get("score")
            except ZeroDivisionError:
                logger.warning("BLEU computation failed (empty predictions), skipping.")

        chrf_metrics = self._chrf.compute(
            predictions=predictions, references=normalised_refs
        )
        chrf_score = (
            chrf_metrics.get("score") if isinstance(chrf_metrics, dict) else None
        )
        if chrf_score is None and isinstance(chrf_metrics, dict):
            chrf_score = chrf_metrics.get("chrf")

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
        if self.task_type == FinetuneTaskType.NAMED_ENTITY_RECOGNITION:
            out["f1"] = compute_ner_span_f1(
                references=[refs[0] if refs else "" for refs in cleaned_refs],
                predictions=predictions,
            )
        elif self.task_type == FinetuneTaskType.POS_TAGGING:
            out["token_accuracy"] = compute_pos_token_accuracy(
                references=[refs[0] if refs else "" for refs in cleaned_refs],
                predictions=predictions,
            )
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
