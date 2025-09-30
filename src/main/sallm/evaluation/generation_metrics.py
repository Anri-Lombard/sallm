from __future__ import annotations

import random
from collections.abc import Iterable

import torch
from datasets import Dataset
from evaluate import load as eval_load
from transformers import AutoTokenizer, PreTrainedModel

from sallm.config import (
    GeneratedExample,
    GenerationEvalResult,
    LanguageEvalResult,
)


class GenerationEvaluator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 64,
        max_samples_per_lang: int | None = 64,
        sample_seed: int | None = None,
        skip_special_tokens: bool = False,
        include_combined: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_samples_per_lang = max_samples_per_lang
        self.sample_seed = sample_seed
        self.skip_special_tokens = skip_special_tokens
        self.include_combined = include_combined

        # Lazily initialise evaluation metrics once
        self._rouge = eval_load("rouge")
        self._bleu = eval_load("bleu")
        self._chrf = eval_load("chrf")

    def evaluate(
        self,
        model: PreTrainedModel,
        dataset: Dataset,
        world_size: int = 1,
        metric_prefix: str = "eval",
        collect_examples: bool = False,
    ) -> GenerationEvalResult:
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        if hasattr(model, "eval"):
            model.eval()
        device = getattr(model, "device", torch.device("cpu"))

        lang_column_present = "lang" in dataset.features
        unique_languages: list[str | None]
        if lang_column_present:
            unique_languages = sorted(set(dataset["lang"]))  # type: ignore[index]
        else:
            unique_languages = [None]

        languages_to_process: Iterable[str | None]
        if self.include_combined and lang_column_present:
            languages_to_process = list(unique_languages) + [None]
        else:
            languages_to_process = unique_languages

        metrics: dict[str, float] = {}
        per_language: dict[str, LanguageEvalResult] = {}

        for lang in languages_to_process:
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
                for sample in capped_dataset:
                    messages: list[dict[str, str]] = sample["messages"]
                    if not messages:
                        continue

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
                        use_cache=False,
                    )

                    generated_ids = gen_ids[0][inputs.shape[-1] :]
                    generated_text = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=self.skip_special_tokens,
                        clean_up_tokenization_spaces=True,
                    )

                    preds.append(generated_text)
                    refs.append(gold_completion)

                    if collect_examples:
                        prompt_text = self.tokenizer.apply_chat_template(
                            prompt_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        examples.append(
                            GeneratedExample(
                                prompt_messages=prompt_messages,
                                prompt_text=prompt_text,
                                prediction=generated_text,
                                reference=gold_completion,
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
        references: list[str],
    ) -> dict[str, float]:
        if not predictions or not references:
            return {}

        rouge_metrics = self._rouge.compute(
            predictions=predictions, references=references
        )
        r1 = rouge_metrics.get("rouge1")
        r2 = rouge_metrics.get("rouge2")
        rl = rouge_metrics.get("rougeL")

        bleu_metrics = self._bleu.compute(
            predictions=predictions, references=[[ref] for ref in references]
        )
        bleu_score = bleu_metrics.get("bleu") or bleu_metrics.get("score")

        chrf_metrics = self._chrf.compute(
            predictions=predictions, references=references
        )
        chrf_score = chrf_metrics.get("score") or chrf_metrics.get("chrf")

        out: dict[str, float] = {}
        if r1 is not None:
            out["rouge1"] = float(r1)
        if r2 is not None:
            out["rouge2"] = float(r2)
        if rl is not None:
            out["rougeL"] = float(rl)
        if bleu_score is not None:
            out["bleu"] = float(bleu_score)
        if chrf_score is not None:
            out["chrf"] = float(chrf_score)
        return out
