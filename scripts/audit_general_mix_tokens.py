#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from sallm.config import ExperimentConfig
from sallm.data.loaders.mix import load_mix_dataset
from sallm.evaluation.lm_eval_runner import _fallback_chat_template
from sallm.models.factory import build_tokenizer


def _assistant_token_count(tokenizer, *, messages: object, max_length: int) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        truncation=True,
        max_length=max_length,
    )
    mask = encoded.get("assistant_masks")
    if mask is None:
        raise ValueError("Tokenizer did not return an assistant token mask")
    return int(sum(mask))


def _balanced_weights(means: dict[str, float]) -> dict[str, float]:
    inverse = {name: 1.0 / value for name, value in means.items()}
    scale = len(inverse) / sum(inverse.values())
    return {name: value * scale for name, value in inverse.items()}


def _load_config(path: Path):
    raw = OmegaConf.load(path)
    return OmegaConf.merge(OmegaConf.structured(ExperimentConfig), raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    config = _load_config(args.config)
    tokenizer = build_tokenizer(config)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = _fallback_chat_template()
    train_mix, _, _ = load_mix_dataset(config)
    components = train_mix._components
    max_length = int(config.dataset.max_seq_length)

    rows: list[dict[str, object]] = []
    means: dict[str, float] = {}
    probabilities = dict(
        zip(train_mix.component_names, train_mix.probabilities, strict=False)
    )
    for component in components:
        total = 0
        nonzero = 0
        for sample in component.dataset:
            count = _assistant_token_count(
                tokenizer,
                messages=sample["messages"],
                max_length=max_length,
            )
            total += count
            nonzero += count > 0
        mean = total / component.size
        means[component.name] = mean
        rows.append(
            {
                "task": component.name,
                "examples": component.size,
                "assistant_tokens": total,
                "mean_assistant_tokens": mean,
                "nonzero_examples": nonzero,
                "sample_probability": probabilities[component.name],
            }
        )

    denominator = sum(probabilities[name] * mean for name, mean in means.items())
    for row in rows:
        name = str(row["task"])
        row["current_target_token_share"] = (
            probabilities[name] * means[name] / denominator
        )

    report = {
        "config": str(args.config),
        "max_length": max_length,
        "components": rows,
        "recommended_mix_temperature": 0.0,
        "recommended_token_balanced_weights": _balanced_weights(means),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
