from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/select_shared_news_general_prompts.py"


def load_script():
    spec = importlib.util.spec_from_file_location("shared_news_prompts", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_summary(module, scores):
    subsets = {}
    for language, n in module.LANGUAGE_ROWS.items():
        for prompt in module.PROMPTS:
            value = scores[language][prompt]
            subsets[f"{language}/{prompt}"] = {
                "n": n,
                "first_token": {
                    "weighted_f1": value,
                    "macro_f1": value,
                    "accuracy": value,
                },
            }
    return {
        "arm": "general",
        "records": 3095,
        "expected_records": 3095,
        "subsets": subsets,
    }


def test_selects_shared_cross_architecture_mean_with_stable_tie_break() -> None:
    module = load_script()
    base = {
        language: {prompt: 0.1 for prompt in module.PROMPTS}
        for language in module.LANGUAGE_ROWS
    }
    summaries = {
        architecture: make_summary(module, base)
        for architecture in module.ARCHITECTURES
    }
    summaries["mzansilm"]["subsets"]["eng/p2"]["first_token"]["weighted_f1"] = 0.9
    summaries["mamba"]["subsets"]["eng/p2"]["first_token"]["weighted_f1"] = 0.9
    summaries["xlstm"]["subsets"]["xho/p3"]["first_token"]["weighted_f1"] = 0.9
    summaries["gdn"]["subsets"]["xho/p3"]["first_token"]["weighted_f1"] = 0.9

    report = module.select(
        summaries, {architecture: "hash" for architecture in module.ARCHITECTURES}
    )

    assert report["selections"]["eng"]["prompt"] == "p2"
    assert report["selections"]["xho"]["prompt"] == "p3"


def test_load_requires_complete_general_validation_summaries(tmp_path: Path) -> None:
    module = load_script()
    scores = {
        language: {prompt: 0.1 for prompt in module.PROMPTS}
        for language in module.LANGUAGE_ROWS
    }
    specs = []
    for architecture in module.ARCHITECTURES:
        summary = make_summary(module, scores)
        if architecture == "gdn":
            summary["records"] = 80
        path = tmp_path / f"{architecture}.json"
        path.write_text(__import__("json").dumps(summary), encoding="utf-8")
        specs.append(f"{architecture}={path}")

    with pytest.raises(ValueError, match="incomplete"):
        module.load_summaries(specs)
