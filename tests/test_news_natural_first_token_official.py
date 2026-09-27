from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/run_news_natural_first_token_official.py"


def load_script():
    spec = importlib.util.spec_from_file_location("news_official", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_summary(module, score_by_language_prompt):
    subsets = {}
    for language in module.LANGUAGE_ROWS:
        for prompt_number in range(1, 6):
            prompt = f"p{prompt_number}"
            score = score_by_language_prompt[language][prompt]
            subsets[f"{language}/{prompt}"] = {
                "first_token": {
                    "weighted_f1": score,
                    "macro_f1": score,
                    "accuracy": score,
                }
            }
    return {
        "arm": "general",
        "records": 3095,
        "expected_records": 3095,
        "subsets": subsets,
    }


def test_recomputes_shared_prompt_selection_instead_of_trusting_report() -> None:
    module = load_script()
    scores = {
        language: {f"p{index}": 0.1 for index in range(1, 6)}
        for language in module.LANGUAGE_ROWS
    }
    scores["eng"]["p3"] = 0.8
    scores["xho"]["p4"] = 0.9
    summaries = {
        architecture: make_summary(module, scores)
        for architecture in module.ARCHITECTURES
    }
    candidates = {
        language: {
            prompt: {
                "mean_weighted_f1": score,
                "per_architecture": {},
            }
            for prompt, score in language_scores.items()
        }
        for language, language_scores in scores.items()
    }
    selection = {
        "schema": "sallm.news_general_shared_prompt_selection/v1",
        "data_boundary": "validation_only",
        "test_accessed": False,
        "architectures": list(module.ARCHITECTURES),
        "candidates": candidates,
        "selections": {"eng": {"prompt": "p3"}, "xho": {"prompt": "p4"}},
    }

    assert module._validate_shared_selection(
        selection=selection,
        summaries=summaries,
    ) == {"eng": "p3", "xho": "p4"}

    selection["candidates"]["eng"]["p3"]["mean_weighted_f1"] = 0.7
    with pytest.raises(ValueError, match="shared validation mean mismatch"):
        module._validate_shared_selection(selection=selection, summaries=summaries)


def test_parses_only_exact_manifest_bound_test_payload() -> None:
    module = load_script()
    payload = (
        b"category\theadline\ttext\turl\n"
        b"business\tHeadline\tBody\thttps://example.test\n"
    )
    spec = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "rows": 1,
        "columns": ["category", "headline", "text", "url"],
        "observed_labels": ["business"],
    }

    rows = module.parse_test_tsv(payload=payload, spec=spec, language="eng")
    assert rows[0]["category"] == "business"

    with pytest.raises(ValueError, match="hash mismatch"):
        module.parse_test_tsv(payload=payload + b"\n", spec=spec, language="eng")


def test_prediction_integrity_requires_complete_unique_repeat_stable_rows() -> None:
    module = load_script()
    module.LANGUAGE_ROWS = {"eng": 2, "xho": 1}
    selected = {"eng": "p3", "xho": "p4"}
    predictions = []
    for language, count in module.LANGUAGE_ROWS.items():
        for source_index in range(count):
            scores = {label: -10.0 for label in module.LABELS}
            scores["sports"] = -1.0
            predictions.append(
                {
                    "language": language,
                    "source_index": source_index,
                    "prompt_id": selected[language],
                    "gold": "sports",
                    "prediction": "sports",
                    "first_token_log_probabilities": scores,
                    "first_token_repeat_log_probabilities": dict(scores),
                }
            )

    report = module.validate_prediction_integrity(
        predictions=predictions,
        selected_prompts=selected,
    )
    assert report["observed_rows"] == 3

    predictions.pop()
    with pytest.raises(ValueError, match="completeness"):
        module.validate_prediction_integrity(
            predictions=predictions,
            selected_prompts=selected,
        )


def test_snapshot_manifest_rejects_changed_payload(tmp_path: Path) -> None:
    module = load_script()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    payload = snapshot / "runner.py"
    payload.write_text("frozen\n", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = snapshot / "SNAPSHOT_PAYLOAD.sha256"
    manifest.write_text(f"{digest}  runner.py\n", encoding="utf-8")
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    module._validate_snapshot_manifest(
        snapshot=snapshot,
        manifest_path=manifest,
        expected_sha256=manifest_digest,
    )

    payload.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot file changed"):
        module._validate_snapshot_manifest(
            snapshot=snapshot,
            manifest_path=manifest,
            expected_sha256=manifest_digest,
        )
