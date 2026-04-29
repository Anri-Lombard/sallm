from __future__ import annotations

from sallm.evaluation.task_metrics import (
    compute_ner_quality_metrics,
    compute_ner_span_f1,
    compute_pos_quality_metrics,
    compute_pos_token_accuracy,
)


def test_ner_quality_metrics_capture_parse_and_nonempty_gold() -> None:
    references = [
        "PER: Alice $$ LOC: Cape Town",
        "",
        "ORG: UCT",
    ]
    predictions = [
        "PER: Alice $$ LOC: Cape Town",
        "",
        "UCT UCT UCT UCT UCT UCT",
    ]

    metrics = compute_ner_quality_metrics(references, predictions)

    assert compute_ner_span_f1(references, predictions) > 0
    assert metrics["parse_rate"] == 2 / 3
    assert metrics["empty_prediction_rate"] == 1 / 3
    assert metrics["nonempty_gold_prediction_rate"] == 1 / 2
    assert metrics["repetition_rate"] == 1 / 3


def test_pos_quality_metrics_capture_length_and_repetition() -> None:
    references = [
        "NOUN VERB PROPN",
        "PRON AUX VERB",
    ]
    predictions = [
        "NOUN VERB PROPN",
        "PRON PRON PRON PRON PRON PRON",
    ]

    metrics = compute_pos_quality_metrics(references, predictions)

    assert compute_pos_token_accuracy(references, predictions) == 2 / 3
    assert metrics["valid_tag_rate"] == 1.0
    assert metrics["length_match_rate"] == 1 / 2
    assert metrics["empty_prediction_rate"] == 0.0
    assert metrics["repetition_rate"] == 1 / 2
