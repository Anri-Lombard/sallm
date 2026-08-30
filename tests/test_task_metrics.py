from __future__ import annotations

from sallm.evaluation.task_metrics import (
    build_ner_debug_record,
    build_pos_debug_record,
    compute_ner_quality_metrics,
    compute_ner_span_f1,
    compute_pos_quality_metrics,
    compute_pos_token_accuracy,
)


def test_ner_parser_preserves_punctuation_and_label_words_in_entity_text() -> None:
    record = build_ner_debug_record(
        "PER: David A. Gross $$ LOC: Kazan, Russia "
        "$$ ORG: The Bomb Shelter Film Company",
        "PER: David A. Gross $$ LOC: Kazan, Russia "
        "$$ ORG: The Bomb Shelter Film Company",
    )

    assert record["gold_spans"] == [
        {"label": "per", "text": "david a. gross"},
        {"label": "loc", "text": "kazan, russia"},
        {"label": "org", "text": "the bomb shelter film company"},
    ]
    assert record["predicted_spans"] == record["gold_spans"]


def test_ner_parser_maps_only_complete_label_fields() -> None:
    record = build_ner_debug_record(
        "PER: Alice $$ LOC: Cape Town",
        "PERSON: Alice\nLOCATION: Cape Town\ncompanywide: Ignore Me",
    )

    assert record["predicted_spans"] == [
        {"label": "per", "text": "alice"},
        {"label": "loc", "text": "cape town"},
    ]


def test_pos_token_accuracy_penalizes_extra_tags() -> None:
    reference = "NOUN VERB"
    prediction = "NOUN VERB ADJ"

    assert compute_pos_token_accuracy([reference], [prediction]) == 2 / 3
    assert build_pos_debug_record(reference, prediction)["token_accuracy"] == 2 / 3


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

    assert compute_pos_token_accuracy(references, predictions) == 7 / 12
    assert metrics["valid_tag_rate"] == 1.0
    assert metrics["length_match_rate"] == 1 / 2
    assert metrics["empty_prediction_rate"] == 0.0
    assert metrics["repetition_rate"] == 1 / 2
