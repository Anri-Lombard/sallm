from __future__ import annotations

from sallm.evaluation.task_metrics import (
    _normalize_ner_prediction,
    _tags_to_spans,
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

    assert compute_pos_token_accuracy(references, predictions) == 7 / 12
    assert metrics["valid_tag_rate"] == 1.0
    assert metrics["length_match_rate"] == 1 / 2
    assert metrics["empty_prediction_rate"] == 0.0
    assert metrics["repetition_rate"] == 1 / 2


def test_ner_parser_preserves_punctuation_and_entity_substrings() -> None:
    text = (
        "PER: David A. Gross $$ LOC: Kazan, Russia $$ "
        "ORG: The Bomb Shelter Film Company $$ ORG: Stimela"
    )

    assert _normalize_ner_prediction(text) == (
        "PER: David A. Gross $ LOC: Kazan, Russia $ "
        "ORG: The Bomb Shelter Film Company $ ORG: Stimela"
    )
    assert _tags_to_spans(text) == [
        ("per", "david a. gross"),
        ("loc", "kazan, russia"),
        ("org", "the bomb shelter film company"),
        ("org", "stimela"),
    ]


def test_ner_parser_maps_only_complete_label_fields() -> None:
    text = "PERSON: Alice\nLOCATION: Cape Town\ncompanywide: Ignore Me"

    assert _tags_to_spans(text) == [
        ("per", "alice"),
        ("loc", "cape town"),
    ]


def test_pos_prefix_plus_extra_label_is_not_full_credit() -> None:
    score = compute_pos_token_accuracy(["NOUN VERB"], ["NOUN VERB X"])

    assert score == 2 / 3


def test_ner_span_f1_matches_nfd_prediction_to_nfc_reference() -> None:
    import unicodedata

    reference = "PER: Mošweu $$ LOC: Tšhwane"  # NFC, as in MasakhaNER
    prediction = unicodedata.normalize("NFD", reference)  # as decoded by the tokenizer
    assert prediction != reference
    assert compute_ner_span_f1([reference], [prediction]) > 0.99
