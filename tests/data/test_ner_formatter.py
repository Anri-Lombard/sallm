from sallm.data.formatters.ner import reconstruct_entities_from_iob

TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]


def test_iob_spans_become_labelled_entities() -> None:
    tokens = ["Nelson", "Mandela", "wazalelwa", "eMvezo", "."]
    tags = [1, 2, 0, 3, 0]

    assert reconstruct_entities_from_iob(tokens, tags, TAGS) == [
        "PER: Nelson Mandela",
        "LOC: eMvezo",
    ]


def test_adjacent_b_tags_start_separate_entities() -> None:
    assert reconstruct_entities_from_iob(["Thabo", "Sipho"], [1, 1], TAGS) == [
        "PER: Thabo",
        "PER: Sipho",
    ]


def test_entity_at_end_of_sentence_is_kept() -> None:
    assert reconstruct_entities_from_iob(["e", "Cape", "Town"], [0, 3, 4], TAGS) == [
        "LOC: Cape Town"
    ]


def test_inside_tag_of_another_label_closes_the_entity_and_is_dropped() -> None:
    tokens = ["Nelson", "Mandela", "Bay"]

    assert reconstruct_entities_from_iob(tokens, [1, 2, 4], TAGS) == [
        "PER: Nelson Mandela"
    ]
