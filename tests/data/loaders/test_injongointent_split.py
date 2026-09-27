from sallm.data.loaders.injongointent_split import (
    exclude_heldout_texts,
    split_injongointent_rows,
)


def test_exclude_heldout_texts_normalizes_case_and_whitespace() -> None:
    rows = [
        {"intent": "alarm", "text": " Wake   me UP "},
        {"intent": "balance", "text": "How much remains?"},
    ]
    heldout = [{"intent": "alarm", "text": "wake me up"}]

    assert exclude_heldout_texts(rows, heldout) == [rows[1]]


def test_split_is_balanced_with_repeated_example_ids_across_labels() -> None:
    rows = [
        {"intent": intent, "example_id": f"train-{index}", "text": str(index)}
        for intent in ("alarm", "balance")
        for index in range(20)
    ]

    train, validation = split_injongointent_rows(rows)

    assert len(train) == 36
    assert len(validation) == 4
    assert {
        intent: sum(row["intent"] == intent for row in validation)
        for intent in ("alarm", "balance")
    } == {"alarm": 2, "balance": 2}
    assert {(row["intent"], row["example_id"]) for row in train}.isdisjoint(
        {(row["intent"], row["example_id"]) for row in validation}
    )
