import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from injongointent_trainonly_validation import (  # noqa: E402
    DATASET_NAME,
    DATASET_REVISION,
    MANIFEST_SCHEMA,
    derive_language_validation,
    load_frozen_validation_rows,
)


def training_bytes() -> bytes:
    rows = [
        {"text": f"utterance {intent} {index}", "intent": intent}
        for intent in ("alarm", "balance")
        for index in range(20)
    ]
    rows.append({"text": "  UTTERANCE alarm 0 ", "intent": "alarm"})
    return ("\n".join(json.dumps(row) for row in rows) + "\n").encode()


def test_train_only_validation_is_deterministic_and_deduplicated() -> None:
    content = training_bytes()

    first_rows, first_evidence = derive_language_validation("eng", content)
    second_rows, second_evidence = derive_language_validation("eng", content)

    assert first_evidence == second_evidence
    assert first_rows == second_rows
    assert first_evidence["raw_row_count"] == 41
    assert first_evidence["deduplicated_row_count"] == 40
    assert first_evidence["within_train_duplicate_count"] == 1
    assert first_evidence["validation_row_count"] == 4
    assert len({row["__validation_source_index"] for row in first_rows}) == 4


def test_frozen_validation_round_trip_uses_local_train_file(tmp_path: Path) -> None:
    content = training_bytes()
    source = tmp_path / "sources" / "eng" / "train.jsonl"
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    expected_rows, evidence = derive_language_validation("eng", content)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "dataset": DATASET_NAME,
        "revision": DATASET_REVISION,
        "source_split": "train",
        "held_out_data_accessed": False,
        "architecture_blind": True,
        "languages": {"eng": {**evidence, "source_path": "sources/eng/train.jsonl"}},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    observed_rows, observed_evidence = load_frozen_validation_rows(
        manifest_path, ["eng"]
    )

    assert observed_rows == expected_rows
    assert observed_evidence["source_split"] == "train"
    assert observed_evidence["held_out_data_accessed"] is False
    assert observed_evidence["architecture_blind"] is True
