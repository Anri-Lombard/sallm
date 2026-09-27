from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/materialize_news_official_test.py"


def load_script():
    spec = importlib.util.spec_from_file_location("news_test_materializer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_gate(tmp_path: Path):
    verification = tmp_path / "VERIFIED.json"
    verification.write_text(
        json.dumps({"status": "VALIDATION_CHAIN_VERIFIED", "test_accessed": False})
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "data_boundary": "validation_only",
                "test_accessed": False,
                "selections": {"eng": {"prompt": "p2"}, "xho": {"prompt": "p4"}},
            }
        )
    )
    tie_sensitivity = tmp_path / "tie-sensitivity.json"
    tie_sensitivity.write_text(
        json.dumps(
            {
                "status": "VALIDATION_CHAIN_INDEPENDENTLY_VERIFIED",
                "test_accessed": False,
                "tie_policy": {
                    "frozen_order": [
                        "business",
                        "entertainment",
                        "health",
                        "politics",
                        "religion",
                        "sports",
                        "technology",
                    ],
                    "shared_prompt_changes_under_reverse": False,
                    "architecture_ranking_changes_under_reverse": False,
                },
            }
        )
    )

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return verification, selection, tie_sensitivity, sha


def test_materializes_only_pinned_test_files_after_gate(tmp_path: Path) -> None:
    module = load_script()
    verification, selection, tie_sensitivity, sha = write_gate(tmp_path)
    revision = module.DATASET_REVISION
    payloads = {
        language: (
            b"category\theadline\ttext\turl\n"
            + b"business\th\tt\tu\n" * module.LANGUAGE_ROWS[language]
        )
        for language in module.LANGUAGE_ROWS
    }
    files = {
        language: {
            "url": f"https://host/repo/resolve/{revision}/data/{language}/test.tsv",
            "rows": module.LANGUAGE_ROWS[language],
        }
        for language in payloads
    }
    protocol = {
        "schema": "sallm.news_official_test_access/v1",
        "status": "SEALED_AWAITING_ROOT_ACKNOWLEDGEMENT",
        "data_boundary": "held_out_test",
        "no_score_based_retry": True,
        "selected_prompts": {"eng": "p2", "xho": "p4"},
        "tie_policy": {
            "labels": module.FROZEN_LABEL_ORDER,
            "rule": "first_label_in_frozen_order",
            "exact_tolerance": 1e-12,
        },
        "validation_gate": {
            "chain_verification": {
                "path": str(verification),
                "sha256": sha(verification),
            },
            "prompt_selection": {"path": str(selection), "sha256": sha(selection)},
            "tie_sensitivity": {
                "path": str(tie_sensitivity),
                "sha256": sha(tie_sensitivity),
            },
        },
        "dataset": {
            "repo": "masakhane/masakhanews",
            "revision": revision,
            "files": files,
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(
        json.dumps(
            {
                "schema": "sallm.news_official_test_authorization/v1",
                "status": "AUTHORIZED_AFTER_VALIDATION_REVIEW",
                "access_protocol_sha256": sha(protocol_path),
            }
        )
    )
    opened = []

    def opener(url, timeout):
        assert timeout == 120
        opened.append(url)
        language = "eng" if "/eng/" in url else "xho"
        return io.BytesIO(payloads[language])

    output = tmp_path / "test"
    report = module.materialize(
        protocol_path=protocol_path,
        authorization_path=authorization_path,
        destination=output,
        opener=opener,
        now=lambda: datetime(2026, 9, 14, tzinfo=UTC),
    )

    assert len(opened) == 2
    assert all(url.endswith("/test.tsv") for url in opened)
    assert report["metrics_computed"] is False
    assert report["files"]["eng"]["rows"] == module.LANGUAGE_ROWS["eng"]
    assert (output / "ACCESS_STARTED.json").is_file()


def test_rejects_non_test_url_before_any_open(tmp_path: Path) -> None:
    module = load_script()
    verification, selection, tie_sensitivity, sha = write_gate(tmp_path)
    revision = module.DATASET_REVISION
    protocol = {
        "schema": "sallm.news_official_test_access/v1",
        "status": "SEALED_AWAITING_ROOT_ACKNOWLEDGEMENT",
        "data_boundary": "held_out_test",
        "no_score_based_retry": True,
        "selected_prompts": {"eng": "p2", "xho": "p4"},
        "tie_policy": {
            "labels": module.FROZEN_LABEL_ORDER,
            "rule": "first_label_in_frozen_order",
            "exact_tolerance": 1e-12,
        },
        "validation_gate": {
            "chain_verification": {
                "path": str(verification),
                "sha256": sha(verification),
            },
            "prompt_selection": {"path": str(selection), "sha256": sha(selection)},
            "tie_sensitivity": {
                "path": str(tie_sensitivity),
                "sha256": sha(tie_sensitivity),
            },
        },
        "dataset": {
            "repo": "masakhane/masakhanews",
            "revision": revision,
            "files": {
                "eng": {
                    "url": f"https://host/resolve/{revision}/data/eng/dev.tsv",
                    "rows": module.LANGUAGE_ROWS["eng"],
                },
                "xho": {
                    "url": f"https://host/resolve/{revision}/data/xho/test.tsv",
                    "rows": module.LANGUAGE_ROWS["xho"],
                },
            },
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(
        json.dumps(
            {
                "schema": "sallm.news_official_test_authorization/v1",
                "status": "AUTHORIZED_AFTER_VALIDATION_REVIEW",
                "access_protocol_sha256": sha(protocol_path),
            }
        )
    )
    opened = []

    with pytest.raises(ValueError, match="non-test URL"):
        module.materialize(
            protocol_path=protocol_path,
            authorization_path=authorization_path,
            destination=tmp_path / "test",
            opener=lambda *args, **kwargs: opened.append(args) or io.BytesIO(),
        )
    assert opened == []
