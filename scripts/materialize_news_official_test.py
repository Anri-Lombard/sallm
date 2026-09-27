#!/usr/bin/env python3
"""Materialize the frozen MasakhaNEWS test payload after the validation gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LABELS = {
    "business",
    "entertainment",
    "health",
    "politics",
    "religion",
    "sports",
    "technology",
}
DATASET_REPO = "masakhane/masakhanews"
DATASET_REVISION = "fa3b5fff8a91d187bf0c5900a39c4271d08cf7fe"
LANGUAGE_ROWS = {"eng": 948, "xho": 297}
FROZEN_LABEL_ORDER = [
    "business",
    "entertainment",
    "health",
    "politics",
    "religion",
    "sports",
    "technology",
]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse(payload: bytes, *, language: str, expected_rows: int) -> tuple[list, list]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), delimiter="\t")
    expected_columns = ["category", "headline", "text", "url"]
    require(reader.fieldnames == expected_columns, f"{language} test schema mismatch")
    rows = [dict(row) for row in reader]
    require(len(rows) == expected_rows, f"{language} test row count mismatch")
    complete_rows = all(
        all(row.get(column) is not None for column in expected_columns) for row in rows
    )
    require(complete_rows, f"{language} test contains a ragged row")
    observed = sorted({row["category"].strip() for row in rows})
    require(
        set(observed).issubset(LABELS),
        f"{language} test contains an unknown label",
    )
    return rows, observed


def materialize(
    *,
    protocol_path: Path,
    authorization_path: Path,
    destination: Path,
    opener: Callable[..., Any] = urllib.request.urlopen,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(
        protocol.get("schema") == "sallm.news_official_test_access/v1",
        "unexpected News official access protocol",
    )
    require(
        protocol.get("status") == "SEALED_AWAITING_ROOT_ACKNOWLEDGEMENT",
        "unexpected access-protocol status",
    )
    require(protocol.get("data_boundary") == "held_out_test", "wrong data boundary")
    require(protocol.get("no_score_based_retry") is True, "retry policy is not frozen")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    require(
        authorization.get("schema") == "sallm.news_official_test_authorization/v1",
        "unexpected authorization marker",
    )
    require(
        authorization.get("status") == "AUTHORIZED_AFTER_VALIDATION_REVIEW",
        "held-out access is not authorized",
    )
    require(
        authorization.get("access_protocol_sha256") == sha256_file(protocol_path),
        "authorization does not bind access protocol",
    )
    require(protocol["dataset"].get("repo") == DATASET_REPO, "dataset repo drift")
    require(
        protocol["dataset"].get("revision") == DATASET_REVISION,
        "dataset revision drift",
    )
    gate = protocol["validation_gate"]
    for key in ("chain_verification", "prompt_selection", "tie_sensitivity"):
        path = Path(gate[key]["path"])
        require(path.is_file(), f"missing validation gate file: {key}")
        require(
            sha256_file(path) == gate[key]["sha256"],
            f"validation gate hash mismatch: {key}",
        )
    verification = json.loads(Path(gate["chain_verification"]["path"]).read_text())
    selection = json.loads(Path(gate["prompt_selection"]["path"]).read_text())
    tie_sensitivity = json.loads(Path(gate["tie_sensitivity"]["path"]).read_text())
    require(
        verification.get("status") == "VALIDATION_CHAIN_VERIFIED",
        "validation chain is not verified",
    )
    require(verification.get("test_accessed") is False, "validation touched test")
    require(
        selection.get("data_boundary") == "validation_only",
        "wrong selection boundary",
    )
    require(selection.get("test_accessed") is False, "prompt selection touched test")
    require(
        tie_sensitivity.get("status") == "VALIDATION_CHAIN_INDEPENDENTLY_VERIFIED",
        "tie-sensitivity validation failed",
    )
    require(tie_sensitivity.get("test_accessed") is False, "tie audit touched test")
    require(
        tie_sensitivity["tie_policy"]["shared_prompt_changes_under_reverse"] is False,
        "reverse tie order changes prompt selection",
    )
    require(
        tie_sensitivity["tie_policy"]["architecture_ranking_changes_under_reverse"]
        is False,
        "reverse tie order changes architecture ranking",
    )
    require(
        protocol.get("tie_policy")
        == {
            "labels": FROZEN_LABEL_ORDER,
            "rule": "first_label_in_frozen_order",
            "exact_tolerance": 1e-12,
        },
        "frozen tie policy drift",
    )
    require(
        tie_sensitivity["tie_policy"]["frozen_order"] == FROZEN_LABEL_ORDER,
        "tie-audit label order drift",
    )
    selected = {
        language: item["prompt"] for language, item in selection["selections"].items()
    }
    require(
        selected == protocol["selected_prompts"],
        "selected prompts differ from gate",
    )

    files = protocol["dataset"]["files"]
    require(set(files) == {"eng", "xho"}, "test language coverage drift")
    for language, spec in files.items():
        require(
            spec.get("rows") == LANGUAGE_ROWS[language],
            f"test row count drift for {language}",
        )
        url = spec["url"]
        require(
            url.endswith(f"/data/{language}/test.tsv"),
            f"refusing non-test URL for {language}",
        )
        require(
            f"/resolve/{protocol['dataset']['revision']}/" in url,
            f"unpinned test URL for {language}",
        )
    require(not destination.exists(), f"test destination exists: {destination}")
    destination.mkdir(parents=True)
    access_started = {
        "schema": "sallm.news_official_test_access_started/v1",
        "started_at_utc": now().isoformat(),
        "protocol_sha256": sha256_file(protocol_path),
        "authorization_sha256": sha256_file(authorization_path),
        "wording": (
            "single prospective corrected replacement; not first-ever test access"
        ),
        "no_score_based_retry": True,
    }
    started_path = destination / "ACCESS_STARTED.json"
    started_path.write_text(
        json.dumps(access_started, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_files = {}
    for language in ("eng", "xho"):
        spec = files[language]
        with opener(spec["url"], timeout=120) as response:
            payload = response.read()
        rows, observed = _parse(
            payload,
            language=language,
            expected_rows=int(spec["rows"]),
        )
        path = destination / f"{language}-test.tsv"
        path.write_bytes(payload)
        manifest_files[language] = {
            "path": str(path),
            "url": spec["url"],
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
            "rows": len(rows),
            "columns": ["category", "headline", "text", "url"],
            "observed_labels": observed,
        }
    manifest = {
        "schema": "sallm.news_official_test_dataset/v1",
        "dataset": protocol["dataset"]["repo"],
        "revision": protocol["dataset"]["revision"],
        "split": "test",
        "metrics_computed": False,
        "access_started_sha256": sha256_file(started_path),
        "files": manifest_files,
    }
    manifest_path = destination / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = materialize(
        protocol_path=args.protocol.resolve(),
        authorization_path=args.authorization.resolve(),
        destination=args.output_root.resolve(),
    )
    print(
        "NEWS_OFFICIAL_TEST_MATERIALIZED "
        f"{args.output_root.resolve()} rows="
        f"{sum(item['rows'] for item in manifest['files'].values())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
