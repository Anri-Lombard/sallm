from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/run_mamba_news_common_validation.py"
PROTOCOL_FILE = REPO / ".audit/mamba_news_common_validation_protocol_20260914.json"
needs_local_protocol = pytest.mark.skipif(
    not PROTOCOL_FILE.is_file(),
    reason=".audit/ is gitignored local protocol staging",
)


def load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_mamba_news_common_validation", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    return load_runner()


@needs_local_protocol
def test_protocol_check_is_sealed_and_validation_only() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            str(REPO),
            "--check",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "MAMBA_NEWS_COMMON_VALIDATION_PROTOCOL_OK"


@needs_local_protocol
def test_protocol_declares_only_revision_pinned_dev_files() -> None:
    protocol = json.loads(
        (REPO / ".audit/mamba_news_common_validation_protocol_20260914.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["data_boundary"] == "validation_only"
    assert protocol["test_access_allowed"] is False
    for spec in protocol["dataset"]["allowed_files"]:
        assert spec["url"].endswith(f"/data/{spec['language']}/dev.tsv")
        assert "/test.tsv" not in spec["url"].lower()


def _tsv(rows: list[tuple[str, str, str, str]]) -> bytes:
    lines = ["category\theadline\ttext\turl"]
    lines.extend("\t".join(row) for row in rows)
    return ("\n".join(lines) + "\n").encode()


def _file_spec(language: str, payload: bytes) -> dict[str, object]:
    rows = payload.decode().strip().splitlines()[1:]
    labels = sorted({row.split("\t", 1)[0] for row in rows})
    return {
        "language": language,
        "url": f"https://example.invalid/data/{language}/dev.tsv",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "rows": len(rows),
        "columns": ["category", "headline", "text", "url"],
        "observed_labels": labels,
    }


def test_materializer_opens_only_the_two_exact_dev_urls(
    runner: ModuleType, tmp_path: Path
) -> None:
    eng = _tsv(
        [
            ("business", "h1", "t1", "u1"),
            ("sports", "h2", "t2", "u2"),
        ]
    )
    xho = _tsv([("health", "h3", "t3", "u3")])
    specs = [_file_spec("eng", eng), _file_spec("xho", xho)]
    payloads = {specs[0]["url"]: eng, specs[1]["url"]: xho}
    opened: list[str] = []

    def fake_open(url: str, *, timeout: int) -> io.BytesIO:
        assert timeout == 120
        opened.append(url)
        return io.BytesIO(payloads[url])

    rows = runner.materialize_validation_files(
        {"allowed_files": specs},
        tmp_path / "validation",
        opener=fake_open,
    )

    assert opened == [specs[0]["url"], specs[1]["url"]]
    assert all(url.endswith("/dev.tsv") for url in opened)
    assert not any("test" in url.lower() for url in opened)
    assert [row["category"] for row in rows["eng"]] == ["business", "sports"]
    assert (tmp_path / "validation/eng-dev.tsv").read_bytes() == eng
    assert (tmp_path / "validation/xho-dev.tsv").read_bytes() == xho


def test_materializer_rejects_non_dev_url_before_opening(
    runner: ModuleType, tmp_path: Path
) -> None:
    payload = _tsv([("business", "h", "t", "u")])
    spec = _file_spec("eng", payload)
    spec["url"] = "https://example.invalid/data/eng/train.tsv"
    opened: list[str] = []

    def fake_open(url: str, *, timeout: int) -> io.BytesIO:
        opened.append(url)
        return io.BytesIO(payload)

    with pytest.raises(runner.ProtocolError, match="non-development URL"):
        runner.materialize_validation_files(
            {"allowed_files": [spec]},
            tmp_path / "validation",
            opener=fake_open,
        )
    assert opened == []


def test_runtime_copy_changes_only_chat_template(
    runner: ModuleType, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    stale = "{%- if add_generation_prompt %}<|assistant|>{%- endif %}\n"
    (source / "chat_template.jinja").write_text(stale, encoding="utf-8")
    (source / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.iterdir()
    }

    runtime = runner.copy_and_normalize_adapter(
        source,
        tmp_path / "runtime",
        expected_source_sha256=hashlib.sha256(stale.encode()).hexdigest(),
        expected_normalized_sha256=hashlib.sha256(
            runner.NORMALIZED_CHAT_TEMPLATE.encode()
        ).hexdigest(),
    )

    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.iterdir()
    }
    assert before == after
    assert (runtime / "adapter_config.json").read_text(encoding="utf-8") == "{}\n"
    assert (runtime / "chat_template.jinja").read_text(
        encoding="utf-8"
    ) == runner.NORMALIZED_CHAT_TEMPLATE


def test_metrics_are_recomputed_independently(runner: ModuleType) -> None:
    metrics = runner.manual_classification_metrics(
        ["a", "a", "b", "b"],
        ["a", "b", "b", "b"],
    )
    assert metrics == pytest.approx(
        {"accuracy": 0.75, "weighted_f1": 11 / 15, "macro_f1": 11 / 15}
    )
    runner.validate_metrics_against_sklearn(
        ["a", "a", "b", "b"],
        ["a", "b", "b", "b"],
        metrics,
    )


def test_mean_token_scoring_differs_from_raw_sum_and_ties_are_stable(
    runner: ModuleType,
) -> None:
    winner, scores = runner.rank_mean_scores(
        ["short", "long"],
        [-2.0, -3.0],
        [1, 3],
    )
    assert winner == "long"
    assert (
        max({"short": -2.0, "long": -3.0}, key={"short": -2.0, "long": -3.0}.get)
        == "short"
    )
    assert scores == {"short": -2.0, "long": -1.0}

    tied, _ = runner.rank_mean_scores(["first", "second"], [-1.0, -2.0], [1, 2])
    assert tied == "first"

    with pytest.raises(runner.ProtocolError, match="non-finite"):
        runner.rank_mean_scores(["first", "second"], [float("nan"), -2.0], [1, 2])


def test_label_round_robin_is_deterministic(runner: ModuleType) -> None:
    rows = [
        {"category": "a", "headline": "a1"},
        {"category": "a", "headline": "a2"},
        {"category": "b", "headline": "b1"},
        {"category": "b", "headline": "b2"},
        {"category": "c", "headline": "c1"},
    ]
    selected = runner.select_label_round_robin(rows, ["a", "b", "c"], 5)
    assert [row["headline"] for row in selected] == ["a1", "b1", "c1", "a2", "b2"]


def _prediction(
    arm: str, language: str, prompt_id: str, source_index: int
) -> dict[str, object]:
    return {
        "arm": arm,
        "language": language,
        "prompt_id": prompt_id,
        "selected_source_index": source_index,
        "gold": "a",
        "prediction": "b",
        "scores": {"a": -2.0, "b": -1.0},
        "token_counts": {"a": 1, "b": 1},
        "top_two_margin": 1.0,
    }


def test_prediction_integrity_checks_completeness_not_class_entropy(
    runner: ModuleType,
) -> None:
    arm = {"id": "mono", "languages": ["eng"]}
    validation_rows = {"eng": [{"category": "a"}, {"category": "b"}]}
    predictions = [
        _prediction("mono", "eng", prompt_id, source_index)
        for source_index in range(2)
        for prompt_id in ("p1", "p2")
    ]

    integrity = runner.validate_prediction_integrity(
        predictions,
        arm=arm,
        validation_rows=validation_rows,
        template_ids=["p1", "p2"],
        labels=["a", "b"],
        limit=0,
    )
    assert integrity["expected_prediction_records"] == 4
    assert integrity["observed_prediction_records"] == 4

    with pytest.raises(runner.ProtocolError, match="completeness"):
        runner.validate_prediction_integrity(
            predictions[:-1],
            arm=arm,
            validation_rows=validation_rows,
            template_ids=["p1", "p2"],
            labels=["a", "b"],
            limit=0,
        )

    invalid = [dict(row) for row in predictions]
    invalid[0]["top_two_margin"] = float("nan")
    with pytest.raises(runner.ProtocolError, match="non-finite"):
        runner.validate_prediction_integrity(
            invalid,
            arm=arm,
            validation_rows=validation_rows,
            template_ids=["p1", "p2"],
            labels=["a", "b"],
            limit=0,
        )
