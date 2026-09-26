import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_official_generation_eval import verify_generation_eval  # noqa: E402


def _write_eval(root: Path, *, rows: int = 2) -> None:
    metrics = {"eval/t2x_xho/all_chrf": 12.5}
    task_root = root / "t2x_xho"
    task_root.mkdir(parents=True)
    summary = {"task": "t2x_xho", "metrics": metrics}
    (root / "evaluation_summary.json").write_text(
        json.dumps([summary]), encoding="utf-8"
    )
    (task_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (task_root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with (task_root / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(rows):
            handle.write(
                json.dumps(
                    {
                        "task": "t2x_xho",
                        "language": "xho",
                        "prediction": f"prediction {index}",
                        "reference": f"reference {index}",
                    }
                )
                + "\n"
            )


def test_verify_generation_eval_accepts_exact_structural_artifact(
    tmp_path: Path,
) -> None:
    _write_eval(tmp_path)

    result = verify_generation_eval(
        tmp_path,
        expected_task="t2x_xho",
        expected_language="xho",
        expected_rows=2,
        required_metric="eval/t2x_xho/all_chrf",
    )

    assert result["verified"] is True
    assert result["rows"] == 2
    assert result["metric_values_included"] is False
    assert set(result["artifact_sha256"]) == {
        "evaluation_summary.json",
        "t2x_xho/examples.jsonl",
        "t2x_xho/metrics.json",
        "t2x_xho/summary.json",
    }


def test_verify_generation_eval_rejects_wrong_coverage(tmp_path: Path) -> None:
    _write_eval(tmp_path, rows=1)

    with pytest.raises(ValueError, match="expected 2 examples, found 1"):
        verify_generation_eval(
            tmp_path,
            expected_task="t2x_xho",
            expected_language="xho",
            expected_rows=2,
            required_metric="eval/t2x_xho/all_chrf",
        )


def test_verify_generation_eval_rejects_metric_disagreement(tmp_path: Path) -> None:
    _write_eval(tmp_path)
    (tmp_path / "t2x_xho" / "metrics.json").write_text(
        json.dumps({"eval/t2x_xho/all_chrf": 9.0}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="metric artifacts disagree"):
        verify_generation_eval(
            tmp_path,
            expected_task="t2x_xho",
            expected_language="xho",
            expected_rows=2,
            required_metric="eval/t2x_xho/all_chrf",
        )


def test_verify_generation_eval_preserves_unicode_line_separator(
    tmp_path: Path,
) -> None:
    _write_eval(tmp_path, rows=1)
    examples_path = tmp_path / "t2x_xho" / "examples.jsonl"
    row = json.loads(examples_path.read_text(encoding="utf-8"))
    row["prediction"] = "first line\u2028second line"
    examples_path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = verify_generation_eval(
        tmp_path,
        expected_task="t2x_xho",
        expected_language="xho",
        expected_rows=1,
        required_metric="eval/t2x_xho/all_chrf",
    )

    assert result["rows"] == 1
