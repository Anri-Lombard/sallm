import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_official_task_pack_eval import verify_task_pack_eval


def _write_artifacts(
    root: Path,
    rows_per_task: int = 2,
    *,
    pack: str = "masakhaner_tsn",
    task_prefix: str = "sallm_masakhaner_tn",
    prompt_count: int = 5,
    task_suffix: str = "_test",
    metric: str = "f1,flexible-extract",
) -> None:
    tasks = [
        f"{task_prefix}_prompt_{index}{task_suffix}"
        for index in range(1, prompt_count + 1)
    ]
    results = {task: {"alias": task, metric: 0.5} for task in tasks}
    samples = {
        task: [{"doc_id": index, "f1": 0.5} for index in range(rows_per_task)]
        for task in tasks
    }
    payload = {
        "results": results,
        "n-samples": {
            task: {"original": rows_per_task, "effective": rows_per_task}
            for task in tasks
        },
        "samples": samples,
    }
    pack_root = root / pack
    pack_root.mkdir(parents=True)
    result_path = pack_root / "results.json"
    result_path.write_text(json.dumps(payload))
    (root / "evaluation_summary.json").write_text(
        json.dumps(
            [
                {
                    "type": "lm_eval",
                    "task_pack": pack,
                    "tasks": tasks,
                    "results": results,
                    "metrics": {},
                    "result_path": str(result_path),
                }
            ]
        )
    )


def test_verifies_exact_task_pack_coverage(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)

    result = verify_task_pack_eval(
        tmp_path,
        expected_pack="masakhaner_tsn",
        expected_task_prefix="sallm_masakhaner_tn",
        expected_rows=10,
    )

    assert result["verified"] is True
    assert result["rows"] == 10
    assert result["metric_values_included"] is False


def test_rejects_wrong_coverage(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)

    with pytest.raises(ValueError, match="expected 11 samples, found 10"):
        verify_task_pack_eval(
            tmp_path,
            expected_pack="masakhaner_tsn",
            expected_task_prefix="sallm_masakhaner_tn",
            expected_rows=11,
        )


def test_verifies_explicit_four_prompt_pack(tmp_path: Path) -> None:
    _write_artifacts(
        tmp_path,
        pack="masakhapos_tsn",
        task_prefix="sallm_masakhapos_tsn",
        prompt_count=4,
        task_suffix="",
        metric="token_accuracy,flexible-extract",
    )

    result = verify_task_pack_eval(
        tmp_path,
        expected_pack="masakhapos_tsn",
        expected_task_prefix="sallm_masakhapos_tsn",
        expected_prompt_count=4,
        expected_task_suffix="",
        required_metric="token_accuracy,flexible-extract",
        expected_rows=8,
    )

    assert result["tasks"] == [
        f"sallm_masakhapos_tsn_prompt_{index}" for index in range(1, 5)
    ]
    assert result["required_metric_present"] == "token_accuracy,flexible-extract"
