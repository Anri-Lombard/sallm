import pytest
from datasets import Dataset
from sallm.training.general_validation import (
    GENERAL_PROCESSED_ROWS,
    GENERAL_RAW_ROWS,
    GENERAL_TASKS,
    GENERAL_TEMPLATES,
    GENERAL_TOTAL_PROCESSED_ROWS,
    compute_equal_family_token_nll,
    validate_general_coverage,
)


def test_equal_family_token_nll_is_not_sample_weighted() -> None:
    summed = {task: float(index + 1) * 10 for index, task in enumerate(GENERAL_TASKS)}
    counts = {task: (index + 1) * 10 for index, task in enumerate(GENERAL_TASKS)}

    macro, per_family = compute_equal_family_token_nll(summed, counts)

    assert per_family == {task: 1.0 for task in GENERAL_TASKS}
    assert macro == 1.0


def test_equal_family_token_nll_rejects_missing_family() -> None:
    summed = {task: 1.0 for task in GENERAL_TASKS if task != "afrihg"}
    counts = {task: 1 for task in GENERAL_TASKS if task != "afrihg"}

    with pytest.raises(ValueError, match="exactly"):
        compute_equal_family_token_nll(summed, counts)


def test_general_coverage_contract_exact_counts() -> None:
    rows = []
    for task in GENERAL_TASKS:
        for template in GENERAL_TEMPLATES[task]:
            for index in range(GENERAL_RAW_ROWS[task]):
                language = "xho" if task != "afrihg" or index % 2 == 0 else "zul"
                rows.append(
                    {
                        "task_name": task,
                        "template_id": template,
                        "lang": language,
                        "messages": [],
                    }
                )

    manifest = validate_general_coverage(Dataset.from_list(rows))

    assert manifest["processed_rows"] == GENERAL_PROCESSED_ROWS
    assert manifest["total_processed_rows"] == GENERAL_TOTAL_PROCESSED_ROWS == 22_167
