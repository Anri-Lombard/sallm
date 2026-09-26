"""Frozen coverage and aggregation contract for General validation selection."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from datasets import Dataset

GENERAL_SELECTION_PROTOCOL = "equal_family_assistant_token_nll_v1"
GENERAL_TASKS = ("sib", "news", "ner", "pos", "afrihg", "t2x")
GENERAL_RAW_ROWS = {
    "sib": 594,
    "news": 619,
    "ner": 2_152,
    "pos": 450,
    "afrihg": 3_082,
    "t2x": 460,
}
GENERAL_TEMPLATES = {
    "sib": tuple(f"sib_topic_classification/lm_eval_p{i}" for i in range(1, 6)),
    "news": tuple(f"masakhane_news_classification/lm_eval_p{i}" for i in range(1, 6)),
    "ner": tuple(
        f"masakhane_named_entity_recognition/lm_eval_p{i}" for i in range(1, 6)
    ),
    "pos": tuple(f"masakhane_pos_tagging/lm_eval_p{i}" for i in range(1, 5)),
    "afrihg": ("afrihg_headline/v1",),
    "t2x": ("t2x_verbalisation/v1",),
}
GENERAL_PROCESSED_ROWS = {
    task: GENERAL_RAW_ROWS[task] * len(GENERAL_TEMPLATES[task])
    for task in GENERAL_TASKS
}
GENERAL_TOTAL_PROCESSED_ROWS = sum(GENERAL_PROCESSED_ROWS.values())


def validate_general_coverage(dataset: Dataset) -> dict[str, Any]:
    required = {"task_name", "template_id", "messages"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"General validation is missing columns {sorted(missing)}")

    task_values = list(dataset["task_name"])
    if any(not isinstance(task, str) or not task for task in task_values):
        raise ValueError("General validation contains a null or empty task_name.")
    task_counts = Counter(task_values)
    expected_task_counts = Counter(GENERAL_PROCESSED_ROWS)
    if task_counts != expected_task_counts:
        raise ValueError(
            "General validation task coverage mismatch: "
            f"observed={dict(task_counts)}, expected={dict(expected_task_counts)}"
        )

    template_counts: dict[str, dict[str, int]] = {}
    for task in GENERAL_TASKS:
        task_dataset = dataset.filter(
            lambda row, _task=task: row["task_name"] == _task,
            load_from_cache_file=False,
        )
        observed_templates = Counter(
            str(value) for value in task_dataset["template_id"]
        )
        expected_templates = Counter(
            {template: GENERAL_RAW_ROWS[task] for template in GENERAL_TEMPLATES[task]}
        )
        if observed_templates != expected_templates:
            raise ValueError(
                f"General validation template coverage mismatch for {task}: "
                f"observed={dict(observed_templates)}, "
                f"expected={dict(expected_templates)}"
            )
        template_counts[task] = dict(sorted(observed_templates.items()))

    if "lang" not in dataset.column_names:
        raise ValueError("General validation must preserve language labels.")
    afrihg = dataset.filter(
        lambda row: row["task_name"] == "afrihg",
        load_from_cache_file=False,
    )
    afrihg_languages = Counter(afrihg["lang"])
    if set(afrihg_languages) != {"xho", "zul"} or any(
        not language for language in afrihg["lang"]
    ):
        raise ValueError(
            f"AfriHG validation language coverage is invalid: {dict(afrihg_languages)}"
        )

    if len(dataset) != GENERAL_TOTAL_PROCESSED_ROWS:
        raise ValueError(
            f"General validation has {len(dataset)} rows; "
            f"expected {GENERAL_TOTAL_PROCESSED_ROWS}."
        )

    return {
        "protocol": GENERAL_SELECTION_PROTOCOL,
        "raw_rows": dict(GENERAL_RAW_ROWS),
        "processed_rows": dict(GENERAL_PROCESSED_ROWS),
        "total_processed_rows": len(dataset),
        "template_counts": template_counts,
        "afrihg_language_counts": dict(sorted(afrihg_languages.items())),
    }


def compute_equal_family_token_nll(
    summed_nll: dict[str, float],
    valid_token_counts: dict[str, int],
) -> tuple[float, dict[str, float]]:
    if set(summed_nll) != set(GENERAL_TASKS):
        raise ValueError(f"General NLL sums must contain exactly {GENERAL_TASKS}.")
    if set(valid_token_counts) != set(GENERAL_TASKS):
        raise ValueError(f"General token counts must contain exactly {GENERAL_TASKS}.")

    family_nll: dict[str, float] = {}
    for task in GENERAL_TASKS:
        total = float(summed_nll[task])
        count = int(valid_token_counts[task])
        if count <= 0:
            raise ValueError(f"General family {task} has no valid assistant tokens.")
        if not math.isfinite(total):
            raise ValueError(f"General family {task} has non-finite summed NLL.")
        value = total / count
        if not math.isfinite(value):
            raise ValueError(f"General family {task} has non-finite token NLL.")
        family_nll[task] = value

    macro_nll = sum(family_nll.values()) / len(GENERAL_TASKS)
    return macro_nll, family_nll
