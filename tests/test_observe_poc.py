from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "observe_poc.py"
SPEC = importlib.util.spec_from_file_location("observe_poc", MODULE_PATH)
observe_poc = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(observe_poc)


def test_sections_metrics_and_gpu_count() -> None:
    sections = observe_poc.split_sections(
        "quota ok\n"
        "__SALLM_SECTION__SQUEUE\n"
        "961688|sallm-gdn-2g|PENDING|0:00|2-00:00:00|1|(Priority)|l40s|gpu:l40s:2\n"
    )
    assert sections["QUOTA"] == "quota ok"
    assert observe_poc.parse_squeue(sections["SQUEUE"])[0]["job_id"] == "961688"
    assert observe_poc.gpu_count("billing=15,gres/gpu:l40s=2") == 2
    assert observe_poc.gpu_count("gpu:l40s:4") == 4

    events = observe_poc.extract_training_events(
        [
            {
                "path": "slurm.out",
                "tail": "{'loss': 1.25, 'epoch': 0.5}\nValueError: bad config",
            }
        ]
    )
    assert events["metrics"][0]["loss"] == 1.25
    assert events["errors"][0]["line"] == "ValueError: bad config"

    stats = observe_poc.build_stats(
        [
            {"start": "2026-06-20T10:00:00", "gpu_hours": 2.0},
            {"start": "None", "gpu_hours": 99.0},
            {"start": "2026-06-22T10:00:00", "gpu_hours": 3.0},
            {"start": "2026-06-23T10:00:00", "gpu_hours": 4.0},
        ]
    )
    assert stats["gpu_hours_7d"] == 108.0
    assert stats["training_streak_days"] == 2
