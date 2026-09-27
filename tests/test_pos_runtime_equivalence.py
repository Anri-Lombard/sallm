import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_pos_row_batch_equivalence import verify  # noqa: E402


def test_pos_runtime_gate_accepts_matching_faster_row_batch() -> None:
    row = {
        "cell": "xho/p1",
        "tokens": 2,
        "predictions": ["NOUN", "VERB"],
        "gold": ["NOUN", "VERB"],
        "correct": 2,
        "selected_sequence_score": -1.0,
    }
    result = {
        "rows": [dict(row) for _ in range(12)],
        "cell_counts": {"xho/p1": {"correct": 24, "total": 24}},
        "cell_metrics": {"xho_p1_token_accuracy": 1.0},
        "all_token_accuracy": 1.0,
    }
    payload = {
        "schema": "sallm_pos_row_batch_equivalence/v1",
        "results": {
            "serial": {
                **result,
                "implementation": "full_prefix_v1",
                "row_batch_size": 1,
                "elapsed_seconds": 12.0,
            },
            "batched": {
                **result,
                "implementation": "row_batch_full_prefix_v1",
                "row_batch_size": 8,
                "elapsed_seconds": 3.0,
            },
        },
    }

    verified = verify(payload)

    assert verified["passed"] is True
    assert verified["speedup"] == 4.0
