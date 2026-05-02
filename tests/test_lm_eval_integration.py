from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sallm.config import ModelEvalConfig
from sallm.evaluation import lm_eval_runner
from sallm.evaluation.config import TaskPack
from sallm.evaluation.registry import (
    RERANK_LM_EVAL_TASK_DIR,
    load_rerank_task_pack,
    load_task_pack,
)


def test_task_pack_keeps_task_manager_kwargs_out_of_evaluator_kwargs() -> None:
    pack = TaskPack(
        name="demo",
        tasks=["demo_task"],
        lm_eval_kwargs={
            "limit": 10,
            "include_path": "src/conf/eval/lm_eval_tasks/sib_validation",
            "include_defaults": False,
        },
        task_manager_kwargs={
            "include_defaults": True,
            "metadata": {"source": "task-manager"},
        },
    )

    assert pack.to_evaluator_kwargs() == {
        "tasks": ["demo_task"],
        "batch_size": "auto:4",
        "num_fewshot": 0,
        "max_batch_size": 64,
        "limit": 10,
    }
    assert pack.to_task_manager_kwargs() == {
        "include_path": "src/conf/eval/lm_eval_tasks/sib_validation",
        "include_defaults": True,
        "metadata": {"source": "task-manager"},
    }


def test_resolve_include_paths_uses_repo_paths_without_site_package_shims(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    resolved_paths = lm_eval_runner._resolve_include_paths(
        "src/conf/eval/lm_eval_tasks/sib_validation"
    )

    expected_path = (
        lm_eval_runner.PROJECT_ROOT / "src/conf/eval/lm_eval_tasks/sib_validation"
    ).resolve()
    assert resolved_paths == [str(expected_path)]
    assert ".venv" not in resolved_paths[0]


def test_resolve_include_paths_rejects_missing_paths(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_tasks"

    with pytest.raises(FileNotFoundError, match="lm-eval include path"):
        lm_eval_runner._resolve_include_paths(str(missing_path))


def test_load_task_pack_rejects_validation_packs_from_final_eval_scope() -> None:
    with pytest.raises(ValueError, match="validation-scoped"):
        load_task_pack("sib_xho_val")


def test_load_rerank_task_pack_loads_validation_pack() -> None:
    pack = load_rerank_task_pack("masakhaner_xho_val")

    assert pack.name == "masakhaner_xho_val"
    assert pack.tasks[0] == "sallm_masakhaner_xh_prompt_1_val"


def test_run_pack_passes_repo_task_paths_to_task_manager(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}

    class FakeTaskManager:
        def __init__(self, **kwargs: Any) -> None:
            self.include_path = kwargs.get("include_path")
            calls["task_manager_kwargs"] = kwargs

    def fake_simple_evaluate(**kwargs: Any) -> dict[str, Any]:
        calls["eval_kwargs"] = kwargs
        return {"results": {"demo": {"acc": 1.0}}, "metrics": {"acc": 1.0}}

    def fake_prepare_tokenizer_for_lm_eval(*args: Any) -> None:
        return None

    monkeypatch.setattr(lm_eval_runner, "TaskManager", FakeTaskManager)
    monkeypatch.setattr(
        lm_eval_runner.evaluator,
        "simple_evaluate",
        fake_simple_evaluate,
    )
    monkeypatch.setattr(
        lm_eval_runner,
        "_prepare_tokenizer_for_lm_eval",
        fake_prepare_tokenizer_for_lm_eval,
    )

    summary = lm_eval_runner._run_pack(
        "masakhaner_xho_val",
        ModelEvalConfig(checkpoint="org/model", device="cpu"),
        tmp_path / "out",
        tmp_path / "work",
        None,
        "org/model",
        None,
        "rerank",
    )

    expected_path = str(RERANK_LM_EVAL_TASK_DIR.resolve())
    assert calls["task_manager_kwargs"]["include_path"] == [expected_path]
    assert calls["eval_kwargs"]["task_manager"].include_path == [expected_path]
    assert ".venv" not in expected_path
    assert summary["type"] == "lm_eval"
    assert summary["task_pack_scope"] == "rerank"
    assert (tmp_path / "out" / "masakhaner_xho_val" / "results.json").exists()
