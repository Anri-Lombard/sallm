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
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def _save_test_tokenizer(path: Path, chat_template: str | None = None) -> None:
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = chat_template
    tokenizer.save_pretrained(path)


def test_raw_lm_eval_tokenizer_adds_bos_without_eos_and_preserves_continuation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _save_test_tokenizer(source)

    prepared_path = lm_eval_runner._prepare_tokenizer_for_lm_eval(
        str(source), tmp_path / "prepared", require_chat_template=False
    )

    assert prepared_path is not None
    tokenizer = AutoTokenizer.from_pretrained(prepared_path, local_files_only=True)
    context_ids = tokenizer.encode("hello", add_special_tokens=True)
    whole_ids = tokenizer.encode("hello world", add_special_tokens=True)
    continuation_ids = whole_ids[len(context_ids) :]
    assert context_ids == [tokenizer.bos_token_id, 3]
    assert continuation_ids == [4]
    assert tokenizer.eos_token_id not in whole_ids


def test_lm_eval_chat_tokenizer_preserves_model_template(tmp_path: Path) -> None:
    source = tmp_path / "source"
    model_template = "MODEL-SPECIFIC {{ messages[0]['content'] }}"
    _save_test_tokenizer(source, chat_template=model_template)

    prepared_path = lm_eval_runner._prepare_tokenizer_for_lm_eval(
        str(source), tmp_path / "prepared", require_chat_template=True
    )

    assert prepared_path is not None
    tokenizer = AutoTokenizer.from_pretrained(prepared_path, local_files_only=True)
    assert tokenizer.chat_template == model_template


def test_raw_lm_eval_tokenizer_fails_closed_when_loading_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        lm_eval_runner.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("load failed")),
    )

    with pytest.raises(RuntimeError, match="raw lm-eval tokenizer"):
        lm_eval_runner._prepare_tokenizer_for_lm_eval(
            "missing/model", tmp_path / "prepared", require_chat_template=False
        )


def test_raw_lm_eval_tokenizer_fails_closed_when_saving_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _save_test_tokenizer(source)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    monkeypatch.setattr(
        lm_eval_runner.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        tokenizer,
        "save_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("save failed")),
    )

    with pytest.raises(RuntimeError, match="raw lm-eval tokenizer"):
        lm_eval_runner._prepare_tokenizer_for_lm_eval(
            str(source), tmp_path / "prepared", require_chat_template=False
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


def test_prepare_tokenizer_for_lm_eval_saves_transformers_tokenizer(
    tmp_path: Path,
) -> None:
    backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
    )
    source = tmp_path / "source"
    tokenizer.save_pretrained(source)

    prepared = lm_eval_runner._prepare_tokenizer_for_lm_eval(
        str(source),
        tmp_path / "cache",
        require_chat_template=True,
    )

    assert prepared is not None
    loaded = AutoTokenizer.from_pretrained(prepared, local_files_only=True)
    assert loaded is not None
    assert loaded.chat_template == lm_eval_runner._fallback_chat_template()


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


def test_run_pack_rejects_lm_eval_response_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeTaskManager:
        def __init__(self, **kwargs: Any) -> None:
            self.include_path = kwargs.get("include_path")

    monkeypatch.setattr(lm_eval_runner, "TaskManager", FakeTaskManager)
    monkeypatch.setattr(
        lm_eval_runner.evaluator,
        "simple_evaluate",
        lambda **kwargs: {"results": {}, "metrics": {}},
    )
    monkeypatch.setattr(
        lm_eval_runner,
        "_prepare_tokenizer_for_lm_eval",
        lambda *args: None,
    )

    with pytest.raises(ValueError, match="response caching is disabled"):
        lm_eval_runner._run_pack(
            "masakhaner_xho_val",
            ModelEvalConfig(checkpoint=str(tmp_path), device="cpu"),
            tmp_path / "out",
            tmp_path / "work",
            {"use_cache": str(tmp_path / "cache")},
            str(tmp_path),
            None,
            "rerank",
        )


def test_run_pack_resolves_raw_override_before_tokenizer_and_model_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}

    def fake_prepare_tokenizer(
        pretrained_path: str,
        cache_root: Path,
        require_chat_template: bool,
    ) -> str:
        calls["require_chat_template"] = require_chat_template
        calls["cache_root"] = cache_root
        return "/prepared/tokenizer"

    def fake_simple_evaluate(**kwargs: Any) -> dict[str, Any]:
        calls["eval_kwargs"] = kwargs
        return {"results": {}, "metrics": {}}

    monkeypatch.setattr(
        lm_eval_runner,
        "_prepare_tokenizer_for_lm_eval",
        fake_prepare_tokenizer,
    )
    monkeypatch.setattr(
        lm_eval_runner.evaluator,
        "simple_evaluate",
        fake_simple_evaluate,
    )

    summary = lm_eval_runner._run_pack(
        "masakhaner_xho_val",
        ModelEvalConfig(checkpoint="org/model", device="cpu"),
        tmp_path / "out",
        tmp_path / "work",
        {"apply_chat_template": False},
        "org/model",
        None,
        "rerank",
    )

    assert calls["require_chat_template"] is False
    assert calls["cache_root"].name == "_tokenizer_raw"
    assert calls["eval_kwargs"]["apply_chat_template"] is False
    assert "add_bos_token=true" in calls["eval_kwargs"]["model_args"]
    assert summary["apply_chat_template"] is False
    assert summary["add_bos_token"] is True
