from pathlib import Path
from types import SimpleNamespace

from lm_eval.tasks import TaskManager
from sallm.evaluation import lm_eval_runner
from sallm.evaluation.config import TaskPack
from sallm.evaluation.lm_eval_runner import (
    _format_model_args,
    _lexical_lm_eval_tasks_root,
    _materialize_model_for_lm_eval,
    _prepare_include_paths,
    _run_pack,
    _set_eval_safe_model_config,
    _to_serializable,
    _use_exact_xlstm_head_dims,
)
from sallm.evaluation.registry import load_task_pack


def test_lm_eval_tasks_root_preserves_symlinked_venv_path(tmp_path: Path) -> None:
    physical = tmp_path / "physical" / "lm_eval" / "tasks"
    physical.mkdir(parents=True)
    lexical = tmp_path / "venv" / "lm_eval" / "tasks"
    lexical.parent.mkdir(parents=True)
    lexical.symlink_to(physical, target_is_directory=True)

    root = _lexical_lm_eval_tasks_root(str(lexical / "__init__.py"))

    assert root == lexical
    assert root != lexical.resolve()


def test_format_model_args_appends_extra_values() -> None:
    model_args = _format_model_args(
        pretrained_path="owner/model",
        dtype="bfloat16",
        peft_adapter=None,
        extra_model_args={
            "logits_cache": False,
            "revision": "main",
            "unused": None,
        },
    )

    assert model_args == (
        "pretrained=owner/model,trust_remote_code=true,add_bos_token=false,"
        "dtype=bfloat16,"
        "logits_cache=false,revision=main"
    )


def test_format_model_args_allows_add_bos_override() -> None:
    model_args = _format_model_args(
        pretrained_path="owner/model",
        dtype=None,
        peft_adapter=None,
        extra_model_args={"add_bos_token": True},
    )

    assert model_args.endswith("add_bos_token=true")
    assert model_args.count("add_bos_token=") == 1


def test_format_model_args_enables_bos_for_raw_base_prompt() -> None:
    model_args = _format_model_args(
        pretrained_path="owner/model",
        dtype=None,
        peft_adapter=None,
        default_add_bos_token=True,
    )

    assert model_args.endswith("add_bos_token=true")
    assert model_args.count("add_bos_token=") == 1


def test_to_serializable_names_callable_values() -> None:
    def task_handler() -> None:
        pass

    serialized = _to_serializable({"handler": task_handler})

    assert serialized["handler"].endswith(
        ".test_to_serializable_names_callable_values.<locals>.task_handler"
    )


def test_run_pack_summary_records_effective_fewshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "sallm.evaluation.lm_eval_runner._load_pack",
        lambda *_: TaskPack(name="bench", tasks=["task"], fewshot=0),
    )
    monkeypatch.setattr(
        "sallm.evaluation.lm_eval_runner._prepare_tokenizer_for_lm_eval",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "sallm.evaluation.lm_eval_runner._format_model_args",
        lambda **_: "pretrained=owner/model",
    )
    monkeypatch.setattr(
        "sallm.evaluation.lm_eval_runner.evaluator.simple_evaluate",
        lambda **_: {"config": {"num_fewshot": 3}, "results": {}, "metrics": {}},
    )
    model_cfg = SimpleNamespace(
        dtype="bfloat16",
        device="cuda:0",
        peft_adapter=None,
        tie_word_embeddings=None,
        lm_eval_model_args={},
    )

    summary = _run_pack(
        "bench",
        model_cfg,
        tmp_path,
        tmp_path / "work",
        {"num_fewshot": 3, "apply_chat_template": False},
        "owner/model",
        None,
        "eval",
    )

    assert summary["fewshot"] == 3
    assert summary["apply_chat_template"] is False


def test_xlstm_lm_eval_uses_inference_mode() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="xlstm",
            mode="train",
            return_last_states=True,
            dtype="bfloat16",
        )
    )

    _set_eval_safe_model_config(model)

    assert model.config.mode == "inference"
    assert model.config.return_last_states is True
    assert model.config.inference_state_dtype == "bfloat16"


def test_xlstm_cache_dimensions_match_external_runtime() -> None:
    from transformers.models.xlstm.configuration_xlstm import xLSTMConfig
    from transformers.models.xlstm.modeling_xlstm import xLSTMCache

    config = xLSTMConfig(
        hidden_size=736,
        embedding_dim=736,
        num_heads=4,
        num_hidden_layers=1,
    )

    _use_exact_xlstm_head_dims(config)
    cell_state, normalizer_state, _ = xLSTMCache(config, 2).rnn_state[0]

    assert config.qk_head_dim == 92
    assert config.v_head_dim == 184
    assert cell_state.shape == (2, 4, 92, 184)
    assert normalizer_state.shape == (2, 4, 92)


def test_adapter_free_xlstm_materializes_eval_safe_checkpoint(
    tmp_path, monkeypatch
) -> None:
    config = SimpleNamespace(
        model_type="xlstm",
        mode="train",
        return_last_states=False,
        dtype="bfloat16",
        tie_word_embeddings=False,
    )
    saved_model_to = []
    saved_tokenizer_to = []

    class Model:
        def __init__(self, model_config) -> None:
            self.config = model_config

        def save_pretrained(self, path, safe_serialization=True) -> None:
            saved_model_to.append((path, safe_serialization))
            if safe_serialization:
                raise RuntimeError("weights contained shared tensors")

    model = SimpleNamespace(
        config=config,
        save_pretrained=Model(config).save_pretrained,
    )
    tokenizer = SimpleNamespace(
        save_pretrained=saved_tokenizer_to.append,
    )
    monkeypatch.setattr(
        "sallm.evaluation.lm_eval_runner.AutoConfig.from_pretrained",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        "sallm.evaluation.lm_eval_runner.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        "sallm.evaluation.lm_eval_runner.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    model_cfg = SimpleNamespace(
        checkpoint="owner/xlstm",
        peft_adapter=None,
        dtype="bfloat16",
    )

    pretrained, adapter = _materialize_model_for_lm_eval(model_cfg, tmp_path)

    expected = tmp_path / "eval_safe_base_model"
    assert pretrained == str(expected)
    assert adapter is None
    assert config.mode == "inference"
    assert config.return_last_states is True
    assert config.inference_state_dtype == "bfloat16"
    assert saved_model_to == [(expected, True), (expected, False)]
    assert saved_tokenizer_to == [expected]


def test_lm_eval_materialization_registers_fla_before_auto_config(
    tmp_path, monkeypatch
) -> None:
    events = []
    monkeypatch.setattr(
        lm_eval_runner,
        "register_fla_gated_deltanet",
        lambda: events.append("register") or True,
    )
    monkeypatch.setattr(
        lm_eval_runner.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: events.append("config")
        or SimpleNamespace(model_type="llama"),
    )
    model_cfg = SimpleNamespace(checkpoint="owner/model", peft_adapter=None)

    pretrained, adapter = _materialize_model_for_lm_eval(model_cfg, tmp_path)

    assert (pretrained, adapter) == ("owner/model", None)
    assert events == ["register", "config"]


def test_lm_eval_preserves_unmerged_peft_adapter(tmp_path, monkeypatch) -> None:
    class Tokenizer:
        def __len__(self) -> int:
            return 19

        def save_pretrained(self, path) -> None:
            pass

    tokenizer = Tokenizer()
    resized_to = []
    saved_to = []

    def save_pretrained(path, safe_serialization=True) -> None:
        saved_to.append((path, safe_serialization))
        if safe_serialization:
            raise RuntimeError("weights contained shared tensors")

    loaded_model = SimpleNamespace(
        resize_token_embeddings=resized_to.append,
        save_pretrained=save_pretrained,
    )
    monkeypatch.setattr(
        "sallm.evaluation.lm_eval_runner.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "sallm.evaluation.lm_eval_runner.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: loaded_model,
    )
    model = SimpleNamespace(
        checkpoint="owner/model",
        peft_adapter="/checkpoints/adapter",
        merge_lora=False,
        dtype="bfloat16",
    )

    pretrained, adapter = _materialize_model_for_lm_eval(model, tmp_path)

    assert pretrained == str(tmp_path / "resized_base_model")
    assert adapter == "/checkpoints/adapter"
    assert resized_to == [19]
    expected = tmp_path / "resized_base_model"
    assert saved_to == [(expected, True), (expected, False)]


def test_masakhaner_test_tasks_index_and_resolve_inheritance(
    tmp_path, monkeypatch
) -> None:
    pack = load_task_pack("masakhaner_all")
    isolated_tasks_root = tmp_path / "lm_eval" / "tasks"
    monkeypatch.setattr(lm_eval_runner, "LM_EVAL_TASKS_ROOT", isolated_tasks_root)

    prepared_paths = _prepare_include_paths(pack.task_manager_kwargs["include_path"])
    try:
        task_manager = TaskManager(include_path=prepared_paths, include_defaults=False)
        matched_tasks = task_manager.match_tasks(pack.tasks)
        task_configs = [task_manager._get_config(task) for task in pack.tasks]
    finally:
        for prepared_path in prepared_paths:
            link_path = Path(prepared_path)
            if link_path.is_symlink():
                link_path.unlink()

    assert set(matched_tasks) == set(pack.tasks)
    assert all(
        config["dataset_path"] == "anrilombard/masakhaner-x-parquet"
        and config["test_split"] == "test"
        and config["validation_split"] is None
        for config in task_configs
    )
