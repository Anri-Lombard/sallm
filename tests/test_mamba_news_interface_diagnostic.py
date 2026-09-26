from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/run_mamba_news_interface_diagnostic.py"


def load_diagnostic() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mamba_news_diagnostic", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generation_truncation_reserves_budget_and_keeps_the_suffix() -> None:
    diagnostic = load_diagnostic()

    kept, removed = diagnostic._left_truncate_for_generation(
        list(range(1100)), context_limit=1024, maximum_new_tokens=8
    )

    assert removed == 84
    assert len(kept) == 1016
    assert kept == list(range(84, 1100))


def test_generation_budget_must_fit_context() -> None:
    diagnostic = load_diagnostic()

    with pytest.raises(ValueError, match="smaller than"):
        diagnostic._left_truncate_for_generation(
            [1, 2], context_limit=8, maximum_new_tokens=8
        )


def test_base_anchor_is_resized_to_selected_runtime_tokenizer() -> None:
    diagnostic = load_diagnostic()

    class Embedding:
        def __init__(self, rows: int):
            self.weight = SimpleNamespace(shape=(rows, 8))

    class Model:
        def __init__(self):
            self.input = Embedding(65_536)
            self.output = Embedding(65_536)
            self.resize_calls = []

        def get_input_embeddings(self):
            return self.input

        def get_output_embeddings(self):
            return self.output

        def resize_token_embeddings(self, rows: int):
            self.resize_calls.append(rows)
            self.input = Embedding(rows)
            self.output = Embedding(rows)

    model = Model()

    class Tokenizer:
        def __len__(self):
            return 65_539

        def get_vocab(self):
            return {"base": 4, "<|assistant|>": 65_538}

    tokenizer = Tokenizer()

    report = diagnostic._align_base_anchor_vocabulary(
        model=model,
        tokenizer=tokenizer,
    )

    assert model.resize_calls == [65_539]
    assert report == {
        "tokenizer_length": 65_539,
        "maximum_token_id": 65_538,
        "input_embeddings_before": 65_536,
        "output_embeddings_before": 65_536,
        "input_embeddings_after": 65_539,
        "output_embeddings_after": 65_539,
    }


def test_disagreement_record_reports_margins_and_score_delta() -> None:
    diagnostic = load_diagnostic()
    row = {
        "arm": "multi",
        "language": "eng",
        "prompt_id": "p3",
        "selected_source_index": 7,
        "gold": "business",
        "full_prediction": "business",
        "first_token_prediction": "sports",
        "full_log_likelihoods": {"business": -1.0, "sports": -1.2},
        "first_token_log_probabilities": {"business": -2.0, "sports": -1.9},
    }

    result = diagnostic._full_first_disagreement(row, ["business", "sports"])

    assert result["full_top_two_margin"] == pytest.approx(0.2)
    assert result["first_token_top_two_margin"] == pytest.approx(0.1)
    assert result["maximum_abs_score_delta"] == pytest.approx(1.0)


def test_select_bound_general_arm_checks_every_runtime_hash(tmp_path: Path) -> None:
    diagnostic = load_diagnostic()
    adapter = tmp_path / "runtime_adapters" / "general"
    adapter.mkdir(parents=True)
    payloads = {
        "adapter_model.safetensors": b"weights",
        "adapter_config.json": json.dumps(
            {"target_modules": ["in_proj", "x_proj"]}
        ).encode(),
        "chat_template.jinja": b"normalized",
        "tokenizer.json": b"tokenizer",
        "tokenizer_config.json": b"tokenizer-config",
    }
    for name, payload in payloads.items():
        (adapter / name).write_bytes(payload)
    hashes = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
    }
    related = {
        "adapter_repo": "example/general",
        "adapter_revision": "revision",
        "adapter_weights_sha256": hashes["adapter_model.safetensors"],
        "adapter_config_sha256": hashes["adapter_config.json"],
        "source_chat_template_sha256": "source-template",
        "expected_target_modules": ["in_proj", "x_proj"],
    }
    binding = {
        "schema": "sallm.mamba_news_validation_arm_binding/v1",
        "data_boundary": "validation_only",
        "test_access_allowed": False,
        "base_revision": "base-revision",
        "arm": {
            "id": "general",
            "languages": ["eng", "xho"],
            **{
                key: related[key]
                for key in (
                    "adapter_repo",
                    "adapter_revision",
                    "adapter_weights_sha256",
                    "adapter_config_sha256",
                    "source_chat_template_sha256",
                )
            },
            "tokenizer_json_sha256": hashes["tokenizer.json"],
            "tokenizer_config_sha256": hashes["tokenizer_config.json"],
        },
    }
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(json.dumps(binding))
    protocol = {
        "base": {"revision": "base-revision"},
        "related_retained_general": related,
        "runtime": {
            "prompt_normalization": {"normalized_sha256": hashes["chat_template.jinja"]}
        },
        "arms": [{"id": "other", "languages": ["eng"]}],
    }
    common = SimpleNamespace(
        require=lambda condition, message: condition
        or (_ for _ in ()).throw(ValueError(message)),
        file_sha256=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    )

    selected, loaded = diagnostic._select_bound_arm(
        common=common,
        protocol=protocol,
        binding_path=binding_path,
        asset_root=tmp_path,
    )

    assert selected["arms"] == [{"id": "general", "languages": ["eng", "xho"]}]
    assert loaded == binding
