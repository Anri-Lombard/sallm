from types import SimpleNamespace

import pytest
from sallm.models import registry


def test_non_gdn_lookup_does_not_import_fla(monkeypatch) -> None:
    calls: list[str] = []
    llama_config = type("LlamaConfig", (), {})

    def fake_import(name: str):
        calls.append(name)
        if name == "transformers":
            return SimpleNamespace(LlamaConfig=llama_config)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(registry, "import_module", fake_import)
    values = registry.LazyRegistry(
        {
            "llama": "LlamaConfig",
            "gated_deltanet": (
                "fla.models.gated_deltanet",
                "GatedDeltaNetConfig",
            ),
        }
    )

    assert values["llama"] is llama_config
    assert calls == ["transformers"]


def test_gated_deltanet_resolves_the_fla_classes(monkeypatch) -> None:
    config_class = type("GatedDeltaNetConfig", (), {})
    model_class = type("GatedDeltaNetForCausalLM", (), {})
    module = SimpleNamespace(
        GatedDeltaNetConfig=config_class,
        GatedDeltaNetForCausalLM=model_class,
    )
    calls: list[str] = []

    def fake_import(name: str):
        calls.append(name)
        return module

    monkeypatch.setattr(registry, "import_module", fake_import)
    configs = registry.LazyRegistry(
        {"gated_deltanet": ("fla.models.gated_deltanet", "GatedDeltaNetConfig")}
    )
    models = registry.LazyRegistry(
        {
            "gated_deltanet": (
                "fla.models.gated_deltanet",
                "GatedDeltaNetForCausalLM",
            )
        }
    )

    assert configs["gated_deltanet"] is config_class
    assert models["gated_deltanet"] is model_class
    assert calls == ["fla.models.gated_deltanet", "fla.models.gated_deltanet"]


def test_missing_top_level_fla_has_an_actionable_error(monkeypatch) -> None:
    def missing_fla(_: str):
        raise ModuleNotFoundError("No module named 'fla'", name="fla")

    monkeypatch.setattr(registry, "import_module", missing_fla)
    values = registry.LazyRegistry(
        {"gated_deltanet": ("fla.models.gated_deltanet", "GatedDeltaNetConfig")}
    )

    with pytest.raises(ModuleNotFoundError, match="uv sync --extra pure-gdn"):
        values["gated_deltanet"]


def test_transitive_fla_import_errors_are_not_swallowed(monkeypatch) -> None:
    error = ModuleNotFoundError("No module named 'triton'", name="triton")
    monkeypatch.setattr(
        registry,
        "import_module",
        lambda _: (_ for _ in ()).throw(error),
    )
    values = registry.LazyRegistry(
        {"gated_deltanet": ("fla.models.gated_deltanet", "GatedDeltaNetConfig")}
    )

    with pytest.raises(ModuleNotFoundError) as raised:
        values["gated_deltanet"]

    assert raised.value is error
