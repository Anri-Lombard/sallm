from types import SimpleNamespace

import torch
from sallm.evaluation import harness


def test_mamba_evaluation_disables_incompatible_generation_cache() -> None:
    config = SimpleNamespace(model_type="mamba2", use_cache=True)

    harness.prepare_model_config_for_evaluation(config)

    assert config.use_cache is False


def test_model_loading_registers_fla_before_auto_model(monkeypatch) -> None:
    events: list[str] = []

    class Tokenizer:
        backend_tokenizer = None
        pad_token = "<pad>"
        eos_token = "</s>"
        eos_token_id = 2
        pad_token_id = 0
        padding_side = "right"

        def __len__(self) -> int:
            return 8

    class Model:
        generation_config = None
        config = SimpleNamespace()

        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros((8, 2)))

        def resize_token_embeddings(self, _: int) -> None:
            raise AssertionError("unexpected resize")

        def to(self, _: object):
            return self

    monkeypatch.setattr(
        harness,
        "register_fla_gated_deltanet",
        lambda: events.append("register") or True,
    )
    monkeypatch.setattr(
        harness.torch,
        "manual_seed",
        lambda seed: events.append(f"seed:{seed}"),
    )
    monkeypatch.setattr(
        harness.random,
        "seed",
        lambda seed: events.append(f"sample-seed:{seed}"),
    )
    monkeypatch.setattr(
        harness,
        "_load_tokenizer_and_pretrained",
        lambda *_args, **_kwargs: (Tokenizer(), "owner/model"),
    )
    monkeypatch.setattr(harness, "_resolve_dtype", lambda _: torch.float32)
    monkeypatch.setattr(
        harness.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: events.append("model") or Model(),
    )
    config = SimpleNamespace(
        checkpoint="owner/model",
        dtype="float32",
        device="cpu",
    )

    harness.load_model_and_tokenizer(config)

    assert events == ["sample-seed:42", "seed:42", "register", "model"]
