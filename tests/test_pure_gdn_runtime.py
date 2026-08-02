import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from datasets import IterableDataset
from trl import pack_dataset
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


def _load_verifier_module():
    path = Path(__file__).parents[1] / "scripts" / "verify_pure_gdn_runtime.py"
    spec = importlib.util.spec_from_file_location("verify_pure_gdn_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chunk_kernel_probe_passes_fused_gate_inputs(monkeypatch) -> None:
    verifier = _load_verifier_module()
    tensors = []
    captured = {}

    class Tensor:
        def __init__(self, shape, device, dtype, requires_grad) -> None:
            self.shape = shape
            self.device = device
            self.dtype = dtype
            self.requires_grad = requires_grad
            self.grad = None
            tensors.append(self)

        def float(self):
            return self

        def mean(self):
            return self

        def backward(self) -> None:
            for tensor in tensors:
                tensor.grad = object()

    class FakeTorch:
        bfloat16 = "bfloat16"
        float32 = "float32"

        def __init__(self) -> None:
            self.cuda = SimpleNamespace(is_available=lambda: True)

        def device(self, name: str) -> str:
            return name

        def randn(self, shape, *, device, dtype, requires_grad):
            return Tensor(shape, device, dtype, requires_grad)

        def randn_like(self, tensor, *, requires_grad):
            return Tensor(tensor.shape, tensor.device, tensor.dtype, requires_grad)

        def zeros(self, shape, *, device, dtype, requires_grad):
            return Tensor(shape, device, dtype, requires_grad)

        def zeros_like(self, tensor, *, requires_grad):
            return Tensor(tensor.shape, tensor.device, tensor.dtype, requires_grad)

    def kernel(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return Tensor((), "cuda", "bfloat16", False)

    monkeypatch.setattr(verifier, "torch", FakeTorch())
    monkeypatch.setattr(
        verifier,
        "import_module",
        lambda name: SimpleNamespace(chunk_gated_delta_rule=kernel)
        if name == "fla.ops.gated_delta_rule"
        else (_ for _ in ()).throw(AssertionError(name)),
    )

    assert verifier._exercise_chunk_gated_delta_kernel().endswith("A_log,dt_bias[HV]")
    q, k, v, g, beta = captured["args"]
    assert q.shape == k.shape == (1, 64, 2, 32)
    assert v.shape == (1, 64, 4, 32)
    assert g.shape == beta.shape == (1, 64, 4)
    assert captured["kwargs"]["A_log"].shape == (4,)
    assert captured["kwargs"]["dt_bias"].shape == (4,)
    assert captured["kwargs"]["A_log"].dtype == "float32"
    assert captured["kwargs"]["dt_bias"].dtype == "float32"
    assert captured["kwargs"]["use_gate_in_kernel"] is True


def test_pure_canary_streams_training_data_only() -> None:
    data = yaml.safe_load(
        (
            Path(__file__).parents[1]
            / "src/conf/base/gated_deltanet_125m_pure_canary.yaml"
        ).read_text()
    )["data"]

    assert data["streaming"] is True
    assert data["test_split"] is None
    full_data = yaml.safe_load(
        (
            Path(__file__).parents[1] / "src/conf/base/gated_deltanet_125m_pure.yaml"
        ).read_text()
    )["data"]
    assert full_data.get("streaming", False) is False


@pytest.mark.parametrize(
    "config_name",
    ["gated_deltanet_125m_pure.yaml", "gated_deltanet_125m_pure_canary.yaml"],
)
def test_pure_configs_use_non_flattening_packing(config_name: str) -> None:
    training = yaml.safe_load(
        (Path(__file__).parents[1] / "src/conf/base" / config_name).read_text()
    )["training"]

    assert training["packing"] is True
    assert training["packing_strategy"] == "wrapped"
    assert training["padding_free"] is False


def test_wrapped_streaming_packing_keeps_two_2048_token_contexts() -> None:
    config = yaml.safe_load(
        (
            Path(__file__).parents[1]
            / "src/conf/base/gated_deltanet_125m_pure_canary.yaml"
        ).read_text()
    )
    training = config["training"]
    assert config["data"]["streaming"] is True

    dataset = IterableDataset.from_generator(
        lambda: (
            {"input_ids": list(range(index * 1024 + 1, (index + 1) * 1024 + 1))}
            for index in range(4)
        )
    )
    packed_rows = list(
        pack_dataset(
            dataset,
            int(training["max_seq_length"]),
            str(training["packing_strategy"]),
        ).take(2)
    )
    batch = DataCollatorForLanguageModeling(
        pad_token_id=0,
        padding_free=bool(training["padding_free"]),
    )(packed_rows)

    assert [len(row["input_ids"]) for row in packed_rows] == [2048, 2048]
    assert tuple(batch["input_ids"].shape) == (2, 2048)
    assert tuple(batch["attention_mask"].shape) == (2, 2048)
    assert batch["input_ids"][:, 0].tolist() == [1, 2049]
    assert "position_ids" not in batch


def test_canary_falls_back_when_the_conda_environment_is_absent() -> None:
    launcher = (
        Path(__file__).parents[1] / "scripts" / "run_pure_gdn_a10080_canary.sh"
    ).read_text()

    assert "if command -v conda >/dev/null 2>&1; then" in launcher
    assert "conda env list | awk '{print $1}' | grep -qx sallm-uv" in launcher
    assert "using the repository .venv" in launcher
    assert launcher.index("using the repository .venv") < launcher.index(
        "source .venv/bin/activate"
    )


def test_canary_uses_grouped_training_cli_overrides() -> None:
    launcher = (
        Path(__file__).parents[1] / "scripts" / "run_pure_gdn_a10080_canary.sh"
    ).read_text()

    assert '"base.training.output_dir=$OUTPUT_DIR"' in launcher
    assert '"base.training.logging_dir=$LOG_DIR"' in launcher
    assert (
        'resume_args+=("base.training.resume_from_checkpoint=$latest_checkpoint")'
        in launcher
    )
    assert '"training.output_dir=' not in launcher
    assert '"training.logging_dir=' not in launcher
    assert '"training.resume_from_checkpoint=' not in launcher
