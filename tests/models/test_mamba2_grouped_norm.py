from __future__ import annotations

import torch
import torch.nn.functional as F
from sallm.models.registry import (
    patch_mamba2_grouped_rmsnorm,
    validate_xlstm_pretrain_kernels,
)
from transformers import Mamba2Config, Mamba2ForCausalLM


def _reference_grouped_rmsnorm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    n_groups: int,
    eps: float,
) -> torch.Tensor:
    y = x.float() * F.silu(gate.float())
    orig_shape = y.shape
    grouped = y.view(*orig_shape[:-1], n_groups, orig_shape[-1] // n_groups)
    var = grouped.pow(2).mean(-1, keepdim=True)
    normed = (grouped * torch.rsqrt(var + eps)).reshape(orig_shape)
    return weight * normed.to(x.dtype)


def test_mamba2_grouped_rmsnorm_matches_per_group_reference() -> None:
    torch.manual_seed(0)
    cfg = Mamba2Config(
        vocab_size=64,
        hidden_size=32,
        state_size=16,
        num_hidden_layers=2,
        expand=2,
        num_heads=4,
        head_dim=16,
        n_groups=4,
        chunk_size=16,
    )
    model = Mamba2ForCausalLM(cfg)
    patched_count = patch_mamba2_grouped_rmsnorm(model)
    assert patched_count == 2

    mixer = model.backbone.layers[0].mixer
    x = torch.randn(2, 8, mixer.intermediate_size)
    # Give different groups distinct scales so whole-channel norm diverges from grouped.
    x[:, :, : mixer.intermediate_size // 4] *= 10.0
    gate = torch.randn_like(x)

    actual = mixer.norm(x, gate)
    expected = _reference_grouped_rmsnorm(
        x,
        gate,
        mixer.norm.weight,
        n_groups=cfg.n_groups,
        eps=mixer.norm.variance_epsilon,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    # Whole-channel norm would produce a noticeably different result on skewed groups.
    whole_var = (x.float() * F.silu(gate.float())).pow(2).mean(-1, keepdim=True)
    whole = mixer.norm.weight * (
        (x.float() * F.silu(gate.float()))
        * torch.rsqrt(whole_var + mixer.norm.variance_epsilon)
    ).to(x.dtype)
    assert (actual - whole).abs().max().item() > 0.5


def test_xlstm_kernel_guard_rejects_native_autograd_when_disallowed() -> None:
    class FakeConfig:
        chunkwise_kernel = "chunkwise--native_autograd"
        sequence_kernel = "native_sequence__native"
        step_kernel = "native"
        mode = "train"
        chunk_size = 64

    try:
        validate_xlstm_pretrain_kernels(FakeConfig(), allow_native=False)
    except ValueError as exc:
        assert "chunkwise--native_autograd" in str(exc)
    else:
        raise AssertionError("Expected ValueError for pure-PyTorch native kernel")

    info = validate_xlstm_pretrain_kernels(FakeConfig(), allow_native=True)
    assert info["chunkwise_kernel"] == "chunkwise--native_autograd"
