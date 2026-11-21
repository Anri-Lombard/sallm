from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sallm.pretrain import (
    LMLoss,
    PackedTokenBlocks,
    build_pretrain_model,
    canonical_arch,
    flops_per_token,
    lr_at,
    param_groups,
)

PRETRAIN_CONF_DIR = Path(__file__).resolve().parents[2] / "src" / "conf" / "pretrain"


def test_pretrain_configs_cover_all_five_architectures() -> None:
    expected = {"mzansilm", "mamba2", "xlstm", "gdn", "hybrid"}
    found = {p.stem.removesuffix("_wsd") for p in PRETRAIN_CONF_DIR.glob("*_wsd.json")}
    assert expected <= found

    xlstm_cfg = json.loads((PRETRAIN_CONF_DIR / "xlstm_wsd.json").read_text())
    assert xlstm_cfg["xlstm_chunkwise_kernel"] == "chunkwise--triton_xl_chunk"


def test_canonical_arch_aliases() -> None:
    assert canonical_arch("mzansilm") == "llama"
    assert canonical_arch("gdn") == "gated_deltanet"
    assert canonical_arch("hybrid") == "hybrid"
    with pytest.raises(ValueError):
        canonical_arch("unknown_arch")


def test_wsd_and_cosine_schedules() -> None:
    wsd = {"type": "wsd", "min_lr_ratio": 0.1, "decay_frac": 0.2}
    assert lr_at(0, wsd, 1e-3, 100, 10) == pytest.approx(1e-4)
    assert lr_at(9, wsd, 1e-3, 100, 10) == pytest.approx(1e-3)
    assert lr_at(50, wsd, 1e-3, 100, 10) == pytest.approx(1e-3)
    assert lr_at(99, wsd, 1e-3, 100, 10) < 3e-4
    assert lr_at(100, wsd, 1e-3, 100, 10) == pytest.approx(1e-4)

    cos = {"type": "cosine", "min_lr_ratio": 0.1}
    assert lr_at(10, cos, 1e-3, 100, 10) == pytest.approx(1e-3)
    assert lr_at(100, cos, 1e-3, 100, 10) == pytest.approx(1e-4)


def test_packed_token_blocks_deterministic_epochs(tmp_path: Path) -> None:
    tokens = np.arange(32, dtype=np.uint16)
    tokens.tofile(tmp_path / "train.bin")
    (tmp_path / "meta.json").write_text(json.dumps({"n_tokens": 32}))

    blocks = PackedTokenBlocks(tmp_path, seq_len=8, seed=42)
    assert blocks.n_blocks == 4
    first_epoch = blocks.get(0, 4)
    assert first_epoch.tolist() == [list(range(i * 8, (i + 1) * 8)) for i in range(4)]

    # Second epoch is deterministically permuted across blocks.
    blocks2 = PackedTokenBlocks(tmp_path, seq_len=8, seed=42)
    assert torch.equal(blocks.get(4, 4), blocks2.get(4, 4))


def test_build_pretrain_model_and_loss_on_tiny_llama() -> None:
    cfg = {
        "arch": "llama",
        "seq_len": 16,
        "model_config": {
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "tie_word_embeddings": True,
        },
    }
    model = build_pretrain_model(cfg)
    groups = param_groups(model, weight_decay=0.1)
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in groups[0]["params"])
    assert all(p.ndim < 2 for p in groups[1]["params"])

    loss_fn = LMLoss(model, fused=False)
    x = torch.randint(0, 64, (2, 16))
    loss = loss_fn(x)
    assert torch.isfinite(loss)

    flops = flops_per_token(model, cfg)
    assert flops["train_total"] > flops["train_dense"] > 0


@pytest.mark.parametrize("name", ["mzansilm_wsd", "mamba2_wsd", "xlstm_wsd"])
def test_checked_in_pretrain_configs_construct_with_exact_param_counts(
    name: str,
) -> None:
    cfg = json.loads((PRETRAIN_CONF_DIR / f"{name}.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = build_pretrain_model(cfg, allow_native_xlstm=False)
    assert sum(p.numel() for p in model.parameters()) == cfg["expected_params"]
