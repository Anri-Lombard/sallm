from pathlib import Path

import pytest
from omegaconf import OmegaConf
from sallm.hpo.trial import _apply_updates


def _config(tmp_path: Path):
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    return OmegaConf.create(
        {
            "wandb": {"project": "test", "id": None},
            "tokenizer": {"path": str(tokenizer)},
            "peft": {"kwargs": {"r": 16, "lora_alpha": 32}},
            "training": {
                "output_dir": str(tmp_path / "checkpoints"),
                "logging_dir": str(tmp_path / "logs"),
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 4,
            },
            "hub": {
                "enabled": True,
                "push_adapter": True,
                "push_merged": True,
            },
        }
    )


def test_sweep_updates_are_isolated_and_derived(tmp_path: Path) -> None:
    cfg = _apply_updates(
        _config(tmp_path),
        {"hpo.lora_rank": 32, "hpo.effective_batch_size": 64},
        "run123",
    )

    assert cfg.wandb.id == "sweep-run123"
    assert cfg.training.output_dir.endswith("checkpoints/run123")
    assert cfg.training.logging_dir.endswith("logs/run123")
    assert cfg.training.gradient_accumulation_steps == 8
    assert cfg.peft.kwargs.r == 32
    assert cfg.peft.kwargs.lora_alpha == 64
    assert not cfg.hub.enabled
    assert not cfg.hub.push_adapter
    assert not cfg.hub.push_merged


def test_effective_batch_must_match_microbatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        _apply_updates(
            _config(tmp_path),
            {"hpo.effective_batch_size": 60},
            "run123",
        )
