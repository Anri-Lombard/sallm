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
            "training": {
                "output_dir": str(tmp_path / "checkpoints"),
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 4,
            },
            "hub": {"enabled": True},
        }
    )


def test_sweep_updates_are_isolated_and_derived(tmp_path: Path) -> None:
    cfg = _apply_updates(
        _config(tmp_path),
        {"hpo.effective_batch_size": 64},
        "run123",
    )

    assert cfg.wandb.id == "sweep-run123"
    assert cfg.training.output_dir.endswith("checkpoints/run123")
    assert cfg.training.gradient_accumulation_steps == 8
    assert not cfg.hub.enabled


def test_effective_batch_must_match_microbatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        _apply_updates(
            _config(tmp_path),
            {"hpo.effective_batch_size": 60},
            "run123",
        )
