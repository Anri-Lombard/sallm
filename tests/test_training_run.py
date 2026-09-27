from types import SimpleNamespace

import pytest
import torch
from sallm.training import run as training_run
from sallm.training.trainer import CustomSFTTrainer, CustomTrainer
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother


class _FakeTrainer:
    def __init__(self, length: int) -> None:
        self._length = length

    def get_train_dataloader(self):
        return [{"input_ids": torch.zeros((1, self._length), dtype=torch.long)}]


def test_fla_gated_deltanet_label_smoothing_uses_causal_shift() -> None:
    class GatedDeltaNetForCausalLM(torch.nn.Module):
        def forward(self, **_kwargs):
            logits = torch.full((1, 4, 5), -4.0)
            logits[0, 0, 1] = 4.0
            logits[0, 1, 2] = 4.0
            logits[0, 2, 3] = 4.0
            logits[0, 3, 4] = 4.0
            return {"logits": logits}

    model = GatedDeltaNetForCausalLM()
    labels = torch.tensor([[0, 1, 2, 3]])
    outputs = model()
    smoother = LabelSmoother(epsilon=0.05)
    expected = smoother(outputs, labels, shift_labels=True)
    unshifted = smoother(outputs, labels, shift_labels=False)
    trainer = object.__new__(Trainer)
    trainer.label_smoother = smoother
    trainer.compute_loss_func = None
    trainer.model_accepts_loss_kwargs = False
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda value: value)
    trainer.args = SimpleNamespace(
        past_index=-1,
        average_tokens_across_devices=False,
        n_gpu=1,
    )

    actual = Trainer.compute_loss(
        trainer,
        model,
        {"input_ids": labels, "labels": labels.clone()},
    )

    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, unshifted)


def test_run_saves_between_barriers(monkeypatch, tmp_path) -> None:
    events: list[str] = []

    class Trainer(_FakeTrainer):
        args = type("Args", (), {"output_dir": tmp_path, "should_save": True})()
        accelerator = type(
            "Accelerator",
            (),
            {"wait_for_everyone": lambda _self: events.append("wait")},
        )()

        def train(self, **_kwargs) -> None:
            events.append("train")

        def save_model(self, _path) -> None:
            events.append("save")

    trainer = Trainer(2048)
    config = type(
        "Config",
        (),
        {
            "wandb": type("Wandb", (), {"id": None, "project": None})(),
            "training": {},
            "hub": type("Hub", (), {"enabled": False})(),
        },
    )()
    monkeypatch.setattr(training_run, "build_tokenizer", lambda _config: object())
    monkeypatch.setattr(
        training_run,
        "build_model",
        lambda *_args: type("Model", (), {})(),
    )
    monkeypatch.setattr(
        training_run,
        "build_datasets",
        lambda *_args, **_kwargs: ([], [], None),
    )
    monkeypatch.setattr(training_run, "build_trainer", lambda *_args: trainer)

    training_run.run(config)

    assert events == ["train", "wait", "save", "wait"]


@pytest.mark.parametrize("trainer_class", [CustomTrainer, CustomSFTTrainer])
def test_non_saving_trainer_rank_does_not_write_model_or_tokenizer(
    trainer_class,
    tmp_path,
) -> None:
    writes: list[str] = []

    class Saveable:
        def save_pretrained(self, *_args, **_kwargs) -> None:
            writes.append("save")

    output_dir = tmp_path / "non-saving-rank"
    trainer = type(
        "NonSavingTrainer",
        (),
        {
            "args": type(
                "Args",
                (),
                {"output_dir": output_dir, "should_save": False},
            )(),
            "model": Saveable(),
            "tokenizer": Saveable(),
        },
    )()

    trainer_class.save_model(trainer)

    assert writes == []
    assert not output_dir.exists()
