from types import SimpleNamespace

import pytest
import torch
from sallm.training.trainer import CustomSFTTrainer
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother


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


@pytest.mark.parametrize("trainer_class", [CustomSFTTrainer])
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
