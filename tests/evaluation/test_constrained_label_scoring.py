from types import SimpleNamespace

import pytest
import torch
from sallm.evaluation.constrained_label_scoring import score_labels


class MambaModel:
    config = SimpleNamespace(model_type="mamba2")

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        weights = torch.Generator().manual_seed(0)
        self.table = torch.randn((40, 40), generator=weights)

    def __call__(self, *, input_ids, attention_mask, use_cache):
        del attention_mask, use_cache
        self.batch_sizes.append(input_ids.shape[0])
        return SimpleNamespace(logits=self.table[input_ids])  # row-independent


def _score(model: MambaModel) -> tuple[str, dict[str, float]]:
    labels = [f"L{i}" for i in range(17)]
    label, _, scores = score_labels(
        model=model,
        context_ids=[1, 2],
        label_ids={label: [3 + i, 20 + i % 5] for i, label in enumerate(labels)},
        labels=labels,
        score_mode="mean",
        pad_token_id=0,
        pad_to_multiple_of=None,
        device=torch.device("cpu"),
    )
    return label, scores


@pytest.mark.parametrize(
    ("microbatch", "expected"), [("1", [1] * 17), ("4", [4] * 4 + [1]), (None, [17])]
)
def test_mamba_label_microbatch_matches_one_label_at_a_time(
    monkeypatch: pytest.MonkeyPatch, microbatch: str | None, expected: list[int]
) -> None:
    monkeypatch.setenv("SALLM_MAMBA_VALIDATION_LABEL_MICROBATCH", "1")
    reference = _score(MambaModel())
    if microbatch is None:
        monkeypatch.delenv("SALLM_MAMBA_VALIDATION_LABEL_MICROBATCH")
    else:
        monkeypatch.setenv("SALLM_MAMBA_VALIDATION_LABEL_MICROBATCH", microbatch)
    model = MambaModel()
    assert _score(model) == reference
    assert model.batch_sizes == expected
