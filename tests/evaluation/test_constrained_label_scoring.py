from types import SimpleNamespace

import torch
from sallm.evaluation.constrained_label_scoring import score_labels


def test_mamba_multitoken_label_scoring_forwards_one_label_at_a_time() -> None:
    class MambaModel:
        config = SimpleNamespace(model_type="mamba2")

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def __call__(self, *, input_ids, attention_mask, use_cache):
            del attention_mask, use_cache
            self.batch_sizes.append(input_ids.shape[0])
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 7))
            for row, targets in enumerate(input_ids[:, 2:].tolist()):
                for offset, target in enumerate(targets):
                    logits[row, 1 + offset, target] = float(target)
            return SimpleNamespace(logits=logits)

    model = MambaModel()
    label, _, _ = score_labels(
        model=model,
        context_ids=[1, 2],
        label_ids={"a": [3, 4], "b": [5, 6]},
        labels=["a", "b"],
        score_mode="mean",
        pad_token_id=0,
        pad_to_multiple_of=None,
        device=torch.device("cpu"),
    )

    assert label == "b"
    assert model.batch_sizes == [1, 1]
