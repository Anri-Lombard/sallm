import torch
from sallm.evaluation.constrained_label_scoring import score_labels
from transformers import Mamba2Config, Mamba2ForCausalLM

LABELS = [f"L{i}" for i in range(17)]
LABEL_IDS = {label: [10 + i, 30 + i % 3] for i, label in enumerate(LABELS)}


def _score(model, labels: list[str]) -> tuple[str, dict[str, float]]:
    label, _, scores = score_labels(
        model=model,
        context_ids=[1, 5, 9, 2],
        label_ids=LABEL_IDS,
        labels=labels,
        score_mode="mean",
        pad_token_id=0,
        pad_to_multiple_of=None,
        device=torch.device("cpu"),
    )
    return label, scores


def test_batched_mamba_label_scores_match_scoring_one_label_at_a_time() -> None:
    torch.manual_seed(0)
    model = Mamba2ForCausalLM(
        Mamba2Config(
            vocab_size=64,
            hidden_size=64,
            num_heads=4,
            head_dim=32,
            state_size=16,
            n_groups=1,
            num_hidden_layers=2,
            chunk_size=8,
        )
    ).eval()
    calls: list[int] = []
    forward = model.forward

    def counting_forward(*args, **kwargs):
        calls.append(kwargs["input_ids"].shape[0])
        return forward(*args, **kwargs)

    model.forward = counting_forward
    batched_label, batched = _score(model, LABELS)
    one_at_a_time = {label: _score(model, [label])[1][label] for label in LABELS}

    assert calls[0] == 17
    assert batched_label == max(one_at_a_time, key=one_at_a_time.__getitem__)
    for label in LABELS:
        assert abs(batched[label] - one_at_a_time[label]) < 1e-5
