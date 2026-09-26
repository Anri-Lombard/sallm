from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from datasets import Dataset
from sallm.evaluation.classification_metrics import (
    ChoiceScoreMode,
    ClassificationEvaluator,
    _gather_target_log_probs,
)
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
NEWS_LABELS = [
    "business",
    "entertainment",
    "health",
    "politics",
    "religion",
    "sports",
    "technology",
]


def test_classification_uses_length_normalized_choice_scores_by_default() -> None:
    evaluator = ClassificationEvaluator(tokenizer=object())

    assert evaluator.choice_score_mode == ChoiceScoreMode.MEAN


def test_classification_cap_is_balanced_by_prompt_and_label() -> None:
    rows = [
        {
            "template_id": prompt,
            "messages": [
                {"role": "user", "content": str(index)},
                {"role": "assistant", "content": label},
            ],
        }
        for prompt in ("p1", "p2")
        for label in ("a", "b")
        for index in range(10)
    ]
    evaluator = ClassificationEvaluator(tokenizer=object(), max_samples_per_lang=9)

    capped = evaluator._cap_dataset(Dataset.from_list(rows), "all")
    counts: dict[tuple[str, str], int] = {}
    for sample in capped:
        key = (sample["template_id"], sample["messages"][-1]["content"])
        counts[key] = counts.get(key, 0) + 1

    assert len(capped) == 8
    assert set(counts.values()) == {2}


def test_gather_target_log_probs_matches_full_log_softmax() -> None:
    logits = torch.tensor(
        [
            [[1.0, -2.0, 0.5, 3.0], [4.0, 1.0, -1.0, 0.0]],
            [[-3.0, 2.0, 0.0, 1.0], [0.5, 0.25, -0.5, 2.0]],
        ],
        dtype=torch.float32,
    )
    target_ids = torch.tensor([[3, 0], [1, 2]])

    expected = torch.gather(
        torch.log_softmax(logits, dim=-1),
        2,
        target_ids.unsqueeze(-1),
    ).squeeze(-1)

    actual = _gather_target_log_probs(logits=logits, target_ids=target_ids)

    torch.testing.assert_close(actual, expected)


def test_encode_choice_pair_scores_only_uniform_label_tokens() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "tokenizer" / "sallm_bpe_tokenizer",
        local_files_only=True,
    )
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|system|>",
                "<|user|>",
                "<|assistant|>",
            ]
        }
    )
    evaluator = ClassificationEvaluator(tokenizer)

    continuation_ids = [
        evaluator._encode_choice_pair("<|assistant|>", label)[1]
        for label in NEWS_LABELS
    ]
    expected_ids = [
        tokenizer.encode(f" {label}", add_special_tokens=False) for label in NEWS_LABELS
    ]

    assert [len(ids) for ids in continuation_ids] == [1] * len(NEWS_LABELS)
    assert continuation_ids == expected_ids
    assert all(tokenizer.eos_token_id not in ids for ids in continuation_ids)


def test_encode_choice_pair_preserves_existing_trailing_whitespace() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "tokenizer" / "sallm_bpe_tokenizer",
        local_files_only=True,
    )
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|system|>",
                "<|user|>",
                "<|assistant|>",
            ]
        }
    )
    evaluator = ClassificationEvaluator(tokenizer)
    context = "<|assistant|>\n        "

    context_ids, continuation_ids = evaluator._encode_choice_pair(context, "business")

    assert context_ids + continuation_ids == tokenizer.encode(
        f"{context}business", add_special_tokens=False
    )


def test_xlstm_choice_inputs_pad_to_chunk_size() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "tokenizer" / "sallm_bpe_tokenizer",
        local_files_only=True,
    )
    evaluator = ClassificationEvaluator(tokenizer)

    input_ids, attention_mask, _ = evaluator._build_choice_inputs(
        prompt_text="Classify this news story:",
        label_choices=NEWS_LABELS,
        model_ctx_limit=1024,
        pad_token_id=int(tokenizer.eos_token_id),
        device=torch.device("cpu"),
        pad_to_multiple_of=64,
    )

    assert input_ids.shape == attention_mask.shape
    assert input_ids.shape[1] % 64 == 0
    assert bool((attention_mask.sum(dim=1) < input_ids.shape[1]).all())


def test_mamba_choice_scoring_forwards_one_label_at_a_time(monkeypatch) -> None:
    evaluator = ClassificationEvaluator.__new__(ClassificationEvaluator)
    evaluator.choice_score_mode = ChoiceScoreMode.MEAN
    input_ids = torch.tensor(
        [[1, 2, 3, 0], [1, 2, 4, 0], [1, 2, 5, 0]], dtype=torch.long
    )
    attention_mask = torch.tensor(
        [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.long
    )
    monkeypatch.setattr(evaluator, "_build_prompt_text", lambda **_kwargs: "prompt")
    monkeypatch.setattr(evaluator, "_resolve_pad_id", lambda *_args: 0)
    monkeypatch.setattr(evaluator, "_get_model_ctx_limit", lambda _model: 4)
    monkeypatch.setattr(evaluator, "_get_model_chunk_size", lambda _model: None)
    monkeypatch.setattr(
        evaluator,
        "_build_choice_inputs",
        lambda **_kwargs: (input_ids, attention_mask, [2, 2, 2]),
    )

    class MambaModel:
        config = SimpleNamespace(model_type="mamba2")

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def __call__(self, *, input_ids, attention_mask, use_cache):
            del attention_mask, use_cache
            self.batch_sizes.append(input_ids.shape[0])
            logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 6))
            for row, target in enumerate(input_ids[:, 2].tolist()):
                logits[row, 1, target] = float(target)
            return SimpleNamespace(logits=logits)

    model = MambaModel()
    prediction = evaluator._score_label_choices(
        model=model,
        prompt_messages=[],
        label_choices=["a", "b", "c"],
        device=torch.device("cpu"),
        pad_id=0,
        eos_id=0,
        fallback_template=None,
        system_message=None,
    )

    assert prediction == "c"
    assert model.batch_sizes == [1, 1, 1]


def test_evaluate_exposes_mean_per_language_macro_f1(monkeypatch) -> None:
    evaluator = ClassificationEvaluator.__new__(ClassificationEvaluator)
    evaluator.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    evaluator.max_samples_per_lang = None
    monkeypatch.setattr(evaluator, "_get_fallback_template", lambda: None)
    monkeypatch.setattr(evaluator, "_cap_dataset", lambda dataset, _lang: dataset)

    def fake_subset(_model, dataset, *_args):
        lang = dataset[0]["lang"]
        return {
            "accuracy": 0.5,
            "f1": 0.8 if lang == "a" else 0.6,
            "macro_f1": 0.3 if lang == "a" else 0.1,
        }

    monkeypatch.setattr(evaluator, "_evaluate_subset", fake_subset)
    dataset = Dataset.from_list(
        [
            {"lang": "a", "messages": []},
            {"lang": "b", "messages": []},
        ]
    )
    model = MagicMock()
    model.device = torch.device("cpu")

    metrics = evaluator.evaluate(model, dataset)

    assert metrics["classification/all_f1"] == 0.7
    assert metrics["classification/all_macro_f1"] == 0.2
