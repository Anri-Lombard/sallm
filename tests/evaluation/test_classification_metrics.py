from pathlib import Path

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
