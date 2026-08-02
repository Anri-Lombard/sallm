from unittest.mock import MagicMock

import torch
from sallm.evaluation.generation_metrics import GenerationEvaluator


def test_generation_batch_reserves_requested_output_tokens() -> None:
    evaluator = GenerationEvaluator.__new__(GenerationEvaluator)
    evaluator.max_new_tokens = 64
    evaluator.decoding_config = MagicMock()
    evaluator.decoding_config.to_generate_kwargs.return_value = {}

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "prompt"
    tokenized = MagicMock()
    tokenized.to.return_value = {
        "input_ids": torch.arange(2048).unsqueeze(0),
        "attention_mask": torch.ones((1, 2048), dtype=torch.long),
    }
    tokenizer.return_value = tokenized
    evaluator.tokenizer = tokenizer

    prepared = evaluator._prepare_generation_batch(
        [
            {
                "messages": [
                    {"role": "user", "content": "article"},
                    {"role": "assistant", "content": "headline"},
                ]
            }
        ],
        None,
        torch.device("cpu"),
        2048,
        0,
        2,
    )

    assert prepared is not None
    assert prepared[3].shape[1] == 1983
    assert prepared[5]["max_new_tokens"] == 64
