from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from sacrebleu.metrics import CHRF
from sallm.evaluation.generation_metrics import GenerationEvaluator
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


def test_generation_chrf_is_direct_sacrebleu_corpus_score() -> None:
    evaluator = GenerationEvaluator.__new__(GenerationEvaluator)
    evaluator._chrf = CHRF(
        char_order=6,
        word_order=0,
        beta=2,
        lowercase=False,
        whitespace=False,
        eps_smoothing=False,
    )
    evaluator._bleu = MagicMock()
    evaluator._bleu.compute.return_value = {}
    evaluator._rouge_scorer = MagicMock()
    evaluator._rouge_scorer.score.return_value = {
        key: SimpleNamespace(fmeasure=0.0) for key in ("rouge1", "rouge2", "rougeL")
    }
    evaluator.task_type = None

    identity = evaluator._compute_metrics(["isiXhosa"], [["isiXhosa"]])["chrf"]
    non_identity = evaluator._compute_metrics(["isiZulu"], [["isiXhosa"]])["chrf"]

    assert identity == 100.0
    assert 0.0 <= non_identity < identity


def test_seeded_generation_sample_indices_are_stable() -> None:
    evaluator = GenerationEvaluator.__new__(GenerationEvaluator)
    evaluator.sample_seed = 20260913

    assert evaluator._sample_indices(460, 16, "xho") == [
        30,
        42,
        49,
        70,
        140,
        179,
        212,
        216,
        229,
        350,
        367,
        374,
        393,
        395,
        406,
        411,
    ]


def test_generation_batch_reserves_requested_output_tokens() -> None:
    evaluator = GenerationEvaluator.__new__(GenerationEvaluator)
    evaluator.max_new_tokens = 64
    evaluator.max_input_tokens = 1024
    evaluator.prompt_format = "chat"
    evaluator.decoding_config = MagicMock()
    evaluator.decoding_config.to_generate_kwargs.return_value = {}

    tokenizer = MagicMock()
    tokenizer.bos_token_id = 0
    tokenizer.apply_chat_template.side_effect = lambda *_args, **kwargs: (
        list(range(2048)) if kwargs.get("tokenize") else "prompt"
    )
    tokenizer.encode.return_value = list(range(2048))
    tokenized = MagicMock()
    tokenized.to.return_value = {
        "input_ids": torch.arange(2048).unsqueeze(0),
        "attention_mask": torch.ones((1, 2048), dtype=torch.long),
    }
    tokenizer.return_value = tokenized
    evaluator.tokenizer = tokenizer
    model = MagicMock()
    model.config.use_cache = False

    prepared = evaluator._prepare_generation_batch(
        model,
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
    assert prepared[3].shape[1] == 1024
    assert prepared[5]["max_new_tokens"] == 64
    assert prepared[5]["use_cache"] is False
    tokenizer.assert_called_once_with(
        ["prompt"],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )


def test_generation_batch_rejects_terminal_eos_after_assistant_marker() -> None:
    evaluator = GenerationEvaluator.__new__(GenerationEvaluator)
    evaluator.max_new_tokens = 8
    evaluator.prompt_format = "chat"
    evaluator.decoding_config = MagicMock()
    evaluator.decoding_config.to_generate_kwargs.return_value = {}

    tokenizer = MagicMock()
    tokenizer.bos_token_id = 7
    tokenizer.eos_token_id = 2
    tokenizer.apply_chat_template.side_effect = lambda *_args, **kwargs: (
        [7, 2] if kwargs.get("tokenize") else "prompt"
    )
    tokenizer.encode.return_value = [7, 2]
    evaluator.tokenizer = tokenizer
    model = MagicMock()
    model.config.use_cache = False

    try:
        evaluator._prepare_generation_batch(
            model,
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
            128,
            0,
            2,
        )
    except ValueError as exc:
        assert "ends in EOS" in str(exc)
    else:
        raise AssertionError("terminal EOS must fail the generation contract")


def test_raw_base_prompt_has_one_bos_and_no_chat_markers() -> None:
    evaluator = GenerationEvaluator.__new__(GenerationEvaluator)
    evaluator.tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "tokenizer" / "sallm_bpe_tokenizer",
        local_files_only=True,
    )
    evaluator.prompt_format = "raw"

    prompt_text, prompt_ids = evaluator._render_generation_prompt(
        [{"role": "user", "content": "Translate this text."}],
        "Return fluent isiXhosa.",
        None,
    )

    assert prompt_ids[0] == evaluator.tokenizer.bos_token_id
    assert prompt_ids.count(evaluator.tokenizer.bos_token_id) == 1
    assert prompt_ids[-1] != evaluator.tokenizer.eos_token_id
    assert prompt_text.startswith(evaluator.tokenizer.bos_token)
    assert "<|system|>" not in prompt_text
    assert "<|user|>" not in prompt_text
    assert "<|assistant|>" not in prompt_text
