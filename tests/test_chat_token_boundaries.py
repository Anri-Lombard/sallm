from __future__ import annotations

import torch
from sallm.evaluation.classification_metrics import ClassificationEvaluator
from sallm.evaluation.generation_metrics import GenerationEvaluator
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


def _chat_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<|user|>": 3,
        "<|assistant|>": 4,
        "hello": 5,
        "answer": 6,
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = (
        "{{ bos_token }}<|user|> {{ messages[0]['content'] }}{{ eos_token }}"
        "<|assistant|>"
    )
    return tokenizer


def test_classification_uses_exact_rendered_chat_ids() -> None:
    tokenizer = _chat_tokenizer()
    evaluator = ClassificationEvaluator(tokenizer)
    rendered = evaluator._build_prompt_text(
        [{"role": "user", "content": "hello"}],
        fallback_template=None,
        system_message=None,
    )

    context_ids, continuation_ids = evaluator._encode_choice_pair(rendered, "answer")

    assert context_ids.count(tokenizer.bos_token_id) == 1
    assert context_ids[-1] == tokenizer.convert_tokens_to_ids("<|assistant|>")
    assert tokenizer.eos_token_id not in context_ids[-1:]
    assert continuation_ids == [tokenizer.convert_tokens_to_ids("answer")]


def test_generation_uses_exact_rendered_chat_ids() -> None:
    tokenizer = _chat_tokenizer()
    evaluator = GenerationEvaluator.__new__(GenerationEvaluator)
    evaluator.tokenizer = tokenizer
    evaluator.max_new_tokens = 4
    evaluator.decoding_config = type(
        "Decoding",
        (),
        {"to_generate_kwargs": lambda self: {}},
    )()
    batch = [
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "answer"},
            ]
        }
    ]

    prepared = evaluator._prepare_generation_batch(
        batch,
        fallback_template=None,
        device=torch.device("cpu"),
        model_ctx_limit=32,
        pad_id=tokenizer.pad_token_id,
        eos_id=tokenizer.eos_token_id,
    )

    assert prepared is not None
    input_ids = prepared[3][0].tolist()
    assert input_ids.count(tokenizer.bos_token_id) == 1
    assert input_ids[-1] == tokenizer.convert_tokens_to_ids("<|assistant|>")
    assert tokenizer.eos_token_id not in input_ids[-1:]


def test_canonical_fallback_renders_one_bos_and_open_assistant_turn() -> None:
    tokenizer = _chat_tokenizer()
    tokenizer.chat_template = None
    evaluator = ClassificationEvaluator(tokenizer)
    fallback = evaluator._get_fallback_template()

    rendered = evaluator._build_prompt_text(
        [{"role": "user", "content": "hello"}],
        fallback_template=fallback,
        system_message=None,
    )
    input_ids = tokenizer.encode(rendered, add_special_tokens=False)

    assert input_ids.count(tokenizer.bos_token_id) == 1
    assert input_ids[-1] == tokenizer.convert_tokens_to_ids("<|assistant|>")
    assert tokenizer.eos_token_id not in input_ids[-1:]
