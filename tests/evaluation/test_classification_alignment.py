from __future__ import annotations

import unicodedata

import torch
import torch.nn.functional as F
from datasets import Dataset
from sallm.chat_template import (
    assistant_content_prefix,
    install_canonical_chat_template,
)
from sallm.evaluation.classification_metrics import ClassificationEvaluator
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def _build_in_memory_bpe_chat_tokenizer() -> PreTrainedTokenizerFast:
    """Build a ByteLevel BPE tokenizer in memory so CI needs no local files."""
    corpus = [
        "market bank stock trade finance business",
        "company profit revenue economy business",
        "football match goal stadium team sports",
        "player league tournament score sports",
        "hospital doctor medicine clinic health",
        "patient vaccine nurse treatment health",
        "Article about football match. sports business health",
        "\n        business\n        sports\n        health",
    ] * 8
    raw = Tokenizer(BPE(unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=256,
        min_frequency=1,
        show_progress=False,
        special_tokens=[
            "[BOS]",
            "[EOS]",
            "[PAD]",
            "[UNK]",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
        ],
    )
    raw.train_from_iterator(corpus, trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        additional_special_tokens=["<|system|>", "<|user|>", "<|assistant|>"],
    )
    install_canonical_chat_template(tokenizer, force=True)
    return tokenizer


def test_classification_prompt_aligns_with_training_assistant_prefix() -> None:
    tokenizer = _build_in_memory_bpe_chat_tokenizer()
    assert assistant_content_prefix(tokenizer) == "\n        "

    evaluator = ClassificationEvaluator(tokenizer)
    prompt_messages = [{"role": "user", "content": "Article about football match."}]
    prompt_text = evaluator._build_prompt_text(
        prompt_messages=prompt_messages,
        fallback_template=None,
        system_message=None,
    )
    assert prompt_text.endswith("<|assistant|>\n        ")

    context_ids, continuation_ids = evaluator._encode_choice_pair(prompt_text, "sports")
    training_rendered = tokenizer.apply_chat_template(
        [*prompt_messages, {"role": "assistant", "content": "sports"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    expected_without_eos = training_rendered.removesuffix(tokenizer.eos_token)
    assert context_ids + continuation_ids == tokenizer.encode(
        expected_without_eos, add_special_tokens=False
    )
    # Only the label token(s) are scored, not the leading newline/indent tokens.
    assert continuation_ids == tokenizer.encode(" sports", add_special_tokens=False)


def test_classification_metric_improves_after_training_on_canonical_template(
    monkeypatch,
) -> None:
    torch.manual_seed(7)
    tokenizer = _build_in_memory_bpe_chat_tokenizer()
    labels = ["business", "sports", "health"]
    samples = [
        ("market bank stock trade finance", "business"),
        ("company profit revenue economy", "business"),
        ("football match goal stadium team", "sports"),
        ("player league tournament score", "sports"),
        ("hospital doctor medicine clinic", "health"),
        ("patient vaccine nurse treatment", "health"),
    ]
    rows = [
        {
            "lang": "eng",
            "template_id": "synthetic_cls",
            "messages": [
                {"role": "user", "content": text},
                {"role": "assistant", "content": label},
            ],
        }
        for text, label in samples
    ]
    dataset = Dataset.from_list(rows)

    cfg = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(cfg)
    evaluator = ClassificationEvaluator(tokenizer, max_samples_per_lang=None)
    monkeypatch.setattr(evaluator, "_get_label_choices", lambda _tid: labels)

    before = evaluator.evaluate(model, dataset)["classification/all_macro_f1"]

    # Train directly on canonical chat template renderings, matching SFTTrainer.
    encoded = [
        torch.tensor(
            tokenizer.apply_chat_template(
                row["messages"], tokenize=True, return_dict=False
            ),
            dtype=torch.long,
        )
        for row in rows
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(80):
        optimizer.zero_grad()
        loss = torch.zeros(())
        for seq in encoded:
            logits = model(input_ids=seq.unsqueeze(0), use_cache=False).logits[0, :-1]
            loss = loss + F.cross_entropy(logits, seq[1:])
        (loss / len(encoded)).backward()
        optimizer.step()

    after = evaluator.evaluate(model, dataset)["classification/all_macro_f1"]
    assert after > before
    assert after >= 0.85


def test_classification_labels_match_across_nfd_and_nfc() -> None:
    evaluator = ClassificationEvaluator(tokenizer=object())
    nfd = unicodedata.normalize("NFD", "Šôthô")
    nfc = unicodedata.normalize("NFC", "šôthô")
    assert evaluator._labels_match(nfd, nfc)
