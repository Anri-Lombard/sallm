from pathlib import Path

from sallm.chat_template import (
    CANONICAL_CHAT_TEMPLATE,
    install_canonical_chat_template,
)
from sallm.evaluation import generation_metrics, harness, lm_eval_runner
from sallm.evaluation.classification_metrics import ClassificationEvaluator
from sallm.evaluation.constrained_label_scoring import chat_messages_prefix
from sallm.fine_tune.run import _configure_chat_template
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_ROOT = ROOT / "tokenizer" / "sallm_bpe_tokenizer"
MESSAGES = [{"role": "user", "content": "Tag this sentence."}]


def _fresh_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ROOT,
        local_files_only=True,
    )
    assert tokenizer.chat_template is None
    return tokenizer


def _prompt_ids(tokenizer) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
    )
    direct = list(
        tokenizer.apply_chat_template(
            MESSAGES,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
        )
    )
    retokenized = tokenizer.encode(rendered, add_special_tokens=False)
    assert direct == retokenized
    assert direct[0] == tokenizer.bos_token_id
    assert direct.count(tokenizer.bos_token_id) == 1
    assert direct[-1] != tokenizer.eos_token_id
    return direct


def test_pretraining_tokenizer_wraps_documents_in_bos_and_eos() -> None:
    tokenizer = _fresh_tokenizer()
    document_ids = tokenizer.encode("Ordinary pretraining document.")

    assert document_ids[0] == tokenizer.bos_token_id
    assert document_ids[-1] == tokenizer.eos_token_id
    assert document_ids.count(tokenizer.bos_token_id) == 1
    assert document_ids.count(tokenizer.eos_token_id) == 1


def test_train_generation_classification_constrained_and_harness_share_ids(
    monkeypatch,
) -> None:
    training_tokenizer = _fresh_tokenizer()
    assert _configure_chat_template(training_tokenizer)

    monkeypatch.setattr(generation_metrics, "eval_load", lambda _name: object())
    generation_tokenizer = _fresh_tokenizer()
    generation_metrics.GenerationEvaluator(generation_tokenizer)

    classification_tokenizer = _fresh_tokenizer()
    classification_evaluator = ClassificationEvaluator(classification_tokenizer)
    classification_tokenizer.chat_template = (
        classification_evaluator._get_fallback_template()
    )

    constrained_tokenizer = _fresh_tokenizer()
    _, constrained_ids = chat_messages_prefix(constrained_tokenizer, MESSAGES)

    harness_tokenizer = harness._prepare_tokenizer(_fresh_tokenizer())

    expected = _prompt_ids(training_tokenizer)
    assert _prompt_ids(generation_tokenizer) == expected
    assert _prompt_ids(classification_tokenizer) == expected
    assert constrained_ids == expected
    assert _prompt_ids(harness_tokenizer) == expected
    assert {
        training_tokenizer.chat_template,
        generation_tokenizer.chat_template,
        classification_tokenizer.chat_template,
        constrained_tokenizer.chat_template,
        harness_tokenizer.chat_template,
    } == {CANONICAL_CHAT_TEMPLATE}


def test_lm_eval_fallback_is_the_same_canonical_template() -> None:
    tokenizer = _fresh_tokenizer()
    tokenizer.chat_template = lm_eval_runner._fallback_chat_template()

    assert tokenizer.chat_template == CANONICAL_CHAT_TEMPLATE
    assert _prompt_ids(tokenizer)


def test_existing_model_specific_template_is_preserved() -> None:
    tokenizer = _fresh_tokenizer()
    tokenizer.chat_template = "model-specific-template"

    assert not install_canonical_chat_template(tokenizer)
    assert tokenizer.chat_template == "model-specific-template"


def test_force_replaces_model_specific_template() -> None:
    tokenizer = _fresh_tokenizer()
    tokenizer.chat_template = "model-specific-template"

    assert install_canonical_chat_template(tokenizer, force=True)
    assert tokenizer.chat_template == CANONICAL_CHAT_TEMPLATE
    assert _prompt_ids(tokenizer)
