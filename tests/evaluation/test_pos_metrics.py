import pytest
import torch
from datasets import Dataset
from sallm.evaluation.constrained_label_scoring import (
    chat_messages_prefix,
    decode_pos_tuple_contract,
)
from sallm.evaluation.pos_batched_scoring import (
    decode_pos_tuple_contract_batch,
)
from sallm.evaluation.pos_metrics import (
    CANONICAL_POS_LANGUAGES,
    CANONICAL_POS_TEMPLATES,
    PosEvaluator,
    aggregate_pos_cell_accuracies,
    parse_pos_tuple_completion,
)


def test_parse_pos_tuple_completion_preserves_one_label_per_token() -> None:
    tokens, labels = parse_pos_tuple_completion(
        "[('Ndiyahamba', 'VERB'), ('.', 'PUNCT')]"
    )

    assert tokens == ["Ndiyahamba", "."]
    assert labels == ["VERB", "PUNCT"]


def test_parse_pos_tuple_completion_rejects_illegal_labels() -> None:
    with pytest.raises(ValueError, match="Illegal UPOS label"):
        parse_pos_tuple_completion("[('word', 'INVALID')]")


def test_pos_selection_is_mean_of_language_prompt_cells() -> None:
    cells = {
        (language, template): (1, 2)
        for language in CANONICAL_POS_LANGUAGES
        for template in CANONICAL_POS_TEMPLATES
    }
    cells[("tsn", "masakhane_pos_tagging/lm_eval_p1")] = (2, 2)

    aggregate, metrics = aggregate_pos_cell_accuracies(cells)

    assert aggregate == (1.0 + 11 * 0.5) / 12
    assert len(metrics) == 12


def test_pos_selection_rejects_missing_prompt_cell() -> None:
    cells = {
        (language, template): (1, 2)
        for language in CANONICAL_POS_LANGUAGES
        for template in CANONICAL_POS_TEMPLATES
    }
    cells.pop(("zul", "masakhane_pos_tagging/lm_eval_p4"))

    with pytest.raises(ValueError, match="coverage mismatch"):
        aggregate_pos_cell_accuracies(cells)


def test_pos_evaluator_accepts_monolingual_five_prompt_grid(monkeypatch) -> None:
    templates = [f"masakhane_pos_tagging/lm_eval_p{i}" for i in range(1, 6)]
    dataset = Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": "tag this"},
                    {"role": "assistant", "content": "[('word', 'NOUN')]"},
                ]
                for _ in templates
            ],
            "lang": ["tsn"] * len(templates),
            "template_id": templates,
        }
    )

    monkeypatch.setattr(
        "sallm.evaluation.pos_metrics.decode_pos_tuple_contract",
        lambda **_kwargs: (["NOUN"], [0.0]),
    )

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

    class Model:
        device = torch.device("cpu")

        def eval(self) -> None:
            pass

    metrics = PosEvaluator(Tokenizer()).evaluate(Model(), dataset)

    assert metrics["eval/all_token_accuracy"] == 1.0
    assert len(metrics) == 6


def test_constrained_prompt_rejects_terminal_eos() -> None:
    class Tokenizer:
        chat_template = "template"
        eos_token_id = 2

        def apply_chat_template(self, _messages, *, tokenize, **_kwargs):
            return [5, 2] if tokenize else "rendered"

        def __call__(self, _text, *, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [5, 2]}

    with pytest.raises(ValueError, match="ends in EOS"):
        chat_messages_prefix(
            Tokenizer(),
            [{"role": "user", "content": "tag this"}],
        )


@pytest.mark.parametrize("labels", [["A", "B"], ["A", "BC"]])
def test_batched_pos_matches_serial_with_fewer_forwards(labels: list[str]) -> None:
    class Tokenizer:
        chat_template = "template"
        eos_token_id = 2

        def apply_chat_template(self, _messages, *, tokenize, **_kwargs):
            return [80] if tokenize else "P"

        def __call__(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [ord(char) for char in text]}

    class Model:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.calls = 0

        def __call__(
            self,
            *,
            input_ids,
            attention_mask,
            **_kwargs,
        ):
            self.calls += 1
            logits = torch.zeros((*input_ids.shape, 128))
            for row in range(input_ids.shape[0]):
                running = 0
                for index, token_id in enumerate(input_ids[row].tolist()):
                    if attention_mask[row, index] == 0:
                        continue
                    running += token_id
                    logits[row, index, ord("A")] = 2.0 if running % 2 == 0 else 1.0
                    logits[row, index, ord("B")] = 1.0 if running % 2 == 0 else 2.0
            return type("Output", (), {"logits": logits})()

    kwargs = {
        "tokenizer": Tokenizer(),
        "labels": labels,
        "score_mode": "mean",
        "pad_token_id": 0,
        "pad_to_multiple_of": None,
        "device": torch.device("cpu"),
    }
    messages = [[{"role": "user", "content": "tag"}]] * 2
    tokens = [["one", "two", "three"], ["four", "five"]]
    serial_model = Model()
    batch_model = Model()

    serial = [
        decode_pos_tuple_contract(
            model=serial_model,
            prompt_messages=row_messages,
            tokens=row_tokens,
            **kwargs,
        )
        for row_messages, row_tokens in zip(messages, tokens, strict=True)
    ]
    batched = decode_pos_tuple_contract_batch(
        model=batch_model,
        prompt_messages_batch=messages,
        tokens_batch=tokens,
        **kwargs,
    )

    assert batched == serial
    assert batch_model.calls < serial_model.calls
