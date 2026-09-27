"""Shared helpers for constrained closed-label decoder scoring.

These utilities keep the diagnostic POS/NER evaluators on the same scoring
contract: each output position is scored by appending each legal label to the
current decoder context and choosing the label with the best log probability.
"""

from __future__ import annotations

from typing import Any

import torch


def tokens_repr(tokens: list[str]) -> str:
    return "[" + ", ".join(repr(token) for token in tokens) + "]"


def chat_prefix(tokenizer: Any, user_prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{user_prompt}\n"


def encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def continuation_ids(
    tokenizer: Any,
    context_text: str,
    context_ids: list[int],
    continuation: str,
) -> list[int]:
    full_ids = encode(tokenizer, context_text + continuation)
    if full_ids[: len(context_ids)] == context_ids:
        return full_ids[len(context_ids) :]
    return encode(tokenizer, continuation)


def append_text(
    tokenizer: Any,
    context_text: str,
    context_ids: list[int],
    continuation: str,
) -> tuple[str, list[int]]:
    extra_ids = continuation_ids(tokenizer, context_text, context_ids, continuation)
    return context_text + continuation, context_ids + extra_ids


def _pad_batch(
    sequences: list[torch.Tensor],
    pad_token_id: int,
    pad_to_multiple_of: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(seq.numel()) for seq in sequences)
    if pad_to_multiple_of and max_len % pad_to_multiple_of:
        max_len = ((max_len // pad_to_multiple_of) + 1) * pad_to_multiple_of

    input_ids = torch.full(
        (len(sequences), max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, seq in enumerate(sequences):
        length = int(seq.numel())
        input_ids[row, :length] = seq
        attention_mask[row, :length] = 1
    return input_ids, attention_mask


def score_labels(
    model: Any,
    context_ids: list[int],
    label_ids: dict[str, list[int]],
    labels: list[str],
    score_mode: str,
    pad_token_id: int,
    pad_to_multiple_of: int | None,
    device: torch.device,
) -> tuple[str, float, dict[str, float]]:
    for label in labels:
        if not label_ids.get(label):
            raise ValueError(f"Label {label!r} encoded to an empty continuation.")

    if context_ids and all(len(label_ids[label]) == 1 for label in labels):
        # ponytail: single-token labels need one next-token forward;
        # fallback handles multi-token labels.
        input_ids, attention_mask = _pad_batch(
            [torch.tensor(context_ids, dtype=torch.long)],
            pad_token_id,
            pad_to_multiple_of,
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits

        log_probs = torch.log_softmax(logits[0, len(context_ids) - 1], dim=-1)
        scores = {
            label: float(log_probs[label_ids[label][0]].item()) for label in labels
        }
        best_label = max(scores, key=scores.__getitem__)
        return best_label, scores[best_label], scores

    sequences = [
        torch.tensor(context_ids + label_ids[label], dtype=torch.long)
        for label in labels
    ]
    input_ids, attention_mask = _pad_batch(sequences, pad_token_id, pad_to_multiple_of)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits

    scores: dict[str, float] = {}
    context_len = len(context_ids)
    for row_idx, label in enumerate(labels):
        token_scores: list[torch.Tensor] = []
        for offset, token_id in enumerate(label_ids[label]):
            logit_index = context_len + offset - 1
            log_probs = torch.log_softmax(logits[row_idx, logit_index], dim=-1)
            token_scores.append(log_probs[token_id])
        stacked = torch.stack(token_scores)
        if score_mode == "sum":
            score = float(stacked.sum().item())
        elif score_mode == "mean":
            score = float(stacked.mean().item())
        else:
            raise ValueError(f"Unsupported score mode: {score_mode}")
        scores[label] = score

    best_label = max(scores, key=scores.__getitem__)
    return best_label, scores[best_label], scores


def decode_tagseq_contract(
    model: Any,
    tokenizer: Any,
    prompt: str,
    tokens: list[str],
    labels: list[str],
    score_mode: str,
    pad_token_id: int,
    pad_to_multiple_of: int | None,
    device: torch.device,
) -> tuple[list[str], float]:
    context_text = chat_prefix(tokenizer, prompt)
    context_ids = encode(tokenizer, context_text)
    predictions: list[str] = []
    total_score = 0.0

    for index, _token in enumerate(tokens):
        if index > 0:
            context_text, context_ids = append_text(
                tokenizer, context_text, context_ids, " "
            )
        ids_by_label = {
            label: continuation_ids(tokenizer, context_text, context_ids, label)
            for label in labels
        }
        label, score, _ = score_labels(
            model=model,
            context_ids=context_ids,
            label_ids=ids_by_label,
            labels=labels,
            score_mode=score_mode,
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            device=device,
        )
        predictions.append(label)
        total_score += score
        context_text, context_ids = append_text(
            tokenizer, context_text, context_ids, label
        )

    return predictions, total_score
