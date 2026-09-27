"""Closed-label continuation scoring for decoder-only task evaluators."""

from __future__ import annotations

from typing import Any

import torch

from sallm.chat_template import install_canonical_chat_template


def encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def chat_messages_prefix(
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> tuple[str, list[int]]:
    install_canonical_chat_template(tokenizer)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    direct_ids = list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
        )
    )
    rendered_ids = encode(tokenizer, text)
    if rendered_ids != direct_ids:
        raise ValueError(
            "Rendered prompt tokenization differs from direct chat-template "
            "tokenization."
        )
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and rendered_ids and rendered_ids[-1] == eos_id:
        raise ValueError("Constrained-scoring prompt ends in EOS.")
    return text, rendered_ids


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
    max_len = max(int(sequence.numel()) for sequence in sequences)
    if pad_to_multiple_of and max_len % pad_to_multiple_of:
        max_len = ((max_len // pad_to_multiple_of) + 1) * pad_to_multiple_of

    input_ids = torch.full(
        (len(sequences), max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        length = int(sequence.numel())
        input_ids[row, :length] = sequence
        attention_mask[row, :length] = 1
    return input_ids, attention_mask


def score_labels(
    *,
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
        input_ids, attention_mask = _pad_batch(
            [torch.tensor(context_ids, dtype=torch.long)],
            pad_token_id,
            pad_to_multiple_of,
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
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
    input_ids, attention_mask = _pad_batch(
        sequences,
        pad_token_id,
        pad_to_multiple_of,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    scores: dict[str, float] = {}
    context_len = len(context_ids)
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    score_batch_size = 1 if "mamba" in model_type else len(labels)
    with torch.no_grad():
        for start in range(0, len(labels), score_batch_size):
            stop = start + score_batch_size
            logits = model(
                input_ids=input_ids[start:stop],
                attention_mask=attention_mask[start:stop],
                use_cache=False,
            ).logits
            for local_row, label in enumerate(labels[start:stop]):
                token_scores: list[torch.Tensor] = []
                for offset, token_id in enumerate(label_ids[label]):
                    logit_index = context_len + offset - 1
                    log_probs = torch.log_softmax(
                        logits[local_row, logit_index], dim=-1
                    )
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


def decode_pos_tuple_contract(
    *,
    model: Any,
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    tokens: list[str],
    labels: list[str],
    score_mode: str,
    pad_token_id: int,
    pad_to_multiple_of: int | None,
    device: torch.device,
    incremental_cache: bool = False,
) -> tuple[list[str], float]:
    if incremental_cache:
        return _decode_pos_tuple_contract_cached(
            model=model,
            tokenizer=tokenizer,
            prompt_messages=prompt_messages,
            tokens=tokens,
            labels=labels,
            score_mode=score_mode,
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            device=device,
        )

    context_text, context_ids = chat_messages_prefix(tokenizer, prompt_messages)
    predictions: list[str] = []
    total_score = 0.0

    for index, token in enumerate(tokens):
        prefix = ("[" if index == 0 else ", ") + f"({repr(token)}, '"
        context_text, context_ids = append_text(
            tokenizer,
            context_text,
            context_ids,
            prefix,
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
        suffix = label + "')"
        if index == len(tokens) - 1:
            suffix += "]"
        context_text, context_ids = append_text(
            tokenizer,
            context_text,
            context_ids,
            suffix,
        )

    return predictions, total_score


def _decode_pos_tuple_contract_cached(
    *,
    model: Any,
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    tokens: list[str],
    labels: list[str],
    score_mode: str,
    pad_token_id: int,
    pad_to_multiple_of: int | None,
    device: torch.device,
) -> tuple[list[str], float]:
    context_text, context_ids = chat_messages_prefix(tokenizer, prompt_messages)
    predictions: list[str] = []
    total_score = 0.0
    past_key_values: Any = None
    cached_length = 0

    for index, token in enumerate(tokens):
        prefix = ("[" if index == 0 else ", ") + f"({repr(token)}, '"
        context_text, context_ids = append_text(
            tokenizer,
            context_text,
            context_ids,
            prefix,
        )
        ids_by_label = {
            label: continuation_ids(tokenizer, context_text, context_ids, label)
            for label in labels
        }
        if any(len(ids_by_label[label]) != 1 for label in labels):
            return decode_pos_tuple_contract(
                model=model,
                tokenizer=tokenizer,
                prompt_messages=prompt_messages,
                tokens=tokens,
                labels=labels,
                score_mode=score_mode,
                pad_token_id=pad_token_id,
                pad_to_multiple_of=pad_to_multiple_of,
                device=device,
                incremental_cache=False,
            )

        new_ids = context_ids[cached_length:]
        if not new_ids:
            raise ValueError("Incremental POS scoring produced an empty cache update.")
        input_ids = torch.tensor([new_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        past_key_values = outputs.past_key_values
        if past_key_values is None:
            return decode_pos_tuple_contract(
                model=model,
                tokenizer=tokenizer,
                prompt_messages=prompt_messages,
                tokens=tokens,
                labels=labels,
                score_mode=score_mode,
                pad_token_id=pad_token_id,
                pad_to_multiple_of=pad_to_multiple_of,
                device=device,
                incremental_cache=False,
            )
        cached_length = len(context_ids)
        log_probs = torch.log_softmax(outputs.logits[0, -1], dim=-1)
        scores = {
            label: float(log_probs[ids_by_label[label][0]].item()) for label in labels
        }
        label = max(scores, key=scores.__getitem__)
        predictions.append(label)
        total_score += scores[label]
        suffix = label + "')"
        if index == len(tokens) - 1:
            suffix += "]"
        context_text, context_ids = append_text(
            tokenizer,
            context_text,
            context_ids,
            suffix,
        )

    return predictions, total_score
