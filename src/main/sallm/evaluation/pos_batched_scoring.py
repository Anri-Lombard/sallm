"""Cross-row full-prefix POS scoring."""

from __future__ import annotations

from typing import Any

import torch

from sallm.evaluation.constrained_label_scoring import (
    _pad_batch,
    append_text,
    chat_messages_prefix,
    continuation_ids,
)


def decode_pos_tuple_contract_batch(
    *,
    model: Any,
    tokenizer: Any,
    prompt_messages_batch: list[list[dict[str, str]]],
    tokens_batch: list[list[str]],
    labels: list[str],
    score_mode: str,
    pad_token_id: int,
    pad_to_multiple_of: int | None,
    device: torch.device,
) -> list[tuple[list[str], float]]:
    if len(prompt_messages_batch) != len(tokens_batch) or not tokens_batch:
        raise ValueError("POS batch inputs must have the same non-zero length.")
    if score_mode not in {"sum", "mean"}:
        raise ValueError(f"Unsupported score mode: {score_mode}")

    contexts = [
        chat_messages_prefix(tokenizer, messages) for messages in prompt_messages_batch
    ]
    predictions = [[] for _ in tokens_batch]
    total_scores = [0.0 for _ in tokens_batch]

    for token_index in range(max(map(len, tokens_batch))):
        active = [
            row for row, tokens in enumerate(tokens_batch) if token_index < len(tokens)
        ]
        label_ids_by_row: list[dict[str, list[int]]] = []
        for row in active:
            context_text, context_ids = contexts[row]
            prefix = ("[" if token_index == 0 else ", ") + (
                f"({tokens_batch[row][token_index]!r}, '"
            )
            context_text, context_ids = append_text(
                tokenizer, context_text, context_ids, prefix
            )
            contexts[row] = (context_text, context_ids)
            label_ids_by_row.append(
                {
                    label: continuation_ids(tokenizer, context_text, context_ids, label)
                    for label in labels
                }
            )

        one_token_labels = all(
            len(ids) == 1
            for label_ids in label_ids_by_row
            for ids in label_ids.values()
        )
        sequences = (
            [torch.tensor(contexts[row][1], dtype=torch.long) for row in active]
            if one_token_labels
            else [
                torch.tensor(contexts[row][1] + label_ids[label], dtype=torch.long)
                for row, label_ids in zip(active, label_ids_by_row, strict=True)
                for label in labels
            ]
        )
        input_ids, attention_mask = _pad_batch(
            sequences, pad_token_id, pad_to_multiple_of
        )
        with torch.no_grad():
            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                use_cache=False,
            ).logits

        for batch_row, row in enumerate(active):
            context_text, context_ids = contexts[row]
            if one_token_labels:
                log_probs = torch.log_softmax(
                    logits[batch_row, len(context_ids) - 1], dim=-1
                )
                scores = {
                    label: float(
                        log_probs[label_ids_by_row[batch_row][label][0]].item()
                    )
                    for label in labels
                }
            else:
                scores = {}
                for label_index, label in enumerate(labels):
                    sequence_row = batch_row * len(labels) + label_index
                    token_scores = [
                        torch.log_softmax(
                            logits[sequence_row, len(context_ids) + offset - 1], dim=-1
                        )[token_id]
                        for offset, token_id in enumerate(
                            label_ids_by_row[batch_row][label]
                        )
                    ]
                    stacked = torch.stack(token_scores)
                    score = stacked.sum() if score_mode == "sum" else stacked.mean()
                    scores[label] = float(score.item())
            label = max(scores, key=scores.__getitem__)
            predictions[row].append(label)
            total_scores[row] += scores[label]
            suffix = label + "')"
            if token_index == len(tokens_batch[row]) - 1:
                suffix += "]"
            contexts[row] = append_text(tokenizer, context_text, context_ids, suffix)

    return list(zip(predictions, total_scores, strict=True))
