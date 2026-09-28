"""Prefix-cached closed-label POS scoring (28 Sep 2026), a faster drop-in for the per-word score_labels call.

The reference re-runs the whole context once per candidate tag. Here the append-only context is fed once through the
model's cache; each tag's first token is scored from the last logits, and multi-token tags continue from a copy of the
cache (one row per tag). Same tokenisation, same mean-logprob rule. Validated on Kombuys (paper repo data/cached_pos):
100% tag agreement with the reference on all four architectures.
"""
from __future__ import annotations

import copy
from typing import Any

import torch


def _get_cache(out: Any) -> tuple[str, Any]:
    for name in ("past_key_values", "cache_params"):
        c = getattr(out, name, None)
        if c is not None:
            return name, c
    raise RuntimeError("model returned no cache")


def _repeat_batch(obj: Any, n: int) -> Any:
    if torch.is_tensor(obj):
        return obj.repeat_interleave(n, dim=0) if obj.dim() > 0 and obj.shape[0] == 1 else obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_repeat_batch(x, n) for x in obj)
    if isinstance(obj, dict):
        return {k: _repeat_batch(v, n) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        new = copy.copy(obj)
        for k, v in vars(obj).items():
            setattr(new, k, _repeat_batch(v, n))
        return new
    return obj


def _expand(cache: Any, n: int) -> Any:
    c = copy.deepcopy(cache)
    try:
        c.batch_repeat_interleave(n)
        return c
    except AttributeError:  # FLA caches inherit the method but their layers lack it
        return _repeat_batch(copy.deepcopy(cache), n)


def _forward(model: Any, x: torch.Tensor, kind: str | None, cache: Any) -> Any:
    """Models whose cached path takes one token per step (Mamba-2) are fed stepwise."""
    if cache is None or not getattr(model, "_one_token_steps", False):
        try:
            return model(input_ids=x, use_cache=True, **({kind: cache} if cache is not None else {}))
        except ValueError as exc:
            if "single new token" not in str(exc):
                raise
            model._one_token_steps = True
    logits = []
    out = None
    for t in range(x.shape[1]):
        out = model(input_ids=x[:, t:t + 1], use_cache=True, **({kind: cache} if cache is not None else {}))
        kind, cache = _get_cache(out)
        logits.append(out.logits)
    out.logits = torch.cat(logits, 1)
    return out


class CachedScorer:
    def __init__(self, model: Any, device: torch.device):
        self.model, self.device = model, device
        self.cache, self.kind, self.fed, self.last_logits = None, None, [], None

    def feed(self, ids: list[int]) -> None:
        if ids[: len(self.fed)] != self.fed:  # context was not append-only (e.g. truncated): start again
            self.cache, self.kind, self.fed = None, None, []
        new = ids[len(self.fed):]
        if not new:
            return
        with torch.no_grad():
            out = _forward(self.model, torch.tensor([new], device=self.device), self.kind, self.cache)
        self.kind, self.cache = _get_cache(out)
        self.last_logits = out.logits[0, -1].float()
        self.fed = list(ids)

    def score(self, label_ids: dict[str, list[int]], labels: list[str]) -> tuple[str, float]:
        first = torch.log_softmax(self.last_logits, -1)
        scores = {lab: [first[label_ids[lab][0]]] for lab in labels}
        multi = [lab for lab in labels if len(label_ids[lab]) > 1]
        if multi:
            width = max(len(label_ids[lab]) for lab in multi) - 1
            rows = torch.tensor([label_ids[lab][:-1] + [0] * (width - len(label_ids[lab]) + 1) for lab in multi],
                                device=self.device)
            with torch.no_grad():
                lp = torch.log_softmax(_forward(self.model, rows, self.kind, _expand(self.cache, len(multi))).logits.float(), -1)
            for i, lab in enumerate(multi):
                for j, tid in enumerate(label_ids[lab][1:]):
                    scores[lab].append(lp[i, j, tid])
        mean = {lab: float(torch.stack(v).mean()) for lab, v in scores.items()}
        best = max(labels, key=lambda lab: mean[lab])  # first maximum in label order, as score_labels
        return best, mean[best]


def enabled(model: Any, n_labels: int) -> bool:
    """Mamba-2 only gains with large tag sets (Kombuys: 0.5x on 17 UPOS tags, 3.9x on NCHLT's)."""
    import os

    if os.environ.get("FFT_POS_CACHED", "0") == "0" or not hasattr(model, "config"):
        return False
    return "mamba" not in type(model).__name__.lower() or n_labels > 20
