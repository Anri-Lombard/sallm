from __future__ import annotations

from typing import Any

from sallm.data.formatters.base import safe_format_prompt
from sallm.templates import registry as tmpl


def format_pos(
    ex: dict[str, Any],
    template_id: str | None = None,
    upos_class_names: list[str] | None = None,
) -> list[dict[str, str]]:
    """Format a POS tagging example as messages.

    Args:
        ex: Dataset example with 'tokens' and 'upos'
        template_id: Optional template ID to use
        upos_class_names: Optional list mapping UPOS tag IDs to names

    Returns:
        List of message dicts with 'role' and 'content'
    """
    tokens: list[str] = ex["tokens"]
    raw_upos = ex["upos"]

    if raw_upos and isinstance(raw_upos[0], int) and upos_class_names:
        upos_tags = [upos_class_names[i] for i in raw_upos]
    else:
        upos_tags = [str(t) for t in raw_upos]

    if len(tokens) != len(upos_tags):
        raise ValueError(
            "MasakhaPOS sample length mismatch: "
            f"tokens={len(tokens)} vs upos={len(upos_tags)}"
        )

    tuple_list_str = (
        "["
        + ", ".join(
            f"({repr(tok)}, {repr(tag)})"
            for tok, tag in zip(tokens, upos_tags, strict=False)
        )
        + "]"
    )

    tokens_repr = "[" + ", ".join(repr(t) for t in tokens) + "]"

    if template_id:
        spec = tmpl.get(template_id)
        user_prompt = safe_format_prompt(spec.prompt, {"tokens": tokens_repr})
    else:
        user_prompt = (
            "Please provide UPOS tags for each token as a list of (token, TAG) "
            f"tuples.\nSentence: {tokens_repr}\nOutput: "
        )

    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": tuple_list_str},
    ]


def get_upos_class_names(raw_ds) -> list[str] | None:
    """Extract UPOS class names from dataset features if available."""
    try:
        upos_feat = raw_ds.features.get("upos")
        if hasattr(upos_feat, "feature") and hasattr(upos_feat.feature, "names"):
            return list(upos_feat.feature.names)
    except Exception:
        pass
    return None
