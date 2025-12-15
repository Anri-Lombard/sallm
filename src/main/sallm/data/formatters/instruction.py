from __future__ import annotations

from typing import Any

from sallm.data.formatters.base import safe_format_prompt
from sallm.templates import registry as tmpl

FIELD_PAIRS = [
    ("source", "target"),  # T2X dataset
    ("text", "title"),  # AfriHG dataset
]


def _extract_instruction_pair(ex: dict[str, Any]) -> tuple[str, str]:
    """Extract (user_prompt, assistant_response) from dataset example."""
    for user_key, asst_key in FIELD_PAIRS:
        if user_key in ex and asst_key in ex:
            return str(ex[user_key]), str(ex[asst_key])

    raise KeyError(
        f"Unable to locate instruction/response fields. "
        f"Expected one of: {FIELD_PAIRS}. Got keys: {list(ex.keys())}"
    )


def format_instruction(
    ex: dict[str, Any], template_id: str | None = None
) -> list[dict[str, str]]:
    """Format an example as instruction-style messages."""
    user_text, assistant_text = _extract_instruction_pair(ex)

    if template_id:
        try:
            spec = tmpl.get(template_id)
            user_text = safe_format_prompt(spec.prompt, ex)
        except Exception:
            pass

    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
