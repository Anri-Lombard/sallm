from __future__ import annotations

from typing import Any


def safe_format_prompt(prompt: str, values: dict[str, Any]) -> str:
    """Format a prompt template with safe string conversion.

    Handles None values and conversion errors gracefully.

    Args:
        prompt: Format string template
        values: Dictionary of values to substitute

    Returns:
        Formatted string
    """
    safe: dict[str, str] = {}
    for k, v in values.items():
        try:
            if v is None:
                safe[k] = ""
            else:
                safe[k] = str(v).strip()
        except Exception:
            safe[k] = str(v)
    return prompt.format(**safe)
