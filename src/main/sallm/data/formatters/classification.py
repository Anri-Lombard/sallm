from __future__ import annotations

from typing import Any

from sallm.data.formatters.base import safe_format_prompt
from sallm.templates import registry as tmpl


def format_classification(
    ex: dict[str, Any],
    template_id: str,
    label_column: str = "label",
) -> list[dict[str, str]]:
    """Format a classification example as messages.

    Args:
        ex: Dataset example
        template_id: Template ID to use
        label_column: Column containing the label

    Returns:
        List of message dicts with 'role' and 'content'
    """
    template_spec = tmpl.get(template_id)
    if not template_spec.label_mapping:
        raise ValueError(
            f"Template '{template_spec.id}' is for a classification task "
            "but is missing 'label_mapping'."
        )

    user_prompt = safe_format_prompt(template_spec.prompt, ex)
    raw_label = ex[label_column]

    label_mapping = template_spec.label_mapping
    numeric_keys = isinstance(next(iter(template_spec.label_mapping.keys())), int)
    if numeric_keys:
        if isinstance(raw_label, str):
            try:
                numeric_key = int(raw_label)
            except ValueError as err:
                str_to_int_key_map = {
                    str(v).lower(): int(k) for k, v in label_mapping.items()
                }
                mapped_key = str_to_int_key_map.get(raw_label.lower())
                if mapped_key is None:
                    raise ValueError("Cannot map label to integer key") from err
                numeric_key = mapped_key
        else:
            numeric_key = int(raw_label)

        if numeric_key not in label_mapping and (numeric_key - 1) in label_mapping:
            numeric_key -= 1
        assistant_response = label_mapping[numeric_key]
    else:
        string_key = str(raw_label)
        assistant_response = label_mapping[string_key]

    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]
