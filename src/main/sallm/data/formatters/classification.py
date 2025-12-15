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

    numeric_keys = isinstance(next(iter(template_spec.label_mapping.keys())), int)
    if numeric_keys:
        if isinstance(raw_label, str):
            try:
                key_to_use = int(raw_label)
            except ValueError as err:
                str_to_int_key_map = {
                    str(v).lower(): k for k, v in template_spec.label_mapping.items()
                }
                key_to_use = str_to_int_key_map.get(raw_label.lower())
                if key_to_use is None:
                    raise ValueError("Cannot map label to integer key") from err
        else:
            key_to_use = int(raw_label)

        if (
            key_to_use not in template_spec.label_mapping
            and (key_to_use - 1) in template_spec.label_mapping
        ):
            key_to_use -= 1
    else:
        key_to_use = str(raw_label)

    assistant_response = template_spec.label_mapping[key_to_use]
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]
