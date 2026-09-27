"""Canonical SALLM chat-template contract shared by train and evaluation."""

from __future__ import annotations

from typing import Any

CANONICAL_CHAT_TEMPLATE = """{{- bos_token -}}
{%- if system_message %}
<|system|>
{{ system_message }}{{ eos_token }}
{%- endif %}
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        <|user|>
        {{ message['content'] }}{{ eos_token }}
    {%- elif message['role'] == 'assistant' %}
        {%- generation -%}
        <|assistant|>
        {{ message['content'] }}{{ eos_token }}
        {%- endgeneration -%}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|assistant|>{%- endif %}
"""


def install_canonical_chat_template(tokenizer: Any, *, force: bool = False) -> bool:
    """Install the canonical template when absent or explicitly required.

    Existing model-specific templates are preserved unless ``force`` is true.
    The return value records whether this call installed the template.
    """

    if getattr(tokenizer, "chat_template", None) and not force:
        return False
    try:
        tokenizer.chat_template = CANONICAL_CHAT_TEMPLATE
    except Exception as exc:  # pragma: no cover - tokenizer implementation guard
        raise ValueError("Unable to install the canonical chat template.") from exc
    if getattr(tokenizer, "chat_template", None) != CANONICAL_CHAT_TEMPLATE:
        raise ValueError("Tokenizer rejected the canonical chat template.")
    return True


def fallback_chat_template(tokenizer: Any) -> str | None:
    """Return the canonical fallback only when the tokenizer needs one."""

    if getattr(tokenizer, "chat_template", None):
        return None
    return CANONICAL_CHAT_TEMPLATE
