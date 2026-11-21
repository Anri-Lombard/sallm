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


_PREFIX_PROBE_SENTINEL = "__SALLM_ASSISTANT_CONTENT_PROBE__"


def assistant_content_prefix(
    tokenizer: Any,
    *,
    chat_template: str | None = None,
) -> str:
    """Return any literal whitespace the chat template inserts between the
    generation prompt and an assistant message's content.

    ``CANONICAL_CHAT_TEMPLATE`` renders ``add_generation_prompt=True`` as
    ``<|assistant|>``, while training assistant turns render
    ``<|assistant|>\\n        {{ content }}``. Scoring classification labels
    right after ``<|assistant|>`` without that prefix is off-distribution (#142).
    """
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_fn):
        return ""
    template_kwargs: dict[str, Any] = {}
    if chat_template is not None:
        template_kwargs["chat_template"] = chat_template
    try:
        prompt = apply_fn(
            [{"role": "user", "content": "u"}],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        full = apply_fn(
            [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": _PREFIX_PROBE_SENTINEL},
            ],
            tokenize=False,
            add_generation_prompt=False,
            **template_kwargs,
        )
    except Exception:
        return ""
    if not (isinstance(prompt, str) and isinstance(full, str)):
        return ""
    if not full.startswith(prompt):
        return ""
    between = full[len(prompt) :]
    probe_idx = between.find(_PREFIX_PROBE_SENTINEL)
    if probe_idx <= 0:
        return ""
    prefix = between[:probe_idx]
    return prefix if prefix.isspace() else ""
