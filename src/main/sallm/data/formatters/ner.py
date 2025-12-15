from __future__ import annotations

from typing import Any

from sallm.data.formatters.base import safe_format_prompt
from sallm.templates import registry as tmpl


def reconstruct_entities_from_iob(
    tokens: list[str], ner_tag_ids: list[int], tag_map: list[str]
) -> list[str]:
    """Reconstruct named entities from IOB-tagged tokens.

    Args:
        tokens: List of tokens
        ner_tag_ids: List of tag IDs for each token
        tag_map: Mapping from tag ID to tag name (e.g., ["O", "B-PER", "I-PER"])

    Returns:
        List of entity strings in "LABEL: text" format
    """
    entities = []
    current_entity_tokens = []
    current_entity_label = None

    for token, tag_id in zip(tokens, ner_tag_ids, strict=False):
        tag_name = tag_map[tag_id]

        if tag_name.startswith("B-"):
            if current_entity_tokens:
                entity_text = " ".join(current_entity_tokens)
                entities.append(f"{current_entity_label}: {entity_text}")

            current_entity_tokens = [token]
            current_entity_label = tag_name[2:]
        elif tag_name.startswith("I-"):
            if current_entity_label == tag_name[2:]:
                current_entity_tokens.append(token)
            else:
                if current_entity_tokens:
                    entity_text = " ".join(current_entity_tokens)
                    entities.append(f"{current_entity_label}: {entity_text}")
                current_entity_tokens = []
                current_entity_label = None
        else:  # O tag
            if current_entity_tokens:
                entity_text = " ".join(current_entity_tokens)
                entities.append(f"{current_entity_label}: {entity_text}")
            current_entity_tokens = []
            current_entity_label = None

    if current_entity_tokens:
        entity_text = " ".join(current_entity_tokens)
        entities.append(f"{current_entity_label}: {entity_text}")

    return entities


def format_ner(ex: dict[str, Any], template_id: str) -> list[dict[str, str]]:
    """Format a NER example as messages.

    Args:
        ex: Dataset example with 'tokens' and 'ner_tags'
        template_id: Template ID to use

    Returns:
        List of message dicts with 'role' and 'content'
    """
    template_spec = tmpl.get(template_id)
    tag_map = template_spec.ner_tags
    if not tag_map:
        raise ValueError(
            f"Template '{template_spec.id}' is for an NER task "
            "but is missing the 'ner_tags' list."
        )

    text_input = " ".join(ex["tokens"])
    user_prompt = safe_format_prompt(template_spec.prompt, {"text": text_input})
    reconstructed_entities = reconstruct_entities_from_iob(
        ex["tokens"], ex["ner_tags"], tag_map
    )
    assistant_response = " $$ ".join(reconstructed_entities)

    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]
