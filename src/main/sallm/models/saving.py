import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def sync_weight_tying_flag(model: Any) -> None:
    """Make config.tie_word_embeddings match whether the weights are shared."""
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "tie_word_embeddings"):
        return
    get_input = getattr(model, "get_input_embeddings", None)
    get_output = getattr(model, "get_output_embeddings", None)
    if not callable(get_input) or not callable(get_output):
        return
    input_emb = get_input()
    output_emb = get_output()
    if input_emb is None or output_emb is None:
        return
    input_weight = getattr(input_emb, "weight", None)
    output_weight = getattr(output_emb, "weight", None)
    if input_weight is None or output_weight is None:
        return
    shared = input_weight is output_weight
    if not shared:
        shared = input_weight.data_ptr() == output_weight.data_ptr()
    if shared and not bool(config.tie_word_embeddings):
        logger.info(
            "Detected tied embeddings; updating config.tie_word_embeddings to "
            "True before saving."
        )
        config.tie_word_embeddings = True
    if not shared and bool(config.tie_word_embeddings):
        logger.info(
            "Detected untied embeddings; updating config.tie_word_embeddings to "
            "False before saving."
        )
        config.tie_word_embeddings = False


def save_pretrained(
    model: Any, output_dir: str | Path, *, safe_serialization: bool = True
) -> None:
    """Save, falling back to torch serialization when safetensors refuses
    shared tensors."""
    try:
        model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    except RuntimeError as exc:
        if not (safe_serialization and "shared tensors" in str(exc)):
            raise
        logger.warning(
            "Retrying save_pretrained with safe_serialization=False due to "
            "shared tensors."
        )
        model.save_pretrained(output_dir, safe_serialization=False)
