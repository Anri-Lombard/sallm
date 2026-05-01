from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def to_resolved_dict(value: Any, *, name: str = "config") -> dict[str, Any]:
    """Resolve an OmegaConf or plain mapping into a string-keyed dict."""
    if isinstance(value, DictConfig):
        resolved = OmegaConf.to_container(value, resolve=True)
    elif isinstance(value, dict):
        resolved = value
    else:
        raise TypeError(f"{name} must be a mapping, got {type(value)!r}.")

    if not isinstance(resolved, dict):
        raise TypeError(f"{name} must resolve to a mapping.")
    return {str(key): item for key, item in resolved.items()}
