"""Small runtime utilities used across the codebase.

This module contains lightweight helper utilities and enums such as the
RunMode enum used to select the high-level pipeline behavior and a helper to
count trainable parameters on a PyTorch model.
"""

import logging
from enum import Enum

from torch import nn

logger = logging.getLogger(__name__)


class RunMode(str, Enum):
    """High-level pipeline run modes for selecting the workflow."""


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters on the given model."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
