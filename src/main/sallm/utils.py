import logging
from enum import Enum

import numpy as np
import torch
from torch import nn
from transformers import EvalPrediction

logger = logging.getLogger(__name__)


class RunMode(str, Enum):
    TRAIN = "train"
    FINETUNE = "finetune"
    EVALUATE = "evaluate"


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

