import numpy as np
from transformers import EvalPrediction
from enum import Enum
import logging
from torch import nn

logger = logging.getLogger(__name__)


class RunMode(str, Enum):
    TRAIN = "train"
    FINETUNE = "finetune"
    EVALUATE = "evaluate"


def count_trainable_parameters(model: nn.Module) -> int:
    """Counts the number of parameters in a model that require gradients."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """
    Computes next-token prediction accuracy for a Causal LM.
    """
    logits, labels = eval_pred
    shifted_logits = logits[..., :-1, :]
    shifted_labels = labels[..., 1:]

    predictions = np.argmax(shifted_logits, axis=-1)
    mask = shifted_labels != -100

    correct_predictions = (predictions == shifted_labels)[mask].sum()
    total_predictions = mask.sum()

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    return {"accuracy": accuracy}
