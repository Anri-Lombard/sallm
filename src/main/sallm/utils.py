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


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    logits, labels = eval_pred
    shifted_logits = logits[..., :-1, :]
    shifted_labels = labels[..., 1:]

    predictions = np.argmax(shifted_logits, axis=-1)
    mask = shifted_labels != -100
    correct_predictions = (predictions == shifted_labels)[mask].sum()
    total_predictions = mask.sum()
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    loss = 0.0
    perplexity = 0.0

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c_logits = (
        torch.tensor(shifted_logits).to(device).view(-1, shifted_logits.shape[-1])
    )
    c_labels = torch.tensor(shifted_labels).to(device).view(-1)
    with torch.no_grad():
        loss = loss_fct(c_logits, c_labels).item()

    perplexity = np.exp(loss)

    return {"accuracy": accuracy, "perplexity": perplexity, "loss": loss}
