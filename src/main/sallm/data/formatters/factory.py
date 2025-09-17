from __future__ import annotations

from datasets import Dataset

from sallm.config import FinetuneDatasetConfig, FinetuneTaskType

from .base import TaskFormatter
from .classification import ClassificationFormatter
from .instruction import InstructionFormatter
from .ner import NamedEntityFormatter
from .pos import POSTaggingFormatter


def build_formatter(
    task: FinetuneTaskType, dataset: Dataset, config: FinetuneDatasetConfig
) -> TaskFormatter:
    if task == FinetuneTaskType.INSTRUCTION:
        formatter = InstructionFormatter()
    elif task == FinetuneTaskType.CLASSIFICATION:
        formatter = ClassificationFormatter()
    elif task == FinetuneTaskType.NAMED_ENTITY_RECOGNITION:
        formatter = NamedEntityFormatter()
    elif task == FinetuneTaskType.POS_TAGGING:
        formatter = POSTaggingFormatter(dataset)
    else:
        raise ValueError(f"Unsupported task type: {task}")
    formatter.validate_dataset(dataset, config)
    return formatter
