import logging
from typing import Optional, Tuple

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    Trainer,
    TrainingArguments,
)

from sallm.config import ExperimentConfig
from sallm.utils import compute_metrics, count_trainable_parameters

logger = logging.getLogger(__name__)


def build_tokenizer(config: ExperimentConfig) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(config.tokenizer.path)


def build_model(
    config: ExperimentConfig, tokenizer: AutoTokenizer
) -> AutoModelForCausalLM:
    """
    Builds a model from a configuration object and validates its parameter count.
    """
    model_conf = config.model

    if model_conf.architecture == "llama":
        model_config_obj = LlamaConfig(**model_conf.config)
    else:
        raise ValueError(f"Unsupported model architecture: {model_conf.architecture}")

    model_config_obj.vocab_size = len(tokenizer)
    model = AutoModelForCausalLM.from_config(model_config_obj)

    if model_conf.param_validation:
        num_params = count_trainable_parameters(model)
        num_params_m = num_params / 1_000_000

        min_p = model_conf.param_validation.min_params_m
        max_p = model_conf.param_validation.max_params_m

        logger.info(f"Validating model size: {num_params_m:.2f}M parameters.")

        if not (min_p <= num_params_m <= max_p):
            raise ValueError(
                f"Model size validation failed! "
                f"Expected between {min_p}M and {max_p}M parameters, "
                f"but got {num_params_m:.2f}M."
            )
        logger.info("Model size validation passed.")

    return model


def build_datasets(
    config: ExperimentConfig, is_hpo: bool
) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    """
    Loads a single DatasetDict and extracts the specified splits.
    If in HPO mode, the test set is explicitly not loaded to prevent data leakage.
    """
    data_conf = config.data

    # TODO: from huggingface rather?
    dataset_dict = load_from_disk(data_conf.path)

    if not isinstance(dataset_dict, DatasetDict):
        raise TypeError(
            f"Expected data at {data_conf.path} to be a DatasetDict, but found {type(dataset_dict)}"
        )

    train_dataset = dataset_dict[data_conf.train_split]
    eval_dataset = dataset_dict[data_conf.eval_split]

    test_dataset = None
    if not is_hpo and data_conf.test_split and data_conf.test_split in dataset_dict:
        test_dataset = dataset_dict[data_conf.test_split]

    return train_dataset, eval_dataset, test_dataset


def build_trainer(
    config: ExperimentConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> Trainer:
    """Builds and returns a Hugging Face Trainer instance."""
    training_args = TrainingArguments(**config.training.model_dump())

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
