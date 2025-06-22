import logging

from transformers import AutoModelForCausalLM, AutoTokenizer

from sallm.config import ExperimentConfig
from sallm.models.registry import MODEL_CONFIG_REGISTRY
from sallm.utils import count_trainable_parameters

logger = logging.getLogger(__name__)


def build_tokenizer(config: ExperimentConfig) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(config.tokenizer.path)


def build_model(
    config: ExperimentConfig, tokenizer: AutoTokenizer
) -> AutoModelForCausalLM:
    """
    Builds a model from a configuration object by looking up the
    architecture in the central registry.
    """
    model_conf = config.model

    config_class = MODEL_CONFIG_REGISTRY[model_conf.architecture]
    model_config_obj = config_class(**model_conf.config)
    model_config_obj.vocab_size = len(tokenizer)

    if hasattr(model_conf, "init_checkpoint") and model_conf.init_checkpoint:
        model = AutoModelForCausalLM.from_pretrained(model_conf.init_checkpoint)
        return model

    model = AutoModelForCausalLM.from_config(
        model_config_obj, attn_implementation="flash_attention_2"
    )

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
