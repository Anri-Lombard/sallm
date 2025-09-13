import logging

from sallm.config import ExperimentConfig
from sallm.models.registry import MODEL_CLASS_REGISTRY, MODEL_CONFIG_REGISTRY
from sallm.utils import count_trainable_parameters
from tokenizers.decoders import ByteLevel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def build_tokenizer(config: ExperimentConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.path)
    tokenizer.backend_tokenizer.decoder = ByteLevel()
    return tokenizer


def build_model(
    config: ExperimentConfig, tokenizer: AutoTokenizer
) -> AutoModelForCausalLM:
    model_conf = config.model
    model_class = MODEL_CLASS_REGISTRY.get(model_conf.architecture)

    if not model_class:
        raise ValueError(f"Unsupported model architecture: {model_conf.architecture}")

    if getattr(model_conf, "init_checkpoint", None):
        logger.info(
            "Loading model of type '"
            f"{model_class.__name__}"
            "' from checkpoint: "
            f"{model_conf.init_checkpoint}"
        )
        attn_impl = getattr(model_conf, "attn_implementation", None)
        if attn_impl:
            model = model_class.from_pretrained(
                model_conf.init_checkpoint, attn_implementation=attn_impl
            )
        else:
            model = model_class.from_pretrained(model_conf.init_checkpoint)

    config_class = MODEL_CONFIG_REGISTRY[model_conf.architecture]
    if model_conf.config is None:
        raise ValueError(
            "`model.config` is required when `init_checkpoint` is not provided."
        )

    model_config_obj = config_class(**model_conf.config)
    model_config_obj.vocab_size = len(tokenizer)

    model = model_class(model_config_obj)

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
