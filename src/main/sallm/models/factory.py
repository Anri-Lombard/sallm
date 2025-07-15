import logging

from transformers import AutoModelForCausalLM, AutoTokenizer

from sallm.config import ExperimentConfig
from sallm.models.registry import MODEL_CONFIG_REGISTRY, MODEL_CLASS_REGISTRY
from sallm.utils import count_trainable_parameters

logger = logging.getLogger(__name__)


def build_tokenizer(config: ExperimentConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.path)

    if tokenizer.chat_template is None:
        tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\\n' + message['content'] + '<|end|>\\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '<|end|>\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\\n' + message['content'] + '<|end|>\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% else %}{{ eos_token }}{% endif %}"""
        logger.info("Tokenizer chat template not found. Applying custom template.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(
            f"tokenizer.pad_token was not set, setting it to eos_token: {tokenizer.eos_token}"
        )

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
            f"Loading model of type {model_class.__name__} from checkpoint: {model_conf.init_checkpoint}"
        )
        attn_impl = getattr(config.model, "attn_implementation", None)
        model = model_class.from_pretrained(
            model_conf.init_checkpoint,
            attn_implementation=attn_impl,
        )
        return model

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
