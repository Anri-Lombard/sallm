from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    Mamba2Config,
    Mamba2ForCausalLM,
    PretrainedConfig,
)

MODEL_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = {
    "llama": LlamaConfig,
    "mamba2": Mamba2Config,
}

MODEL_CLASS_REGISTRY: dict[str, type[AutoModelForCausalLM]] = {
    "llama": LlamaForCausalLM,
    "mamba2": Mamba2ForCausalLM,
}
