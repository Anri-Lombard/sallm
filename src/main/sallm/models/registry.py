from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Mamba2Config,
    Mamba2ForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)

MODEL_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = {
    "llama": LlamaConfig,
    "mamba2": Mamba2Config,
}

MODEL_CLASS_REGISTRY: dict[str, type[PreTrainedModel]] = {
    "llama": LlamaForCausalLM,
    "mamba2": Mamba2ForCausalLM,
}
