from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Mamba2Config,
    Mamba2ForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)

# TODO add mamba
# TODO consider mixtral?
MODEL_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = {
    "llama": LlamaConfig,
    "mamba2": Mamba2Config,
    # "xlstm": xLSTMConfig,
}

MODEL_CLASS_REGISTRY: dict[str, type[PreTrainedModel]] = {
    "llama": LlamaForCausalLM,
    "mamba2": Mamba2ForCausalLM,
}
