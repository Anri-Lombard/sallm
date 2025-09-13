from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Mamba2Config,
    Mamba2ForCausalLM,
    # xLSTMConfig,
    # xLSTMForCausalLM,
    PretrainedConfig,
)

# TODO add mamba
# TODO consider mixtral?
MODEL_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = {
    "llama": LlamaConfig,
    "mamba2": Mamba2Config,
    # "xlstm": xLSTMConfig,
}

MODEL_CLASS_REGISTRY: dict[str, type[PretrainedConfig]] = {
    "llama": LlamaForCausalLM,
    "mamba2": Mamba2ForCausalLM,
    # "xlstm": xLSTMForCausalLM,
}
