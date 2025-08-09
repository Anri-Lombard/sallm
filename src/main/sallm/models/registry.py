from typing import Dict, Type

from transformers import (
    LlamaConfig,
    PretrainedConfig,
    Mamba2Config,
    LlamaForCausalLM,
    Mamba2ForCausalLM,
)

# TODO add mamba
# TODO consider mixtral?
MODEL_CONFIG_REGISTRY: Dict[str, Type[PretrainedConfig]] = {
    "llama": LlamaConfig,
    "mamba2": Mamba2Config,
}

MODEL_CLASS_REGISTRY: Dict[str, Type[PretrainedConfig]] = {
    "llama": LlamaForCausalLM,
    "mamba2": Mamba2ForCausalLM,
}
