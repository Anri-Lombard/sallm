from typing import Dict, Type

from transformers import LlamaConfig, PretrainedConfig

# TODO add mamba
# TODO consider mixtral?
MODEL_CONFIG_REGISTRY: Dict[str, Type[PretrainedConfig]] = {
    "llama": LlamaConfig,
}
