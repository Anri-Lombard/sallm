from importlib import import_module


class LazyRegistry(dict):
    """Dict that lazily imports classes from transformers on first access."""

    def __init__(self, mappings: dict[str, str]):
        """mappings: {architecture: class_name} where class_name is in transformers."""
        super().__init__()
        self._mappings = mappings

    def __getitem__(self, key: str):
        if key not in dict.keys(self):
            if key not in self._mappings:
                raise KeyError(key)
            class_name = self._mappings[key]
            module = import_module("transformers")
            dict.__setitem__(self, key, getattr(module, class_name))
        return dict.__getitem__(self, key)

    def __contains__(self, key: object) -> bool:
        return key in self._mappings

    def get(self, key: str, default=None):
        if key not in self._mappings:
            return default
        return self[key]


MODEL_CONFIG_REGISTRY = LazyRegistry(
    {
        "llama": "LlamaConfig",
        "mamba2": "Mamba2Config",
        "recurrent_gemma": "RecurrentGemmaConfig",
        "rwkv": "RwkvConfig",
        "xlstm": "xLSTMConfig",
    }
)

MODEL_CLASS_REGISTRY = LazyRegistry(
    {
        "llama": "LlamaForCausalLM",
        "mamba2": "Mamba2ForCausalLM",
        "recurrent_gemma": "RecurrentGemmaForCausalLM",
        "rwkv": "RwkvForCausalLM",
        "xlstm": "xLSTMForCausalLM",
    }
)
