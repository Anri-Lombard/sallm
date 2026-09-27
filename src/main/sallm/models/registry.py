from importlib import import_module

from sallm.models.llama_compatibility import LlamaConfig, LlamaForCausalLM

type ModuleClassTarget = str | tuple[str, str]


_PURE_GDN_INSTALL_MESSAGE = (
    "Pure GatedDeltaNet (`gated_deltanet`) requires "
    "`flash-linear-attention==0.5.1`. On Linux, install it with "
    "`uv sync --extra pure-gdn --frozen --inexact`."
)


class LazyRegistry(dict):
    """Dict that lazily imports configured classes on first access."""

    def __init__(self, mappings: dict[str, ModuleClassTarget]):
        """Map architecture keys to a class or explicit ``(module, class)`` target."""
        super().__init__()
        self._mappings = mappings

    def __getitem__(self, key: str):
        if key not in dict.keys(self):
            if key not in self._mappings:
                raise KeyError(key)
            target = self._mappings[key]
            module_name, class_name = (
                target if isinstance(target, tuple) else ("transformers", target)
            )
            try:
                module = import_module(module_name)
            except ModuleNotFoundError as exc:
                if module_name.startswith("fla.") and exc.name == "fla":
                    raise ModuleNotFoundError(_PURE_GDN_INSTALL_MESSAGE) from exc
                raise
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
        "gated_deltanet": (
            "fla.models.gated_deltanet",
            "GatedDeltaNetConfig",
        ),
        "qwen3next_gdn_hybrid": "Qwen3NextConfig",
        "mamba2": "Mamba2Config",
        "recurrent_gemma": "RecurrentGemmaConfig",
        "rwkv": "RwkvConfig",
        "xlstm": "xLSTMConfig",
    }
)

MODEL_CLASS_REGISTRY = LazyRegistry(
    {
        "llama": "LlamaForCausalLM",
        "gated_deltanet": (
            "fla.models.gated_deltanet",
            "GatedDeltaNetForCausalLM",
        ),
        "qwen3next_gdn_hybrid": "Qwen3NextForCausalLM",
        "mamba2": "Mamba2ForCausalLM",
        "recurrent_gemma": "RecurrentGemmaForCausalLM",
        "rwkv": "RwkvForCausalLM",
        "xlstm": "xLSTMForCausalLM",
    }
)

dict.__setitem__(MODEL_CONFIG_REGISTRY, "llama", LlamaConfig)
dict.__setitem__(MODEL_CLASS_REGISTRY, "llama", LlamaForCausalLM)
