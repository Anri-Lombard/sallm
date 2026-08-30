from huggingface_hub.dataclasses import strict
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import LlamaConfig as TransformersLlamaConfig
from transformers import LlamaForCausalLM as TransformersLlamaForCausalLM

_LEGACY_LLAMA_125M_SHAPE = (512, 9, 3, 56)


@strict
class LlamaConfig(TransformersLlamaConfig):
    """LLaMA config that preserves SALLM's frozen 125M projection shapes."""

    def validate_architecture(self) -> None:
        shape = (
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        if shape == _LEGACY_LLAMA_125M_SHAPE:
            return
        super().validate_architecture()


class LlamaForCausalLM(TransformersLlamaForCausalLM):
    config_class = LlamaConfig


def register_llama_compatibility() -> None:
    """Use the compatible classes in Transformers AutoModel load paths."""
    AutoConfig.register(LlamaConfig.model_type, LlamaConfig, exist_ok=True)
    AutoModelForCausalLM.register(
        LlamaConfig,
        LlamaForCausalLM,
        exist_ok=True,
    )
