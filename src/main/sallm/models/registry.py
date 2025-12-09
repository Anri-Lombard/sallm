from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Mamba2Config,
    Mamba2ForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    RecurrentGemmaConfig,
    RecurrentGemmaForCausalLM,
    RwkvConfig,
    RwkvForCausalLM,
    xLSTMConfig,
    xLSTMForCausalLM,
)

MODEL_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = {
    "llama": LlamaConfig,
    "mamba2": Mamba2Config,
    "recurrent_gemma": RecurrentGemmaConfig,
    "rwkv": RwkvConfig,
    "xlstm": xLSTMConfig,
}

MODEL_CLASS_REGISTRY: dict[str, type[PreTrainedModel]] = {
    "llama": LlamaForCausalLM,
    "mamba2": Mamba2ForCausalLM,
    "recurrent_gemma": RecurrentGemmaForCausalLM,
    "rwkv": RwkvForCausalLM,
    "xlstm": xLSTMForCausalLM,
}
