from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from sallm.training.trainer import CustomSFTTrainer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from trl import SFTConfig


def test_custom_sft_trainer_saves_model_and_processing_class(tmp_path: Path) -> None:
    backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "hello": 3},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
        )
    )
    trainer = CustomSFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(tmp_path / "trainer"),
            report_to=[],
            use_cpu=True,
            bf16=False,
            fp16=False,
        ),
        train_dataset=Dataset.from_dict({"text": ["hello"]}),
        processing_class=tokenizer,
    )
    output_dir = tmp_path / "saved"

    trainer.save_model(output_dir)

    assert (output_dir / "config.json").is_file()
    assert (output_dir / "model.safetensors").is_file()
    assert (output_dir / "tokenizer.json").is_file()
    loaded = PreTrainedTokenizerFast.from_pretrained(
        output_dir,
        local_files_only=True,
    )
    assert loaded.convert_tokens_to_ids("hello") == 3
