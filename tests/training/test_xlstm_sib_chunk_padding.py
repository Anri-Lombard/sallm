from pathlib import Path

import yaml
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


def test_xlstm_sib_padding_preserves_assistant_loss_mask() -> None:
    config_path = Path(__file__).parents[2] / "src/conf/finetune/xlstm_sib_all.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["training"]["pad_to_multiple_of"] == 64

    collator = DataCollatorForLanguageModeling(
        pad_token_id=2,
        completion_only_loss=True,
        padding_free=False,
        pad_to_multiple_of=64,
    )
    long_ids = list(range(277))
    short_ids = list(range(193))
    examples = [
        {
            "input_ids": long_ids,
            "assistant_masks": [0] * 270 + [1] * 7,
        },
        {
            "input_ids": short_ids,
            "assistant_masks": [0] * 187 + [1] * 6,
        },
    ]

    batch = collator(examples)
    assert tuple(batch["input_ids"].shape) == (2, 320)
    assert tuple(batch["attention_mask"].shape) == (2, 320)
    assert tuple(batch["labels"].shape) == (2, 320)
    assert bool((batch["input_ids"][:, 277:] == 2).all())
    assert bool((batch["attention_mask"][0, :277] == 1).all())
    assert bool((batch["attention_mask"][0, 277:] == 0).all())
    assert bool((batch["labels"][0, :270] == -100).all())
    assert batch["labels"][0, 270:277].tolist() == long_ids[270:277]
    assert bool((batch["labels"][0, 277:] == -100).all())
    assert bool((batch["labels"][1, :187] == -100).all())
    assert batch["labels"][1, 187:193].tolist() == short_ids[187:193]
    assert bool((batch["labels"][1, 193:] == -100).all())
