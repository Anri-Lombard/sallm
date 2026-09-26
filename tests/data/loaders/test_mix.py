from pathlib import Path

import yaml
from datasets import Dataset
from sallm.config import (
    FinetuneDatasetConfig,
    FinetuneTaskType,
    TemplateChoice,
    TemplateRef,
)
from sallm.data.loaders import mix


def test_attach_task_name_is_explicit_and_conflict_safe() -> None:
    dataset = Dataset.from_list([{"messages": []}, {"messages": []}])

    attached = mix._attach_task_name(dataset, "afrihg")

    assert attached["task_name"] == ["afrihg", "afrihg"]


def test_pos_general_selection_uses_only_four_frozen_prompts(monkeypatch) -> None:
    raw = Dataset.from_list([{"tokens": ["Molo"], "upos": ["INTJ"], "lang": "xho"}])
    monkeypatch.setattr(mix, "_load_component_raw", lambda _config: (raw, raw))
    all_templates = [
        TemplateRef(id=f"masakhane_pos_tagging/lm_eval_p{index}")
        for index in range(1, 6)
    ]
    eval_templates = all_templates[:4]
    config = FinetuneDatasetConfig(
        hf_name="masakhane/masakhapos",
        languages=["xho"],
        task=FinetuneTaskType.POS_TAGGING,
        splits={"train": "train", "val": "validation"},
        templates=all_templates,
        template_choice=TemplateChoice.CYCLE,
        max_seq_length=2048,
        packing=False,
        assistant_only_loss=True,
    )

    train, validation = mix._process_component(config, eval_templates=eval_templates)

    assert len(train) == 1
    assert len(validation) == 4
    assert set(validation["template_id"]) == {
        template.id for template in eval_templates
    }


def test_general_mix_preserves_six_training_families_and_four_pos_prompts() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "src/main/sallm/data/mixes/sa_general.yaml"
    )
    components = yaml.safe_load(path.read_text(encoding="utf-8"))["components"]

    assert {component["name"] for component in components} == {
        "news",
        "ner",
        "pos",
        "sib",
        "t2x",
        "afrihg",
    }
    pos = next(component for component in components if component["name"] == "pos")
    assert [template["id"] for template in pos["templates"]] == [
        f"masakhane_pos_tagging/lm_eval_p{index}" for index in range(1, 5)
    ]
