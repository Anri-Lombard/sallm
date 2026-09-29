import pytest
from sallm.config import (
    ExperimentConfig,
    FinetuneDatasetConfig,
    FinetuneTaskType,
    HubConfig,
    ModelConfig,
)
from sallm.fine_tune.run import (
    _build_hub_repo_id,
    _configure_task_truncation,
)
from sallm.utils import RunMode


def _experiment(*, repo_id: str) -> ExperimentConfig:
    return ExperimentConfig(
        mode=RunMode.FINETUNE,
        wandb=None,
        model=ModelConfig(architecture="llama", config={}),
        dataset=FinetuneDatasetConfig(
            hf_name="mix:sa_multitask",
            max_seq_length=1024,
            packing=False,
            assistant_only_loss=True,
            mix_weights={"sib": 1.0},
        ),
        hub=HubConfig(repo_id=repo_id),
    )


def test_build_hub_repo_id_uses_explicit_id() -> None:
    config = _experiment(repo_id="owner/experiment")

    assert _build_hub_repo_id(config) == "owner/experiment"


def test_build_hub_repo_id_rejects_invalid_explicit_id() -> None:
    config = _experiment(repo_id="experiment")

    with pytest.raises(ValueError, match="owner/model"):
        _build_hub_repo_id(config)


def test_build_hub_repo_id_derives_a_safe_name_from_the_run() -> None:
    config = ExperimentConfig(
        mode=RunMode.FINETUNE,
        wandb=None,
        model=ModelConfig(architecture="mamba2", config={}),
        dataset=FinetuneDatasetConfig(
            hf_name="github:dadelani/AfriHG",
            languages=["xho", "zul"],
            max_seq_length=1024,
            packing=False,
            assistant_only_loss=True,
        ),
        hub=HubConfig(organization="owner"),
    )

    assert (
        _build_hub_repo_id(config)
        == "owner/sallm-mamba2-github-dadelani-afrihg-xho-zul"
    )


@pytest.mark.parametrize(
    ("task_type", "expected"),
    [
        (FinetuneTaskType.CLASSIFICATION, "left"),
        (FinetuneTaskType.INSTRUCTION, "left"),
        (None, "right"),
    ],
)
def test_configure_task_truncation(
    task_type: FinetuneTaskType | None,
    expected: str,
) -> None:
    tokenizer = type("Tokenizer", (), {"truncation_side": "right"})()

    _configure_task_truncation(tokenizer=tokenizer, task_type=task_type)

    assert tokenizer.truncation_side == expected
