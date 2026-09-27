from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import yaml
from datasets import Dataset
from sallm.config import FinetuneTaskType, RunMode
from sallm.evaluation.classification_metrics import ClassificationEvaluator
from sallm.training import factory as training_factory

MULTILINGUAL_CLASSIFICATION_FAMILIES = (
    "injongointent_all",
    "news_all",
    "sib_all",
)


@pytest.mark.parametrize("architecture", ("llama", "mamba", "xlstm"))
@pytest.mark.parametrize("family", MULTILINGUAL_CLASSIFICATION_FAMILIES)
def test_multilingual_classification_sweeps_select_macro_f1(
    architecture, family
) -> None:
    path = f"src/conf/sweeps/{architecture}_{family}.yaml"
    with open(path, encoding="utf-8") as handle:
        sweep = yaml.safe_load(handle)

    assert sweep["metric"] == {
        "name": "classification/all_macro_f1",
        "goal": "maximize",
    }


@pytest.mark.parametrize("family", MULTILINGUAL_CLASSIFICATION_FAMILIES)
def test_multilingual_classification_checkpoints_select_macro_f1(family) -> None:
    path = f"src/conf/finetune/mamba_{family}.yaml"
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["training"]["metric_for_best_model"] == (
        "eval_classification/all_macro_f1"
    )


def test_classification_factory_defaults_to_macro_f1(monkeypatch, tmp_path) -> None:
    class StubTrainer:
        def __init__(self, *, args, **_kwargs) -> None:
            self.args = args
            self.processing_class = None

    monkeypatch.setattr(training_factory, "CustomSFTTrainer", StubTrainer)

    config = SimpleNamespace(
        mode=RunMode.FINETUNE,
        training={
            "early_stopping_patience": 2,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "output_dir": str(tmp_path),
            "report_to": [],
            "use_cpu": True,
            "bf16": False,
            "fp16": False,
        },
        dataset=SimpleNamespace(
            task=FinetuneTaskType.CLASSIFICATION,
            max_seq_length=128,
            packing=False,
            assistant_only_loss=True,
        ),
        generation_decoding=None,
    )
    dataset = Dataset.from_list([{"lang": "xho", "messages": []}])

    trainer = training_factory.build_trainer(
        config=config,
        model=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    assert trainer.args.metric_for_best_model == ("eval_classification/all_macro_f1")
    assert trainer.args.greater_is_better is True


def test_multilingual_selection_metric_is_mean_language_macro_f1(monkeypatch) -> None:
    evaluator = ClassificationEvaluator.__new__(ClassificationEvaluator)
    evaluator.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    evaluator.max_samples_per_lang = None
    monkeypatch.setattr(evaluator, "_get_fallback_template", lambda: None)
    monkeypatch.setattr(evaluator, "_cap_dataset", lambda dataset, _lang: dataset)

    def fake_subset(_model, dataset, *_args):
        lang = dataset[0]["lang"]
        return {
            "f1": 0.8 if lang == "a" else 0.6,
            "macro_f1": 0.3 if lang == "a" else 0.1,
        }

    monkeypatch.setattr(evaluator, "_evaluate_subset", fake_subset)
    model = MagicMock(device=torch.device("cpu"))
    dataset = Dataset.from_list(
        [
            {"lang": "a", "messages": []},
            {"lang": "b", "messages": []},
        ]
    )

    metrics = evaluator.evaluate(model, dataset)

    assert metrics["classification/all_f1"] == 0.7
    assert metrics["classification/all_macro_f1"] == 0.2
