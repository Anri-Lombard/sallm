import datasets
import pytest
from sallm.training.trainer import CustomSFTTrainer


def _trainer(general_selection: bool) -> CustomSFTTrainer:
    trainer = object.__new__(CustomSFTTrainer)
    trainer.general_selection = general_selection
    trainer.eval_dataset = datasets.Dataset.from_dict({"text": ["x"]})
    trainer._evaluate_general_selection = lambda dataset, prefix: {"routed": prefix}
    return trainer


def test_general_selection_routes_evaluation_to_general_nll() -> None:
    assert _trainer(True).evaluate(metric_key_prefix="eval") == {"routed": "eval"}


def test_general_selection_rejects_streaming_datasets() -> None:
    trainer = _trainer(True)
    trainer.eval_dataset = [{"text": "x"}]

    with pytest.raises(TypeError, match="Hugging Face Dataset"):
        trainer.evaluate()
