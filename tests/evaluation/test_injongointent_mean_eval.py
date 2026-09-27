import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

from sallm.evaluation.classification_metrics import ClassificationEvaluator

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "run_injongointent_mean_eval.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = spec_from_file_location("run_injongointent_mean_eval", SCRIPT)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_single_language_config_uses_dataset_subset() -> None:
    config = SimpleNamespace(languages=None, subset="xho")

    assert MODULE.resolve_languages(config) == ["xho"]


def test_f1_selects_headline_when_accuracy_winner_differs() -> None:
    gold = ["a"] * 8 + ["b", "c"]
    accuracy_winner = ["a"] * 10
    f1_winner = ["a"] * 5 + ["b", "b", "c", "b", "c"]
    rows = [
        {
            "lang": "eng",
            "template_id": prompt,
            "gold": reference,
            "prediction": prediction,
        }
        for prompt, predictions in (
            ("accuracy_prompt", accuracy_winner),
            ("f1_prompt", f1_winner),
        )
        for reference, prediction in zip(gold, predictions, strict=True)
    ]

    summary = MODULE.grouped_metrics(
        rows,
        ClassificationEvaluator(tokenizer=object()),
    )

    assert summary["metric_prompt_summary"]["accuracy"]["best_prompt"] == (
        "accuracy_prompt"
    )
    assert summary["best_prompt"] == "f1_prompt"
    assert summary["headline_metric"] == "f1"


def test_language_specific_template_selection_must_cover_exact_languages(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selection.json"
    path.write_text('{"eng": "p1", "xho": "p2"}')

    assert MODULE.load_template_selection(path, ["eng", "xho"]) == {
        "eng": "p1",
        "xho": "p2",
    }


def test_template_registry_is_bound_to_frozen_config_tree() -> None:
    repo = Path(__file__).resolve().parents[2]
    config = repo / "src/conf/finetune/llama_injongointent_all.yaml"
    template_id = "injongointent_intent_classification/lm_eval_p1"

    root = MODULE.bind_template_registry(config, [template_id])

    assert root == repo / "src/conf/templates"
    assert MODULE.template_registry.get(template_id).task == (
        "injongointent_intent_classification"
    )
