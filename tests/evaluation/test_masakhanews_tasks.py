import importlib.util
from pathlib import Path

import pytest
import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = ROOT / "src" / "conf" / "eval" / "lm_eval_tasks"
LABELS = [
    "business",
    "entertainment",
    "health",
    "politics",
    "religion",
    "sports",
    "technology",
]


class _TaskLoader(yaml.SafeLoader):
    pass


_TaskLoader.add_constructor(
    "!function",
    lambda loader, node: loader.construct_scalar(node),
)


def _load_utils(task_dir: Path):
    spec = importlib.util.spec_from_file_location(
        f"{task_dir.name}_utils",
        task_dir / "utils.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load task utilities from {task_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "task_dir_name",
    ["masakhanews_validation", "masakhanews_test"],
)
def test_masakhanews_choices_are_uniform_single_tokens(task_dir_name: str) -> None:
    task_dir = TASK_ROOT / task_dir_name
    common_path = next(task_dir.glob("*_common.yaml"))
    config = yaml.load(common_path.read_text(), Loader=_TaskLoader)
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "tokenizer" / "sallm_bpe_tokenizer",
        local_files_only=True,
    )

    choices = config["doc_to_choice"]
    token_counts = [
        len(tokenizer.encode(choice, add_special_tokens=False)) for choice in choices
    ]

    assert choices == [f" {label}" for label in LABELS]
    assert token_counts == [1] * len(LABELS)


@pytest.mark.parametrize(
    "task_dir_name",
    ["masakhanews_validation", "masakhanews_test"],
)
def test_masakhanews_target_mapping_is_exact(task_dir_name: str) -> None:
    task_utils = _load_utils(TASK_ROOT / task_dir_name)

    assert [task_utils.doc_to_target({"category": label}) for label in LABELS] == list(
        range(len(LABELS))
    )
    with pytest.raises(ValueError, match="Unknown MasakhaNEWS category"):
        task_utils.doc_to_target({"category": "geography"})
