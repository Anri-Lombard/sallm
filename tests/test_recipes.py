from pathlib import Path

import pytest
import yaml
from sallm.recipes import get_recipe, load_recipes, resolve_config_target


def write_registry(path: Path, recipes: list[dict]) -> None:
    path.write_text(yaml.safe_dump(recipes, sort_keys=False))


def test_load_default_recipes() -> None:
    recipes = load_recipes()

    assert set(recipes) == {
        "llama_t2x_xho",
        "mamba_news_xho",
        "xlstm_sib_xho",
    }
    assert recipes["llama_t2x_xho"].configs == {
        "finetune": "finetune/llama_t2x_xho",
        "evaluate": "eval/run_llama_t2x_xho",
    }


def test_get_recipe() -> None:
    recipe = get_recipe("mamba_news_xho")

    assert recipe.configs["finetune"] == "finetune/mamba_news_xho"
    assert "masakhanews" in recipe.tags


def test_duplicate_recipe_ids_fail(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    conf_root = tmp_path / "conf"
    (conf_root / "finetune").mkdir(parents=True)
    (conf_root / "finetune" / "demo.yaml").write_text("mode: FINETUNE\n")
    write_registry(
        registry_path,
        [
            {
                "id": "demo",
                "configs": {"finetune": "finetune/demo"},
            },
            {
                "id": "demo",
                "configs": {"finetune": "finetune/demo"},
            },
        ],
    )

    with pytest.raises(ValueError, match="Duplicate recipe id: demo"):
        load_recipes(registry_path, conf_root)


def test_missing_config_targets_fail(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    conf_root = tmp_path / "conf"
    conf_root.mkdir()
    write_registry(
        registry_path,
        [
            {
                "id": "demo",
                "configs": {"finetune": "finetune/missing"},
            },
        ],
    )

    with pytest.raises(FileNotFoundError, match="finetune/missing"):
        load_recipes(registry_path, conf_root)


def test_config_targets_must_stay_under_conf_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside the Hydra config tree"):
        resolve_config_target("../outside", tmp_path)
