from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONF_ROOT = _REPO_ROOT / "src" / "conf"
_REGISTRY_PATH = _REPO_ROOT / "recipes" / "registry.yaml"


@dataclass(frozen=True)
class Recipe:
    id: str
    description: str
    configs: dict[str, str]
    tags: tuple[str, ...] = ()


def load_recipes(
    registry_path: Path | None = None,
    conf_root: Path | None = None,
) -> dict[str, Recipe]:
    path = registry_path or _REGISTRY_PATH
    root = conf_root or _CONF_ROOT

    with path.open() as f:
        raw_recipes = yaml.safe_load(f)

    if not isinstance(raw_recipes, list):
        raise TypeError("Recipe registry must be a YAML list.")

    recipes: dict[str, Recipe] = {}
    for raw_recipe in raw_recipes:
        recipe = _parse_recipe(raw_recipe)
        if recipe.id in recipes:
            raise ValueError(f"Duplicate recipe id: {recipe.id}")

        for config_target in recipe.configs.values():
            resolve_config_target(config_target, root)

        recipes[recipe.id] = recipe

    return recipes


def get_recipe(
    recipe_id: str,
    registry_path: Path | None = None,
    conf_root: Path | None = None,
) -> Recipe:
    return load_recipes(registry_path, conf_root)[recipe_id]


def resolve_config_target(config_target: str, conf_root: Path | None = None) -> Path:
    root = conf_root or _CONF_ROOT
    path = Path(config_target)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"Config target must stay inside the Hydra config tree: {config_target}"
        )

    config_path = root / path.with_suffix(".yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Recipe config target does not exist: {config_target}")

    return config_path


def _parse_recipe(raw_recipe: Any) -> Recipe:
    if not isinstance(raw_recipe, dict):
        raise TypeError("Each recipe entry must be a mapping.")

    recipe_id = raw_recipe.get("id")
    description = raw_recipe.get("description", "")
    configs = raw_recipe.get("configs")
    tags = raw_recipe.get("tags", [])

    if not isinstance(recipe_id, str) or not recipe_id:
        raise ValueError("Recipe id must be a non-empty string.")
    if not isinstance(description, str):
        raise TypeError(f"Recipe {recipe_id} description must be a string.")
    if not isinstance(configs, dict) or not configs:
        raise ValueError(f"Recipe {recipe_id} must define at least one config target.")
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise TypeError(f"Recipe {recipe_id} tags must be a list of strings.")

    parsed_configs: dict[str, str] = {}
    for name, target in configs.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"Recipe {recipe_id} config names must be non-empty strings."
            )
        if not isinstance(target, str) or not target:
            raise ValueError(
                f"Recipe {recipe_id} config targets must be non-empty strings."
            )
        parsed_configs[name] = target

    return Recipe(
        id=recipe_id,
        description=description,
        configs=parsed_configs,
        tags=tuple(tags),
    )
