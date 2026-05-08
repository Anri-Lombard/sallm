from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml
from hydra import compose, initialize
from omegaconf import DictConfig

from sallm.main import run_experiment
from sallm.recipes import Recipe, get_recipe, load_recipes


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command

    if command == "recipes":
        return _list_recipes()
    if command == "recipe":
        return _show_recipe(args.recipe_id)
    if command in {"finetune", "evaluate"}:
        return _run_recipe(command, args.recipe_id, args.dry_run)

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sallm",
        description="Run known SALLM recipes through the existing Hydra entrypoint.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    recipes_parser = subparsers.add_parser("recipes", help="Inspect recipe IDs.")
    recipes_subparsers = recipes_parser.add_subparsers(
        dest="recipes_command",
        required=True,
    )
    list_parser = recipes_subparsers.add_parser("list", help="List available recipes.")
    list_parser.set_defaults(command="recipes")

    recipe_parser = subparsers.add_parser("recipe", help="Inspect one recipe.")
    recipe_subparsers = recipe_parser.add_subparsers(
        dest="recipe_command",
        required=True,
    )
    show_parser = recipe_subparsers.add_parser("show", help="Show a recipe.")
    show_parser.add_argument("recipe_id")
    show_parser.set_defaults(command="recipe")

    for command in ("finetune", "evaluate"):
        run_parser = subparsers.add_parser(command, help=f"Run {command} for a recipe.")
        run_parser.add_argument("recipe_id")
        run_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the resolved config and command without launching it.",
        )

    return parser


def _list_recipes() -> int:
    recipes = load_recipes()
    for recipe in recipes.values():
        print(f"{recipe.id}\t{recipe.description}")
    return 0


def _show_recipe(recipe_id: str) -> int:
    recipe = _load_recipe(recipe_id)
    if recipe is None:
        return 2

    print(
        yaml.safe_dump(
            {
                "id": recipe.id,
                "description": recipe.description,
                "configs": recipe.configs,
                "tags": list(recipe.tags),
            },
            sort_keys=False,
        ).strip()
    )
    return 0


CONF_PATH = Path(__file__).resolve().parents[2] / "conf"


def _run_recipe(command: str, recipe_id: str, dry_run: bool) -> int:
    recipe = _load_recipe(recipe_id)
    if recipe is None:
        return 2

    config_target = recipe.configs.get(command)
    if config_target is None:
        print(
            f"Recipe '{recipe_id}' does not define a {command} config.",
            file=sys.stderr,
        )
        return 2

    hydra_command = ["hydra", "compose", "--config-name", config_target]
    _print_resolved_run(recipe, command, config_target, hydra_command)
    if dry_run:
        return 0

    with initialize(version_base=None, config_path=str(CONF_PATH)):
        cfg = compose(config_name=config_target)
        run_experiment(cfg)
    return 0


def _load_recipe(recipe_id: str) -> Recipe | None:
    try:
        return get_recipe(recipe_id)
    except KeyError:
        print(f"Unknown recipe id: {recipe_id}", file=sys.stderr)
        return None


def _print_resolved_run(
    recipe: Recipe,
    command: str,
    config_target: str,
    hydra_command: list[str],
) -> None:
    print(f"recipe: {recipe.id}")
    print(f"action: {command}")
    print(f"config: {config_target}")
    print("command: " + " ".join(hydra_command))


if __name__ == "__main__":
    raise SystemExit(main())
