from __future__ import annotations

import subprocess
import sys

from sallm import cli


def test_recipes_list_prints_recipe_ids(capsys) -> None:
    exit_code = cli.main(["recipes", "list"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "llama_t2x_xho" in output
    assert "mamba_news_xho" in output
    assert "xlstm_sib_xho" in output


def test_recipe_show_prints_recipe(capsys) -> None:
    exit_code = cli.main(["recipe", "show", "llama_t2x_xho"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "id: llama_t2x_xho" in output
    assert "finetune: finetune/llama_t2x_xho" in output
    assert "evaluate: eval/run_llama_t2x_xho" in output


def test_finetune_dry_run_prints_resolved_config(capsys) -> None:
    exit_code = cli.main(["finetune", "llama_t2x_xho", "--dry-run"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "recipe: llama_t2x_xho" in output
    assert "action: finetune" in output
    assert "config: finetune/llama_t2x_xho" in output
    assert (
        f"command: {sys.executable} -m sallm.main --config-name finetune/llama_t2x_xho"
    ) in output


def test_evaluate_dry_run_prints_resolved_config(capsys) -> None:
    exit_code = cli.main(["evaluate", "llama_t2x_xho", "--dry-run"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "action: evaluate" in output
    assert "config: eval/run_llama_t2x_xho" in output


def test_unknown_recipe_returns_error(capsys) -> None:
    exit_code = cli.main(["recipe", "show", "missing"])

    assert exit_code == 2
    assert "Unknown recipe id: missing" in capsys.readouterr().err


def test_run_recipe_calls_existing_hydra_entrypoint(monkeypatch, capsys) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    exit_code = cli.main(["finetune", "llama_t2x_xho"])

    assert exit_code == 0
    assert calls == [
        [sys.executable, "-m", "sallm.main", "--config-name", "finetune/llama_t2x_xho"]
    ]
    assert "config: finetune/llama_t2x_xho" in capsys.readouterr().out
