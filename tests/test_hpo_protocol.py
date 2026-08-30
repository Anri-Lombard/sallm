from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
REGISTRY = ROOT / "src/conf/hpo/pure_gdn_enhanced_v1.json"
PROTOCOL = ROOT / "scripts/hpo_protocol.py"
MANIFEST = ROOT / "scripts/create_execution_manifest.py"
CONFIG_KEYS = (
    "learning_rate",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "warmup_ratio",
)


def run(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        check=False,
    )


def write_manifest(
    path: Path,
    *,
    model: str,
    family: str,
    stage: str,
    candidate: str,
    seed: int = 42,
    variable_overrides: dict[str, str | None] | None = None,
) -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in registry["stage_a"] + registry["stage_b"]}
    config = by_id[candidate]
    variables: dict[str, str] = {
        "SALLM_HPO_MODEL_ID": model,
        "SALLM_HPO_FAMILY": family,
        "SALLM_HPO_STAGE": stage,
        "SALLM_HPO_CANDIDATE": candidate,
        "SALLM_HPO_SEED": str(seed),
        "SALLM_HPO_LEARNING_RATE": str(config["learning_rate"]),
        "SALLM_HPO_LORA_RANK": str(config["lora_rank"]),
        "SALLM_HPO_LORA_ALPHA": str(config["lora_alpha"]),
        "SALLM_HPO_LORA_DROPOUT": str(config["lora_dropout"]),
        "SALLM_HPO_WARMUP_RATIO": str(config["warmup_ratio"]),
    }
    for key, value in (variable_overrides or {}).items():
        if value is None:
            variables.pop(key, None)
        else:
            variables[key] = value
    path.write_text(
        json.dumps(
            {
                "schema": "sallm_execution_manifest/v1",
                "environment": {"variables": variables},
            }
        ),
        encoding="utf-8",
    )
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )


def write_trial(
    root: Path,
    *,
    candidate: str,
    seed: int,
    metric: float,
    model: str = "pure_gdn",
    family: str = "ner",
    stage: str | None = None,
) -> Path:
    stage = stage or ("stage_a" if candidate.startswith("a") else "stage_b")
    root.mkdir(parents=True, exist_ok=True)
    trial_dir = root / f"{candidate}-{seed}"
    trial_dir.mkdir()
    manifest = trial_dir / "execution_manifest.json"
    write_manifest(
        manifest,
        model=model,
        family=family,
        stage=stage,
        candidate=candidate,
        seed=seed,
    )
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in registry["stage_a"] + registry["stage_b"]}
    (trial_dir / "hpo_trial.json").write_text(
        json.dumps(
            {
                "selection_split": "validation",
                "model": model,
                "family": family,
                "stage": stage,
                "candidate": candidate,
                "seed": seed,
                "candidate_config": {key: by_id[candidate][key] for key in CONFIG_KEYS},
                "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
                "execution_manifest": str(manifest),
                "execution_manifest_sha256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": 10, "best_metric": metric}),
        encoding="utf-8",
    )
    return trial_dir


def seed42_trials(
    root: Path, metric_by_candidate: dict[str, float] | None = None
) -> list[Path]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return [
        write_trial(
            root,
            candidate=item["id"],
            seed=42,
            metric=(metric_by_candidate or {}).get(item["id"], 0.1),
        )
        for item in registry["stage_a"] + registry["stage_b"]
    ]


def write_seed42_ranking(
    root: Path, metric_by_candidate: dict[str, float]
) -> tuple[Path, list[Path]]:
    trials = seed42_trials(root, metric_by_candidate)
    output = root / "seed42-ranking.json"
    result = run(
        str(PROTOCOL),
        "rank",
        "--registry",
        str(REGISTRY),
        "--stage",
        "seed42",
        "--direction",
        "max",
        "--output",
        str(output),
        *(str(path) for path in trials),
    )
    assert result.returncode == 0, result.stderr
    return output, trials


def test_manifest_hashes_ops_and_rejects_tampering(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    for relative in ("src/main/sallm", "src/conf", "scripts", "ops"):
        (repo / relative).mkdir(parents=True)
    (repo / "src/main/sallm/module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "src/conf/base.yaml").write_text("value: 1\n", encoding="utf-8")
    (repo / "scripts/run.py").write_text("print('ok')\n", encoding="utf-8")
    job = repo / "ops/train.sh"
    job.write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
    output = tmp_path / "manifest.json"

    created = run(
        str(MANIFEST),
        "--repo-root",
        str(repo),
        "--output",
        str(output),
        "--entrypoint",
        "ops/train.sh",
    )
    assert created.returncode == 0, created.stderr
    assert "ops/train.sh" in json.loads(output.read_text())["source_hashes"]

    original = output.read_bytes()
    rewritten = json.loads(original)
    rewritten["entrypoint"] = "ops/other.sh"
    output.write_text(json.dumps(rewritten), encoding="utf-8")
    self_authorized = run(str(MANIFEST), "--verify", str(output))
    assert self_authorized.returncode != 0
    assert "sidecar" in self_authorized.stderr
    output.write_bytes(original)

    job.write_text("#!/bin/sh\nfalse\n", encoding="utf-8")
    verified = run(str(MANIFEST), "--verify", str(output))
    assert verified.returncode != 0
    assert "ops/train.sh" in verified.stderr

    job.write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
    (repo / "ops/unrecorded.sh").write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
    added = run(str(MANIFEST), "--verify", str(output))
    assert added.returncode != 0
    assert "ops/unrecorded.sh" in added.stderr


def test_write_trial_rejects_candidate_stage_mismatch(tmp_path: Path) -> None:
    manifest = tmp_path / "execution_manifest.json"
    write_manifest(
        manifest,
        model="pure_gdn",
        family="ner",
        stage="stage_a",
        candidate="b0",
    )
    result = run(
        str(PROTOCOL),
        "write-trial",
        "--registry",
        str(REGISTRY),
        "--candidate",
        "b0",
        "--seed",
        "42",
        "--model",
        "pure_gdn",
        "--family",
        "ner",
        "--stage",
        "stage_a",
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "hpo_trial.json"),
    )
    assert result.returncode != 0
    assert "belongs to stage_b" in result.stderr


def test_write_trial_rejects_manifest_metadata_mismatch(tmp_path: Path) -> None:
    manifest = tmp_path / "execution_manifest.json"
    write_manifest(
        manifest,
        model="pure_gdn",
        family="pos",
        stage="stage_a",
        candidate="a0",
    )
    result = run(
        str(PROTOCOL),
        "write-trial",
        "--registry",
        str(REGISTRY),
        "--candidate",
        "a0",
        "--seed",
        "42",
        "--model",
        "pure_gdn",
        "--family",
        "ner",
        "--stage",
        "stage_a",
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "hpo_trial.json"),
    )
    assert result.returncode != 0
    assert "SALLM_HPO_FAMILY" in result.stderr


@pytest.mark.parametrize(
    ("variable", "value"),
    (("SALLM_HPO_SEED", "87"), ("SALLM_HPO_LEARNING_RATE", None)),
)
def test_write_trial_rejects_seed_or_hyperparameter_manifest_mismatch(
    tmp_path: Path,
    variable: str,
    value: str | None,
) -> None:
    manifest = tmp_path / "execution_manifest.json"
    write_manifest(
        manifest,
        model="pure_gdn",
        family="ner",
        stage="stage_a",
        candidate="a0",
        variable_overrides={variable: value},
    )
    result = run(
        str(PROTOCOL),
        "write-trial",
        "--registry",
        str(REGISTRY),
        "--candidate",
        "a0",
        "--seed",
        "42",
        "--model",
        "pure_gdn",
        "--family",
        "ner",
        "--stage",
        "stage_a",
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "hpo_trial.json"),
    )
    assert result.returncode != 0
    assert variable in result.stderr


def test_write_trial_binds_matching_manifest_metadata(tmp_path: Path) -> None:
    manifest = tmp_path / "execution_manifest.json"
    output = tmp_path / "hpo_trial.json"
    write_manifest(
        manifest,
        model="pure_gdn",
        family="ner",
        stage="stage_a",
        candidate="a0",
    )
    result = run(
        str(PROTOCOL),
        "write-trial",
        "--registry",
        str(REGISTRY),
        "--candidate",
        "a0",
        "--seed",
        "42",
        "--model",
        "pure_gdn",
        "--family",
        "ner",
        "--stage",
        "stage_a",
        "--manifest",
        str(manifest),
        "--output",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    trial = json.loads(output.read_text(encoding="utf-8"))
    assert trial["schema"] == "sallm_hpo_trial/v1"
    assert (
        trial["execution_manifest_sha256"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )


def test_ranking_rejects_cross_trial_metadata_mismatch(tmp_path: Path) -> None:
    trials = [
        write_trial(tmp_path, candidate="a0", seed=42, metric=0.4),
        write_trial(
            tmp_path,
            candidate="a1",
            seed=42,
            metric=0.5,
            family="pos",
        ),
    ]
    result = run(
        str(PROTOCOL),
        "rank",
        "--registry",
        str(REGISTRY),
        "--stage",
        "seed42",
        "--direction",
        "max",
        *(str(path) for path in trials),
    )
    assert result.returncode != 0
    assert "model/family mismatch" in result.stderr


def test_ranking_rejects_candidate_stage_mismatch(tmp_path: Path) -> None:
    trial = write_trial(
        tmp_path,
        candidate="b0",
        seed=42,
        metric=0.5,
        stage="stage_a",
    )
    result = run(
        str(PROTOCOL),
        "rank",
        "--registry",
        str(REGISTRY),
        "--stage",
        "seed42",
        "--direction",
        "max",
        str(trial),
    )
    assert result.returncode != 0
    assert "belongs to stage_b" in result.stderr


def test_seed42_ranking_rejects_incomplete_candidate_set(tmp_path: Path) -> None:
    trial = write_trial(tmp_path, candidate="a0", seed=42, metric=0.5)
    result = run(
        str(PROTOCOL),
        "rank",
        "--registry",
        str(REGISTRY),
        "--stage",
        "seed42",
        "--direction",
        "max",
        str(trial),
    )
    assert result.returncode != 0
    assert "complete candidate set" in result.stderr


def test_seed42_ranking_uses_frozen_tie_breaks(tmp_path: Path) -> None:
    trials = seed42_trials(
        tmp_path, {candidate: 0.5 for candidate in ("b0", "b1", "b2")}
    )
    result = run(
        str(PROTOCOL),
        "rank",
        "--registry",
        str(REGISTRY),
        "--stage",
        "seed42",
        "--direction",
        "max",
        *(str(path) for path in trials),
    )
    assert result.returncode == 0, result.stderr
    ranking = json.loads(result.stdout)["ranking"]
    assert [row["candidate"] for row in ranking[:3]] == ["b2", "b0", "b1"]


def test_confirmation_ranking_reuses_seed42_and_requires_confirmation_stages(
    tmp_path: Path,
) -> None:
    ranking, seed42 = write_seed42_ranking(
        tmp_path / "selection", {"b0": 0.9, "b1": 0.8}
    )
    top_two = {"b0", "b1"}
    trials = [path for path in seed42 if path.name.split("-")[0] in top_two]
    for candidate in top_two:
        trials.extend(
            [
                write_trial(
                    tmp_path / "confirmation",
                    candidate=candidate,
                    seed=seed,
                    metric=0.7,
                    stage="confirm",
                )
                for seed in (13, 87)
            ]
        )
    result = run(
        str(PROTOCOL),
        "rank",
        "--registry",
        str(REGISTRY),
        "--stage",
        "confirm",
        "--direction",
        "max",
        "--seed42-ranking",
        str(ranking),
        *(str(path) for path in trials),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert [row["candidate"] for row in payload["ranking"]] == ["b0", "b1"]
    assert (
        payload["seed42_ranking_sha256"]
        == hashlib.sha256(ranking.read_bytes()).hexdigest()
    )


def test_confirmation_ranking_rejects_candidates_outside_frozen_top_two(
    tmp_path: Path,
) -> None:
    ranking, seed42 = write_seed42_ranking(
        tmp_path / "selection", {"b0": 0.9, "b1": 0.8}
    )
    b2_seed42 = next(path for path in seed42 if path.name == "b2-42")
    trials = [b2_seed42]
    trials.extend(
        write_trial(
            tmp_path / "confirmation",
            candidate="b2",
            seed=seed,
            metric=0.7,
            stage="confirm",
        )
        for seed in (13, 87)
    )
    result = run(
        str(PROTOCOL),
        "rank",
        "--registry",
        str(REGISTRY),
        "--stage",
        "confirm",
        "--direction",
        "max",
        "--seed42-ranking",
        str(ranking),
        *(str(path) for path in trials),
    )
    assert result.returncode != 0
    assert "frozen top two" in result.stderr
