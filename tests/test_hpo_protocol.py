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

# The registry is gitignored by the repo-wide *.json rule, so it exists only in
# working copies that ran the protocol.
pytestmark = pytest.mark.skipif(
    not REGISTRY.is_file(), reason=f"local-only HPO registry missing: {REGISTRY}"
)


def test_registry_and_uniform_wrapper_dry_run(tmp_path: Path) -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    candidates = registry["stage_a"] + registry["stage_b"]
    assert len(candidates) == 11
    assert len({candidate["id"] for candidate in candidates}) == 11
    assert registry["stage_b"][0] == {
        "id": "b0",
        "sobol_point": [
            0.8795701265335083,
            0.31370800733566284,
            0.28929591178894043,
            0.5698526501655579,
        ],
        "learning_rate": 0.00015156541821567134,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.028929591178894043,
        "warmup_ratio": 0.061286738514900206,
    }
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/run_validation_hpo_trial.sh"),
            "pure_gdn",
            "ner",
            "stage_b",
            "b0",
            "42",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "SALLM_DRY_RUN": "1",
            "SALLM_REPO_DIR": str(ROOT),
            "SCRATCH": str(tmp_path),
        },
    )
    assert (
        "model=pure_gdn family=ner stage=stage_b candidate=b0 seed=42" in result.stdout
    )
    assert "lr=0.00015156541821567134 rank=8 alpha=16" in result.stdout
    assert "/adapter_hpo_v3/pure_gdn/ner/stage_b/b0/seed_42" in result.stdout


def test_mamba_t2x_profile_dry_run_uses_frozen_runtime_contract(
    tmp_path: Path,
) -> None:
    model = tmp_path / "mamba-125m"
    tokenizer = tmp_path / "tokenizer"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/run_validation_hpo_trial.sh"),
            "mamba125",
            "t2x",
            "stage_a",
            "a0",
            "42",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "SALLM_DRY_RUN": "1",
            "SALLM_REPO_DIR": str(ROOT),
            "SCRATCH": str(tmp_path),
            "SALLM_HPO_MODEL": str(model),
            "SALLM_HPO_TOKENIZER": str(tokenizer),
        },
    )
    assert f"model_path={model} tokenizer={tokenizer}" in result.stdout
    assert "targets=[in_proj,out_proj]" in result.stdout
    assert "train_batch=1 eval_batch=1 accumulation=8" in result.stdout
    assert "max_length=1024 gradient_checkpointing=false" in result.stdout


def test_pure_gdn_news_profile_dry_run_uses_corrected_validation_contract(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/run_validation_hpo_trial.sh"),
            "pure_gdn",
            "news",
            "stage_a",
            "a0",
            "42",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "SALLM_DRY_RUN": "1",
            "SALLM_REPO_DIR": str(ROOT),
            "SCRATCH": str(tmp_path),
        },
    )
    assert "config=gdn_pure_news_all_hpo_r1" in result.stdout
    assert "metric=eval_classification/all_macro_f1 greater=true" in result.stdout
    assert "epochs=10" in result.stdout
    assert "max_length=1024 gradient_checkpointing=false" in result.stdout
    assert (
        "train_batch=4 eval_batch=4 accumulation=2 effective_batch=8" in result.stdout
    )


def test_validation_ranking_uses_frozen_tie_breaks(tmp_path: Path) -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    by_id = {
        candidate["id"]: candidate
        for candidate in registry["stage_a"] + registry["stage_b"]
    }
    registry_hash = hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    trial_dirs = []
    for candidate_id in ("b0", "b1", "b2"):
        trial_dir = tmp_path / candidate_id
        trial_dir.mkdir()
        manifest = trial_dir / "execution_manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        config = {
            key: by_id[candidate_id][key]
            for key in (
                "learning_rate",
                "lora_rank",
                "lora_alpha",
                "lora_dropout",
                "warmup_ratio",
            )
        }
        (trial_dir / "hpo_trial.json").write_text(
            json.dumps(
                {
                    "selection_split": "validation",
                    "candidate": candidate_id,
                    "seed": 42,
                    "candidate_config": config,
                    "registry_sha256": registry_hash,
                    "execution_manifest": str(manifest),
                    "execution_manifest_sha256": hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        (trial_dir / "trainer_state.json").write_text(
            json.dumps({"global_step": 10, "best_metric": 0.5}),
            encoding="utf-8",
        )
        trial_dirs.append(str(trial_dir))
    result = subprocess.run(
        [
            sys.executable,
            str(PROTOCOL),
            "rank",
            "--registry",
            str(REGISTRY),
            "--stage",
            "seed42",
            "--direction",
            "max",
            *trial_dirs,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    ranking = json.loads(result.stdout)["ranking"]
    assert [row["candidate"] for row in ranking] == ["b2", "b0", "b1"]


def test_reduced_confirmation_ranking_requires_seeds_13_and_42(
    tmp_path: Path,
) -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    by_id = {
        candidate["id"]: candidate
        for candidate in registry["stage_a"] + registry["stage_b"]
    }
    registry_hash = hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    trial_dirs = []
    candidate_metrics = {
        "a0": {13: 0.7, 42: 0.5},
        "a1": {13: 0.4, 42: 0.6},
    }
    for candidate_id, metrics in candidate_metrics.items():
        for seed, metric in metrics.items():
            trial_dir = tmp_path / candidate_id / str(seed)
            trial_dir.mkdir(parents=True)
            manifest = trial_dir / "execution_manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            config = {
                key: by_id[candidate_id][key]
                for key in (
                    "learning_rate",
                    "lora_rank",
                    "lora_alpha",
                    "lora_dropout",
                    "warmup_ratio",
                )
            }
            (trial_dir / "hpo_trial.json").write_text(
                json.dumps(
                    {
                        "selection_split": "validation",
                        "candidate": candidate_id,
                        "seed": seed,
                        "candidate_config": config,
                        "registry_sha256": registry_hash,
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
            trial_dirs.append(str(trial_dir))

    result = subprocess.run(
        [
            sys.executable,
            str(PROTOCOL),
            "rank",
            "--registry",
            str(REGISTRY),
            "--stage",
            "reduced-confirm",
            "--direction",
            "max",
            *trial_dirs,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["stage"] == "reduced-confirm"
    assert [row["candidate"] for row in payload["ranking"]] == ["a0", "a1"]
    assert payload["ranking"][0]["metrics"] == {"13": 0.7, "42": 0.5}
