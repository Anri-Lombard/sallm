#!/usr/bin/env python3
"""Resolve, record, and rank validation-only adapter HPO trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

HPO_MANIFEST_FIELDS = {
    "SALLM_HPO_MODEL_ID": "model",
    "SALLM_HPO_FAMILY": "family",
    "SALLM_HPO_STAGE": "stage",
    "SALLM_HPO_CANDIDATE": "candidate",
}
HPO_MANIFEST_CONFIG_FIELDS = {
    "SALLM_HPO_LEARNING_RATE": ("learning_rate", float),
    "SALLM_HPO_LORA_RANK": ("lora_rank", int),
    "SALLM_HPO_LORA_ALPHA": ("lora_alpha", int),
    "SALLM_HPO_LORA_DROPOUT": ("lora_dropout", float),
    "SALLM_HPO_WARMUP_RATIO": ("warmup_ratio", float),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha256_sidecar(path: Path) -> str:
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def verify_sha256_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        expected, filename = sidecar.read_text(encoding="utf-8").split()
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"invalid or missing SHA-256 sidecar: {sidecar}") from error
    actual = sha256(path)
    if filename != path.name or expected != actual:
        raise ValueError(
            f"SHA-256 sidecar mismatch for {path}: expected={expected}, actual={actual}"
        )
    return actual


def load_registry(
    path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, str]]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema") not in {
        "sallm_adapter_hpo/v1",
        "sallm_pure_gdn_hpo/v1",
    }:
        raise ValueError("unsupported HPO registry schema")
    candidates = registry.get("stage_a", []) + registry.get("stage_b", [])
    by_id = {candidate["id"]: candidate for candidate in candidates}
    if len(by_id) != len(candidates) or not candidates:
        raise ValueError("candidate IDs must be nonempty and unique")
    candidate_stages = {
        candidate["id"]: stage
        for stage in ("stage_a", "stage_b")
        for candidate in registry.get(stage, [])
    }
    space = registry["search_space"]
    lr_space = space["learning_rate"]
    dropout_space = space["lora_dropout"]
    warmup_space = space["warmup_ratio"]
    ranks = set(space["lora_rank"]["values"])
    for candidate in candidates:
        lr = candidate["learning_rate"]
        rank = candidate["lora_rank"]
        dropout = candidate["lora_dropout"]
        warmup = candidate["warmup_ratio"]
        if not lr_space["low"] <= lr <= lr_space["high"]:
            raise ValueError(f"{candidate['id']}: learning_rate out of range")
        if rank not in ranks or candidate["lora_alpha"] != 2 * rank:
            raise ValueError(f"{candidate['id']}: invalid LoRA rank/alpha")
        if not dropout_space["low"] <= dropout <= dropout_space["high"]:
            raise ValueError(f"{candidate['id']}: dropout out of range")
        if not warmup_space["low"] <= warmup <= warmup_space["high"]:
            raise ValueError(f"{candidate['id']}: warmup out of range")
    return registry, by_id, candidate_stages


def candidate_config(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "learning_rate",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "warmup_ratio",
        )
    }


def write_json(path: Path | None, payload: Any) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if path is None:
        print(encoded, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")


def validate_candidate_stage(
    candidate: str,
    stage: str,
    candidate_stages: dict[str, str],
) -> None:
    registered = candidate_stages[candidate]
    if stage != "confirm" and stage != registered:
        raise ValueError(f"{candidate} belongs to {registered}, not {stage}")


def validate_manifest_metadata(path: Path, metadata: dict[str, Any]) -> None:
    verify_sha256_sidecar(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "sallm_execution_manifest/v1":
        raise ValueError(f"{path}: unsupported execution manifest schema")
    variables = manifest.get("environment", {}).get("variables", {})
    for variable, field in HPO_MANIFEST_FIELDS.items():
        expected = str(metadata[field])
        actual = variables.get(variable)
        if actual != expected:
            raise ValueError(
                f"{path}: {variable} mismatch: expected {expected!r}, got {actual!r}"
            )
    numeric_fields = {
        "SALLM_HPO_SEED": (metadata["seed"], int),
        **{
            variable: (metadata["candidate_config"][field], parser)
            for variable, (field, parser) in HPO_MANIFEST_CONFIG_FIELDS.items()
        },
    }
    for variable, (expected, parser) in numeric_fields.items():
        raw = variables.get(variable)
        try:
            actual = parser(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}: missing or invalid {variable}: {raw!r}"
            ) from error
        if actual != expected:
            raise ValueError(
                f"{path}: {variable} mismatch: expected {expected!r}, got {actual!r}"
            )


def command_validate(args: argparse.Namespace) -> None:
    registry, candidates, _ = load_registry(args.registry)
    print(
        json.dumps(
            {
                "schema": registry["schema"],
                "candidates": len(candidates),
                "sha256": sha256(args.registry),
            },
            sort_keys=True,
        )
    )


def command_resolve(args: argparse.Namespace) -> None:
    _, candidates, _ = load_registry(args.registry)
    try:
        config = candidate_config(candidates[args.candidate])
    except KeyError as error:
        raise SystemExit(f"unknown candidate: {args.candidate}") from error
    if args.tsv:
        print(
            "\t".join(
                str(config[key])
                for key in (
                    "learning_rate",
                    "lora_rank",
                    "lora_alpha",
                    "lora_dropout",
                    "warmup_ratio",
                )
            )
        )
    else:
        write_json(None, {"id": args.candidate, **config})


def command_write_trial(args: argparse.Namespace) -> None:
    registry, candidates, candidate_stages = load_registry(args.registry)
    if args.candidate not in candidates:
        raise SystemExit(f"unknown candidate: {args.candidate}")
    validate_candidate_stage(args.candidate, args.stage, candidate_stages)
    payload = {
        "schema": "sallm_hpo_trial/v1",
        "selection_split": "validation",
        "model": args.model,
        "family": args.family,
        "stage": args.stage,
        "candidate": args.candidate,
        "seed": args.seed,
        "candidate_config": candidate_config(candidates[args.candidate]),
        "registry": str(args.registry.resolve()),
        "registry_schema": registry["schema"],
        "registry_sha256": sha256(args.registry),
        "execution_manifest": str(args.manifest.resolve()),
        "execution_manifest_sha256": sha256(args.manifest),
    }
    validate_manifest_metadata(args.manifest, payload)
    write_json(args.output, payload)


def trial_result(
    trial_dir: Path,
    registry_hash: str,
    candidates: dict[str, dict[str, Any]],
    candidate_stages: dict[str, str],
) -> dict[str, Any]:
    metadata_path = trial_dir / "hpo_trial.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidate_id = metadata["candidate"]
    if candidate_id not in candidates:
        raise ValueError(f"{trial_dir}: unknown candidate {candidate_id}")
    validate_candidate_stage(candidate_id, metadata["stage"], candidate_stages)
    if metadata.get("selection_split") != "validation":
        raise ValueError(f"{trial_dir}: selection split is not validation")
    if metadata["registry_sha256"] != registry_hash:
        raise ValueError(f"{trial_dir}: registry hash mismatch")
    if metadata["candidate_config"] != candidate_config(candidates[candidate_id]):
        raise ValueError(f"{trial_dir}: candidate config mismatch")
    manifest = Path(metadata["execution_manifest"])
    if sha256(manifest) != metadata["execution_manifest_sha256"]:
        raise ValueError(f"{trial_dir}: execution manifest hash mismatch")
    validate_manifest_metadata(manifest, metadata)
    states = list(trial_dir.glob("trainer_state.json")) + list(
        trial_dir.glob("checkpoint-*/trainer_state.json")
    )
    if not states:
        raise ValueError(f"{trial_dir}: no trainer_state.json")
    state_path = max(
        states,
        key=lambda path: json.loads(path.read_text(encoding="utf-8")).get(
            "global_step", -1
        ),
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    metric = state.get("best_metric")
    if not isinstance(metric, (int, float)) or not math.isfinite(metric):
        raise ValueError(f"{state_path}: invalid best_metric")
    return {
        **metadata,
        "metric": metric,
        "trainer_state": str(state_path.resolve()),
        "trainer_state_sha256": sha256(state_path),
    }


def frozen_top_two(
    path: Path,
    *,
    registry_hash: str,
    model: str,
    family: str,
    direction: str,
    candidates: dict[str, dict[str, Any]],
) -> tuple[list[str], str]:
    digest = verify_sha256_sidecar(path)
    ranking = json.loads(path.read_text(encoding="utf-8"))
    expected_metadata = {
        "schema": "sallm_hpo_ranking/v1",
        "selection_split": "validation",
        "model": model,
        "family": family,
        "direction": direction,
        "stage": "seed42",
        "registry_sha256": registry_hash,
    }
    for field, expected in expected_metadata.items():
        if ranking.get(field) != expected:
            raise ValueError(
                f"{path}: {field} mismatch: expected {expected!r}, "
                f"got {ranking.get(field)!r}"
            )
    rows = ranking.get("ranking", [])
    candidate_ids = [row.get("candidate") for row in rows]
    if len(candidate_ids) != len(candidates) or set(candidate_ids) != set(candidates):
        raise ValueError(f"{path}: seed42 ranking is not a complete candidate set")
    for row in rows:
        candidate_id = row["candidate"]
        if row.get("candidate_config") != candidate_config(candidates[candidate_id]):
            raise ValueError(f"{path}: {candidate_id} candidate config mismatch")
    return candidate_ids[:2], digest


def command_rank(args: argparse.Namespace) -> None:
    registry, candidates, candidate_stages = load_registry(args.registry)
    registry_hash = sha256(args.registry)
    results = [
        trial_result(path, registry_hash, candidates, candidate_stages)
        for path in args.trial_dirs
    ]
    contexts = {(result["model"], result["family"]) for result in results}
    if len(contexts) != 1:
        raise ValueError(f"model/family mismatch across trials: {sorted(contexts)}")
    model, family = contexts.pop()
    selection_seed = registry["selection_seed"]
    confirmation_seeds = set(registry["confirmation_seeds"])
    expected_seeds = {selection_seed} if args.stage == "seed42" else confirmation_seeds
    for result in results:
        expected_stage = (
            candidate_stages[result["candidate"]]
            if result["seed"] == selection_seed
            else "confirm"
        )
        if result["stage"] != expected_stage:
            raise ValueError(
                f"{result['candidate']} seed {result['seed']}: expected stage "
                f"{expected_stage}, got {result['stage']}"
            )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["candidate"], []).append(result)
    ranking_binding: dict[str, str] = {}
    if args.stage == "seed42":
        if set(grouped) != set(candidates):
            raise ValueError(
                "seed42 ranking requires the complete candidate set: "
                f"expected {sorted(candidates)}, got {sorted(grouped)}"
            )
    else:
        if args.seed42_ranking is None:
            raise ValueError("confirmation ranking requires --seed42-ranking")
        top_two, ranking_hash = frozen_top_two(
            args.seed42_ranking,
            registry_hash=registry_hash,
            model=model,
            family=family,
            direction=args.direction,
            candidates=candidates,
        )
        if set(grouped) != set(top_two):
            raise ValueError(
                f"confirmation candidates must equal frozen top two {top_two}, "
                f"got {sorted(grouped)}"
            )
        ranking_binding = {
            "seed42_ranking": str(args.seed42_ranking.resolve()),
            "seed42_ranking_sha256": ranking_hash,
        }
    rows = []
    for candidate_id, candidate_results in grouped.items():
        seeds = {result["seed"] for result in candidate_results}
        if seeds != expected_seeds or len(seeds) != len(candidate_results):
            raise ValueError(
                f"{candidate_id}: expected seeds {sorted(expected_seeds)}, "
                f"got {sorted(seeds)}"
            )
        metrics = [result["metric"] for result in candidate_results]
        rows.append(
            {
                "candidate": candidate_id,
                "mean": statistics.mean(metrics),
                "sample_sd": statistics.stdev(metrics) if len(metrics) > 1 else None,
                "metrics": {
                    str(result["seed"]): result["metric"]
                    for result in sorted(
                        candidate_results, key=lambda item: item["seed"]
                    )
                },
                "candidate_config": candidate_config(candidates[candidate_id]),
            }
        )
    direction = -1 if args.direction == "max" else 1
    rows.sort(
        key=lambda row: (
            direction * row["mean"],
            row["candidate_config"]["lora_rank"],
            row["candidate_config"]["learning_rate"],
            row["candidate"],
        )
    )
    payload = {
        "schema": "sallm_hpo_ranking/v1",
        "selection_split": "validation",
        "model": model,
        "family": family,
        "direction": args.direction,
        "stage": args.stage,
        "registry_sha256": registry_hash,
        "ranking": rows,
        **ranking_binding,
    }
    write_json(args.output, payload)
    if args.output is not None:
        write_sha256_sidecar(args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--registry", type=Path, required=True)
    validate.set_defaults(func=command_validate)

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--registry", type=Path, required=True)
    resolve.add_argument("--candidate", required=True)
    resolve.add_argument("--tsv", action="store_true")
    resolve.set_defaults(func=command_resolve)

    write_trial = subparsers.add_parser("write-trial")
    write_trial.add_argument("--registry", type=Path, required=True)
    write_trial.add_argument("--candidate", required=True)
    write_trial.add_argument("--seed", type=int, required=True)
    write_trial.add_argument("--model", required=True)
    write_trial.add_argument("--family", required=True)
    write_trial.add_argument(
        "--stage", choices=("stage_a", "stage_b", "confirm"), required=True
    )
    write_trial.add_argument("--manifest", type=Path, required=True)
    write_trial.add_argument("--output", type=Path, required=True)
    write_trial.set_defaults(func=command_write_trial)

    rank = subparsers.add_parser("rank")
    rank.add_argument("--registry", type=Path, required=True)
    rank.add_argument("--stage", choices=("seed42", "confirm"), required=True)
    rank.add_argument("--direction", choices=("max", "min"), required=True)
    rank.add_argument("--seed42-ranking", type=Path)
    rank.add_argument("--output", type=Path)
    rank.add_argument("trial_dirs", type=Path, nargs="+")
    rank.set_defaults(func=command_rank)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
