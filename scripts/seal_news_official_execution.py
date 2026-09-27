#!/usr/bin/env python3
"""Seal the News execution protocol after authorized test materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ARCHITECTURES = ("mzansilm", "mamba", "xlstm", "gdn")
LANGUAGES = ("eng", "xho")
LABELS = [
    "business",
    "entertainment",
    "health",
    "politics",
    "religion",
    "sports",
    "technology",
]
VALIDATION_ROOT = Path(
    "/scratch/alombard/sallm/results/downstream_standardized_20260913_v1/"
    "diagnostics/news_general_four_architecture_validation_20260914_v13"
)
MAMBA_VALIDATION_ROOT = Path(
    "/scratch/alombard/sallm/results/downstream_standardized_20260913_v1/"
    "diagnostics/mamba_news_general_full_validation_20260914_v10"
)
ASSET_BASE = "/scratch/alombard/sallm/assets"
ASSET_ROOTS = {
    "mzansilm": f"{ASSET_BASE}/mzansilm-news-general-validation-20260914-v11",
    "mamba": f"{ASSET_BASE}/mamba-news-general-validation-20260914-v10",
    "xlstm": f"{ASSET_BASE}/xlstm-news-general-validation-20260914-v11",
    "gdn": f"{ASSET_BASE}/gdn-news-general-validation-20260914-v11",
}
BINDING_FILENAMES = {
    "mzansilm": "mzansilm_news_general_validation_binding_20260914.json",
    "mamba": "mamba_news_general_validation_binding_20260914.json",
    "xlstm": "xlstm_news_general_validation_binding_20260914.json",
    "gdn": "gdn_news_general_validation_binding_20260914.json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, str]:
    path = path.resolve()
    require(path.is_file(), f"missing protocol input: {path}")
    return {"path": str(path), "sha256": sha256(path)}


def build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = args.repo.resolve()
    snapshot_manifest = snapshot / "SNAPSHOT_PAYLOAD.sha256"
    access_protocol_path = args.access_protocol.resolve()
    authorization_path = args.authorization.resolve()
    dataset_manifest_path = args.dataset_manifest.resolve()
    access_started_path = dataset_manifest_path.parent / "ACCESS_STARTED.json"

    access = json.loads(access_protocol_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    access_started = json.loads(access_started_path.read_text(encoding="utf-8"))
    require(
        access["schema"] == "sallm.news_official_test_access/v1", "access schema drift"
    )
    require(access["selected_prompts"] == {"eng": "p4", "xho": "p4"}, "prompt drift")
    require(access["no_score_based_retry"] is True, "access retry-policy drift")
    require(
        authorization["status"] == "AUTHORIZED_AFTER_VALIDATION_REVIEW",
        "test access is not authorized",
    )
    require(
        authorization["access_protocol_sha256"] == sha256(access_protocol_path),
        "authorization binding drift",
    )
    require(
        access_started["protocol_sha256"] == sha256(access_protocol_path),
        "access marker protocol drift",
    )
    require(
        access_started["authorization_sha256"] == sha256(authorization_path),
        "access marker authorization drift",
    )
    require(
        dataset["schema"] == "sallm.news_official_test_dataset/v1",
        "dataset schema drift",
    )
    require(dataset["split"] == "test", "dataset split drift")
    require(dataset["metrics_computed"] is False, "test metrics were already computed")
    require(
        dataset["access_started_sha256"] == sha256(access_started_path),
        "dataset access-marker drift",
    )
    require(
        {key: value["rows"] for key, value in dataset["files"].items()}
        == {"eng": 948, "xho": 297},
        "test row-count drift",
    )

    selection_path = Path(access["validation_gate"]["prompt_selection"]["path"])
    chain_path = Path(access["validation_gate"]["chain_verification"]["path"])
    tie_path = Path(access["validation_gate"]["tie_sensitivity"]["path"])
    for key, path in (
        ("prompt_selection", selection_path),
        ("chain_verification", chain_path),
        ("tie_sensitivity", tie_path),
    ):
        require(
            sha256(path) == access["validation_gate"][key]["sha256"],
            f"{key} drift",
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    tie_audit = json.loads(tie_path.read_text(encoding="utf-8"))
    require(
        tie_audit["status"] == "VALIDATION_CHAIN_INDEPENDENTLY_VERIFIED",
        "tie audit failed",
    )
    require(tie_audit["test_accessed"] is False, "tie audit accessed test")
    require(tie_audit["tie_policy"]["frozen_order"] == LABELS, "tie order drift")
    require(
        tie_audit["tie_policy"]["shared_prompt_changes_under_reverse"] is False,
        "reverse tie order changes prompt selection",
    )
    require(
        tie_audit["tie_policy"]["architecture_ranking_changes_under_reverse"] is False,
        "reverse tie order changes architecture ranking",
    )

    validation_summary_paths = {
        "mzansilm": VALIDATION_ROOT / "mzansilm_full/general_summary.json",
        "mamba": MAMBA_VALIDATION_ROOT / "general_summary.json",
        "xlstm": VALIDATION_ROOT / "xlstm_full/general_summary.json",
        "gdn": VALIDATION_ROOT / "gdn_full/general_summary.json",
    }
    validation_summaries = {
        architecture: ref(path)
        for architecture, path in validation_summary_paths.items()
    }
    require(
        {
            architecture: item["sha256"]
            for architecture, item in validation_summaries.items()
        }
        == selection["input_summary_sha256"],
        "validation summary binding drift",
    )
    models = {
        architecture: {
            "display_name": {
                "mzansilm": "MzansiLM",
                "mamba": "Mamba",
                "xlstm": "xLSTM",
                "gdn": "GDN",
            }[architecture],
            "binding": ref(snapshot / ".audit" / BINDING_FILENAMES[architecture]),
            "asset_root": ASSET_ROOTS[architecture],
        }
        for architecture in ARCHITECTURES
    }
    for architecture, model in models.items():
        require(Path(model["asset_root"]).is_dir(), f"missing {architecture} asset")

    validation_tie_counts = {
        architecture: {
            "all_prompts": tie_audit["reports"][architecture]["central_top_score_ties"],
            "selected_prompt": {
                language: tie_audit["reports"][architecture][
                    "central_top_score_ties_by_subset"
                ][f"{language}/p4"]
                for language in LANGUAGES
            },
        }
        for architecture in ARCHITECTURES
    }
    return {
        "schema": "sallm.news_general_official_execution/v1",
        "status": "READY_FOR_SINGLE_CORRECTED_REPLACEMENT",
        "data_boundary": "held_out_test",
        "split": "test",
        "regime": "General",
        "no_score_based_retry": True,
        "selected_prompts": access["selected_prompts"],
        "validation_tie_counts": validation_tie_counts,
        "validation_protocol": ref(
            snapshot / ".audit/mamba_news_common_validation_protocol_20260914.json"
        ),
        "validation_gate": {
            "chain_verification": ref(chain_path),
            "prompt_selection": ref(selection_path),
            "tie_sensitivity": ref(tie_path),
        },
        "validation_summaries": validation_summaries,
        "test_access": {
            "protocol": ref(access_protocol_path),
            "authorization": ref(authorization_path),
            "access_started": ref(access_started_path),
        },
        "dataset_manifest": ref(dataset_manifest_path),
        "runtime": {
            "snapshot": str(snapshot),
            "snapshot_manifest": ref(snapshot_manifest),
            "visible_gpu": "1",
            "device": "cuda:0",
            "dtype": "bfloat16",
            "maximum_input_tokens": 1024,
            "central_metric": "singleton_natural_first_label_token_log_probability",
            "labels": LABELS,
            "exact_tie_break": "first_label_in_frozen_order",
        },
        "models": models,
        "output_root": str(args.results.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--access-protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "official execution protocol already exists")
    protocol = build_protocol(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "output": str(args.output),
                "sha256": sha256(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
