#!/usr/bin/env python3
"""Verify the limited validation-only General prompt-policy GPU canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from injongointent_trainonly_validation import MANIFEST_SCHEMA

LANGUAGES = ("eng", "sot", "xho", "zul")
PROMPTS = range(1, 6)
INTENT_PROMPTS = tuple(
    f"injongointent_intent_classification/lm_eval_p{prompt}" for prompt in PROMPTS
)
MODEL_INTERFACES = {
    "mzansilm": {
        "dtype": "bfloat16",
        "merge_lora": False,
        "tie_word_embeddings": None,
    },
    "mamba2": {
        "dtype": "float32",
        "merge_lora": False,
        "tie_word_embeddings": False,
    },
    "xlstm": {
        "dtype": "float32",
        "merge_lora": True,
        "tie_word_embeddings": False,
    },
    "gdn": {
        "dtype": "bfloat16",
        "merge_lora": False,
        "tie_word_embeddings": None,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_lm_tasks() -> set[str]:
    tasks = set()
    for language in LANGUAGES:
        for prompt in PROMPTS:
            tasks.add(f"sallm_val_afrixnli_{language}_prompt_{prompt}")
            tasks.add(f"sallm_val_afrimmlu_direct_{language}_prompt_{prompt}")
    return tasks


def verify_lm(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "sallm.general_prompt_lm_eval/v1":
        raise ValueError("Unexpected lm-eval canary schema")
    if payload.get("phase") != "validation" or payload.get("limit") != 1:
        raise ValueError("lm-eval canary must be validation-only with limit=1")
    if payload.get("maximum_input_tokens") != 1024:
        raise ValueError("lm-eval canary did not bind max_length=1024")
    architecture = str(payload.get("architecture"))
    if payload.get("model_interface") != MODEL_INTERFACES.get(architecture):
        raise ValueError("lm-eval canary model interface changed")
    expected = expected_lm_tasks()
    evidence = set(payload.get("task_evidence", {}))
    call = payload.get("calls", {}).get("validation", {})
    results = set(call.get("results", {}))
    samples = call.get("samples", {})
    if evidence != expected or results != expected or set(samples) != expected:
        raise ValueError(
            "lm-eval canary task mismatch: "
            f"evidence={len(evidence)}, results={len(results)}, samples={len(samples)}"
        )
    bad_counts = {task: len(rows) for task, rows in samples.items() if len(rows) != 1}
    if bad_counts:
        raise ValueError(f"lm-eval canary sample counts are not one: {bad_counts}")
    return {"task_count": len(expected), "sample_count": len(expected)}


def verify_intent(
    payload: dict[str, Any], manifest: dict[str, Any], manifest_sha256: str
) -> dict[str, Any]:
    if payload.get("schema") != "sallm.injongointent_mean_eval/v2":
        raise ValueError("Unexpected Intent canary schema")
    if payload.get("split") != "validation":
        raise ValueError("Intent canary must be validation-only")
    if payload.get("data_boundary") != "pinned_train_only_validation":
        raise ValueError("Intent canary is not bound to train-only validation")
    if payload.get("held_out_data_accessed") is not False:
        raise ValueError("Intent canary accessed held-out data")
    if payload.get("validation_manifest_sha256") != manifest_sha256:
        raise ValueError("Intent canary validation manifest changed")
    evidence = payload.get("validation_data", {})
    if evidence.get("source_split") != "train":
        raise ValueError("Intent canary source is not pinned train")
    if evidence.get("held_out_data_accessed") is not False:
        raise ValueError("Intent canary evidence accessed held-out data")
    if evidence.get("architecture_blind") is not True:
        raise ValueError("Intent canary split is not architecture-blind")
    if payload.get("score_mode") != "mean_token_logprob":
        raise ValueError("Intent canary did not use mean-token choice scoring")
    if payload.get("maximum_input_tokens") != 1024:
        raise ValueError("Intent canary did not bind max_length=1024")
    architecture = str(payload.get("architecture"))
    if payload.get("model_interface") != MODEL_INTERFACES.get(architecture):
        raise ValueError("Intent canary model interface changed")
    if set(payload.get("languages", [])) != set(LANGUAGES):
        raise ValueError("Intent canary language mismatch")
    if set(payload.get("prompts", [])) != set(INTENT_PROMPTS):
        raise ValueError("Intent canary prompt mismatch")
    rows = payload.get("rows", [])
    expected_cells = {
        (language, prompt) for language in LANGUAGES for prompt in INTENT_PROMPTS
    }
    counts = Counter((str(row["lang"]), str(row["template_id"])) for row in rows)
    if set(counts) != expected_cells or any(count != 1 for count in counts.values()):
        raise ValueError(f"Intent canary cells are incomplete: {dict(counts)}")
    identities = {
        (str(row["lang"]), str(row["template_id"]), str(row["example_id"]))
        for row in rows
    }
    if len(identities) != len(rows):
        raise ValueError(
            "Intent canary contains duplicate language-prompt-example rows"
        )
    if any(len(str(row.get("rendered_prompt_sha256", ""))) != 64 for row in rows):
        raise ValueError("Intent canary is missing rendered-prompt hashes")
    frozen = {
        language: {
            (int(record["source_index"]), str(record["row_sha256"]))
            for record in values["validation_records"]
        }
        for language, values in manifest["languages"].items()
    }
    for row in rows:
        identity = (
            int(row["validation_source_index"]),
            str(row["validation_row_sha256"]),
        )
        if identity not in frozen[str(row["lang"])]:
            raise ValueError("Intent canary row is outside the frozen train split")
    return {
        "cell_count": len(counts),
        "sample_count": len(rows),
        "validation_manifest_sha256": manifest_sha256,
        "held_out_data_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lm", type=Path, required=True)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--intent-validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json.tmp").exists():
        raise FileExistsError(args.output)
    lm_payload = json.loads(args.lm.read_text())
    intent_payload = json.loads(args.intent.read_text())
    manifest = json.loads(args.intent_validation_manifest.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("Unexpected frozen Intent validation manifest")
    if manifest.get("held_out_data_accessed") is not False:
        raise ValueError("Frozen Intent validation manifest accessed held-out data")
    manifest_sha256 = sha256_file(args.intent_validation_manifest)
    verified = {
        "schema": "sallm.general_prompt_canary_verified/v2",
        "status": "VERIFIED",
        "held_out_access": False,
        "lm": verify_lm(lm_payload),
        "intent": verify_intent(intent_payload, manifest, manifest_sha256),
        "artifacts": {
            str(args.lm): sha256_file(args.lm),
            str(args.intent): sha256_file(args.intent),
        },
    }
    encoded = json.dumps(verified, indent=2, sort_keys=True) + "\n"
    temp = args.output.with_suffix(".json.tmp")
    temp.write_text(encoded)
    temp.replace(args.output)
    print(f"GENERAL_PROMPT_CANARY_VERIFIED {args.output}")


if __name__ == "__main__":
    main()
