#!/usr/bin/env python3
"""Resolve and hash every execution binding without opening evaluation data."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    root = root.resolve()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    )
    if not files:
        raise ValueError(f"Empty binding tree: {root}")
    payload = "".join(
        f"{sha256(path)}  ./{path.relative_to(root).as_posix()}\n" for path in files
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def ref_id(reference: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(reference).encode()).hexdigest()


def resolve(reference: dict[str, Any], api: HfApi) -> dict[str, Any]:
    if reference["kind"] == "none":
        return {"reference": reference, "path": None, "tree_sha256": None, "files": {}}
    resolved_revision = None
    if reference["kind"] == "hub":
        info = api.model_info(reference["repo_id"], revision=reference.get("revision"))
        resolved_revision = info.sha
        path = Path(
            snapshot_download(
                repo_id=reference["repo_id"],
                revision=resolved_revision,
            )
        ).resolve()
    elif reference["kind"] == "path":
        path = Path(reference["path"]).resolve()
    else:
        raise ValueError(f"Unsupported reference kind: {reference}")
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = {}
    for name in (
        "config.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "model.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        candidate = path / name
        if candidate.is_file():
            files[name] = sha256(candidate)
    if not ("config.json" in files or "adapter_config.json" in files):
        raise ValueError(f"Binding has neither model nor adapter config: {path}")
    observed_tree = tree_sha256(path)
    expected_tree = reference.get("expected_tree_sha256")
    if expected_tree and observed_tree != expected_tree:
        raise ValueError(f"Tree hash mismatch for {path}: {observed_tree}")
    expected_weight = reference.get("expected_adapter_model_sha256")
    if expected_weight and files.get("adapter_model.safetensors") != expected_weight:
        raise ValueError(f"Adapter weight hash mismatch for {path}")
    return {
        "reference": reference,
        "path": str(path),
        "resolved_revision": resolved_revision,
        "tree_sha256": observed_tree,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--protocol-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    units = json.loads((args.inventory / "official_units.json").read_text())
    arms = json.loads((args.inventory / "validation_arms.json").read_text())
    refs: dict[str, dict[str, Any]] = {}
    for unit in units:
        binding = unit["binding"]
        for reference in (binding["base"], *binding["adapter_candidates"]):
            refs.setdefault(ref_id(reference), reference)
    for arm in arms:
        for reference in (arm["base"], arm["adapter"]):
            refs.setdefault(ref_id(reference), reference)
    api = HfApi()
    resolved = {key: resolve(reference, api) for key, reference in sorted(refs.items())}
    args.output.mkdir(parents=True, exist_ok=False)
    binding_payload = {
        "schema": "sallm.full_matrix_resolved_bindings/v1",
        "data_boundary": "metadata_and_model_artifacts_only",
        "test_accessed": False,
        "entries": resolved,
    }
    binding_path = args.output / "BINDINGS.json"
    binding_path.write_text(json.dumps(binding_payload, indent=2, sort_keys=True) + "\n")

    template = json.loads(args.protocol_template.read_text())
    sequence_root = args.output / "sequence_protocols"
    sequence_root.mkdir()
    for unit in units:
        if unit["task_group"] not in {"ner", "pos"}:
            continue
        base = resolved[ref_id(unit["binding"]["base"])]
        candidates = unit["binding"]["adapter_candidates"]
        if len(candidates) != 1:
            continue
        adapter = resolved[ref_id(candidates[0])]
        protocol = deepcopy(template)
        architecture = unit["architecture"]
        interface = {
            "mzansilm": {"dtype": "bfloat16", "merge_lora": False, "tie_word_embeddings": None},
            "mamba2": {"dtype": "float32", "merge_lora": False, "tie_word_embeddings": False},
            "xlstm": {"dtype": "float32", "merge_lora": True, "tie_word_embeddings": False},
            "gdn": {"dtype": "bfloat16", "merge_lora": False, "tie_word_embeddings": None},
        }[architecture]
        protocol["models"] = {
            architecture: {
                **interface,
                "execution_status": "runnable",
                "base_path": base["path"],
                "adapter_path": adapter["path"],
                "base_tree_sha256": base["tree_sha256"],
                "adapter_tree_sha256": adapter["tree_sha256"],
                "base_files": base["files"],
                "adapter_files": adapter["files"],
            }
        }
        (sequence_root / f"{unit['unit_id']}.json").write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n"
        )
    marker = {
        "schema": "sallm.full_matrix_binding_preflight/v1",
        "status": "BINDINGS_RESOLVED",
        "bindings_sha256": sha256(binding_path),
        "reference_count": len(resolved),
        "sequence_protocol_count": len(list(sequence_root.glob("*.json"))),
        "held_out_accessed": False,
    }
    (args.output / "BINDINGS_RESOLVED.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(marker, sort_keys=True))


if __name__ == "__main__":
    main()
