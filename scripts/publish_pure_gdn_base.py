#!/usr/bin/env python3
"""Publish and verify the canonical pure-GDN base model on Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()

    files = sorted(path for path in args.model_dir.iterdir() if path.is_file())
    if not files:
        raise SystemExit(f"No files found in {args.model_dir}")
    manifest = {path.name: sha256(path) for path in files}

    token = args.token_file.read_text().strip() if args.token_file else None
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=args.model_dir,
        commit_message="Publish verified pure-GDN 125M base model",
    )

    card = """---
library_name: transformers
license: apache-2.0
tags:
- gated-deltanet
- causal-lm
- sallm
---

# SALLM pure Gated DeltaNet 125M

Pure FLA `GatedDeltaNetForCausalLM` with `attn=None` and 127,425,448
parameters. It was pretrained for exactly 67,498 optimizer steps with global
sequence batch 48 and context 2,048: 6,635,323,392 token slots, the
xLSTM-matched three-epoch-equivalent budget (not three literal dataset epochs).

Final validation loss: `3.421191930770874`.

The repository files are verified against `artifact_sha256.json` after a fresh
Hub snapshot download. Historical SALLM Qwen3Next results are GDN-Attention
Hybrid and are not this pure-GDN model.
"""
    api.upload_file(
        path_or_fileobj=io.BytesIO(card.encode()),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add pure-GDN model card",
    )
    api.upload_file(
        path_or_fileobj=io.BytesIO(
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        ),
        path_in_repo="artifact_sha256.json",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add canonical artifact hashes",
    )

    with tempfile.TemporaryDirectory(prefix="pure-gdn-hub-verify-") as temp:
        snapshot = Path(
            snapshot_download(
                args.repo_id,
                repo_type="model",
                local_dir=temp,
                allow_patterns=list(manifest),
                force_download=True,
                token=token,
            )
        )
        remote = {name: sha256(snapshot / name) for name in manifest}
    if remote != manifest:
        raise SystemExit(f"Hub hash mismatch: local={manifest}, remote={remote}")
    print(json.dumps({"repo_id": args.repo_id, "sha256": manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
