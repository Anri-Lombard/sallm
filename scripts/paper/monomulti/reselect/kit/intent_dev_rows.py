#!/usr/bin/env python3
"""Validation rows for the Feb-era InjongoIntent Mono retrains, which train on the full upstream train split:
the upstream dev.jsonl at the pinned revision fe4be388 (sot/xho/zul only; eng has no dev file)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SNAP = Path("/scratch/lmbanr001/hf-cache/hub/datasets--masakhane--InjongoIntent/snapshots/fe4be3882a1614161dfe231ec793197bb74f4b44")
DEV_SHA = {"sot": "6da3e561d6b8b5d15aac11ed6d8e86fd13cdf44adcadf7de809d13dad425d71e",
           "xho": "58cd6eb7a4f4ae899e926f8b4bfb9c53d5028a3de99e5e8d93866cb1e8d9189e",
           "zul": "3df8a98fbde6f143d6874c5666d0b308d7a22b9888716527b0b19b6c350caaa9"}


def norm(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "reselect.injongointent_upstream_dev_validation/v1", "dataset": "masakhane/InjongoIntent", "revision": SNAP.name,
                "used_for": "mamba2/mzansilm intent Mono sot/xho/zul (Feb recipe cf0ed06 trains on the full upstream train split)", "languages": {}}
    for lang, expected in DEV_SHA.items():
        dev_file = SNAP / lang / "dev.jsonl"
        assert hashlib.sha256(dev_file.read_bytes()).hexdigest() == expected, dev_file
        dev, train, test = load(dev_file), load(SNAP / lang / "train.jsonl"), load(SNAP / lang / "test.jsonl")
        train_texts, test_texts = {norm(r["text"]) for r in train}, {norm(r["text"]) for r in test}
        path = out_dir / f"{lang}.jsonl"
        path.write_text("".join(json.dumps({"example_id": r["example_id"], "intent": r["intent"], "text": r["text"], "lang": lang}, ensure_ascii=False) + "\n"
                                for r in dev), encoding="utf-8")
        manifest["languages"][lang] = {"source_file": str(dev_file), "source_sha256": expected, "rows": len(dev),
                                       "intents": len({r["intent"] for r in dev}),
                                       "text_overlap_with_upstream_train": sum(norm(r["text"]) in train_texts for r in dev),
                                       "text_overlap_with_test": sum(norm(r["text"]) in test_texts for r in dev),
                                       "file": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("rows", "intents", "text_overlap_with_upstream_train", "text_overlap_with_test")} for k, v in manifest["languages"].items()}))


if __name__ == "__main__":
    main()
