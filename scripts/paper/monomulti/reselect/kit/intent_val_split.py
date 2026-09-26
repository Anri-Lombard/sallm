#!/usr/bin/env python3
"""Rebuild the InjongoIntent validation rows that the sallm training loader holds out of training.

The training loader (sallm/data/loaders/huggingface.py::_load_injongointent_dataset, identical in the
pure-gdn-t2x-official-test-20260901-v2, downstream-generation-20260914-v8 and repo snapshots) does, per language:
  train = {**row, "lang": lang} for train.jsonl ; test = same for test.jsonl
  train = exclude_heldout_texts(train, test)
  train_rows, val_rows = split_injongointent_rows(train)      # 10% per intent, md5-ordered keys
and fine-tunes on train_rows only. This script replays exactly that with the snapshot's own functions on the
pinned files (masakhane/InjongoIntent fe4be388, byte-identical to resolve/main on 2026-09-25), then drops any
validation row whose normalized text also occurs in train_rows, so the rows are disjoint by id and by text.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from datasets import Dataset
from sallm.data.loaders.injongointent_split import _normalized_text, exclude_heldout_texts, split_injongointent_rows

SNAP = Path("/scratch/lmbanr001/hf-cache/hub/datasets--masakhane--InjongoIntent/snapshots/fe4be3882a1614161dfe231ec793197bb74f4b44")
LANGS = ("eng", "sot", "xho", "zul")


def rows(path: Path, lang: str) -> list[dict]:
    raw = [{**json.loads(line), "lang": lang} for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return Dataset.from_list(raw).to_list()


def dedup_first(rs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rs:
        key = (str(r.get("intent", "")).strip(), _normalized_text(r.get("text")))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "reselect.injongointent_trainheldout_validation/v1", "dataset": "masakhane/InjongoIntent",
                "revision": SNAP.name, "loader": "sallm.data.loaders.huggingface._load_injongointent_dataset (replayed)",
                "injongointent_split_sha256": hashlib.sha256(Path(sys.modules["sallm.data.loaders.injongointent_split"].__file__).read_bytes()).hexdigest(),
                "languages": {}}
    for lang in LANGS:
        train_file, test_file = SNAP / lang / "train.jsonl", SNAP / lang / "test.jsonl"
        train, test = rows(train_file, lang), rows(test_file, lang)
        kept = Dataset.from_list(exclude_heldout_texts(train, test)).to_list()
        train_rows, val_rows = split_injongointent_rows(kept)
        train_ids = {(r["intent"], r["example_id"]) for r in train_rows}
        train_texts = {_normalized_text(r["text"]) for r in train_rows}
        test_texts = {_normalized_text(r["text"]) for r in test}
        assert not train_ids & {(r["intent"], r["example_id"]) for r in val_rows}
        final = [r for r in val_rows if _normalized_text(r["text"]) not in train_texts]
        assert not {_normalized_text(r["text"]) for r in final} & (train_texts | test_texts)
        general_train, general_val = split_injongointent_rows(dedup_first(train))
        gv = {(r["intent"], r["example_id"]) for r in general_val}
        lines = [json.dumps({"example_id": r["example_id"], "intent": r["intent"], "text": r["text"], "lang": lang}, ensure_ascii=False) for r in final]
        path = out_dir / f"{lang}.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest["languages"][lang] = {
            "train_file_sha256": hashlib.sha256(train_file.read_bytes()).hexdigest(),
            "test_file_sha256": hashlib.sha256(test_file.read_bytes()).hexdigest(),
            "raw_train_rows": len(train), "test_rows": len(test),
            "train_rows_after_test_text_exclusion": len(kept),
            "loader_train_rows": len(train_rows), "loader_validation_rows": len(val_rows),
            "validation_rows_dropped_text_in_train": len(val_rows) - len(final),
            "validation_rows": len(final), "intents": len({r["intent"] for r in final}),
            "intent_example_id_overlap_with_train": 0, "text_overlap_with_train_or_test": 0,
            "overlap_with_general_prompt_selection_validation_ids": len(gv & {(r["intent"], r["example_id"]) for r in final}),
            "general_prompt_selection_validation_rows": len(gv),
            "file": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("loader_validation_rows", "validation_rows", "validation_rows_dropped_text_in_train", "overlap_with_general_prompt_selection_validation_ids", "general_prompt_selection_validation_rows")} for k, v in manifest["languages"].items()}, indent=1))


if __name__ == "__main__":
    main()
