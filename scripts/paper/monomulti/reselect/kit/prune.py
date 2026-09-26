#!/usr/bin/env python3
"""Post-selection cleanup of one label: delete the non-selected checkpoint-*/ directories.

Preconditions (else refuses): val/<label>/selection.json with training_complete, test/<label>.json scored on that
selected checkpoint (not an --adapter override), and the selected checkpoint's tree sha256 still equal to the one
recorded by the test. Before deleting, every checkpoint's trainer_state.json is copied to val/<label>/trainer_states/.
Kept: the selected checkpoint, and final_adapter's non-weight files; final_adapter's weights are deleted only when they
differ from the selected checkpoint's. Dry run by default; --yes deletes. Writes val/<label>/PRUNED.json."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from reselect import TEST_ROOT, TRAIN_ROOT, VAL_ROOT, checkpoints, sha, tree_sha256, write_json

WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")


def weight_digest(d: Path) -> str | None:
    files = [d / w for w in WEIGHTS if (d / w).is_file()]
    return hashlib.sha256("".join(f"{f.name}:{sha(f)}" for f in files).encode()).hexdigest() if files else None


def remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    label = args.label
    sel_path, test_path = VAL_ROOT / label / "selection.json", TEST_ROOT / f"{label}.json"
    if not sel_path.is_file() or not test_path.is_file():
        raise SystemExit(f"{label}: needs {sel_path} and {test_path}")
    selection, test = json.loads(sel_path.read_text()), json.loads(test_path.read_text())
    selected = Path(selection["selected"]["adapter"])
    if not selection.get("training_complete") or test.get("adapter_override") or Path(test["adapter"]) != selected:
        raise SystemExit(f"{label}: selection incomplete or test not on the selected checkpoint")
    if tree_sha256(selected) != test["adapter_tree_sha256"]:
        raise SystemExit(f"{label}: selected checkpoint changed since the test run")
    ckpts = checkpoints(label)
    scored = {r["checkpoint"] for r in selection["scores_by_epoch"]}
    if {c.name for _, c in ckpts} != scored:
        raise SystemExit(f"{label}: checkpoints on disk differ from the scored ones: {sorted({c.name for _, c in ckpts} ^ scored)}")

    states = VAL_ROOT / label / "trainer_states"
    plan = {"label": label, "selected": str(selected), "dry_run": not args.yes, "trainer_states": {}, "deleted": [], "kept": [str(selected)]}
    for _, ckpt in ckpts:
        src = ckpt / "trainer_state.json"
        if src.is_file():
            dst = states / f"{ckpt.name}.trainer_state.json"
            if args.yes:
                states.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                assert sha(dst) == sha(src)
            plan["trainer_states"][ckpt.name] = str(dst)
        if ckpt != selected:
            plan["deleted"].append(str(ckpt))
    final = TRAIN_ROOT / label / "final_adapter"
    final_same = weight_digest(final) == weight_digest(selected)
    plan["final_adapter_weights_equal_selected"] = final_same
    if final.exists() and not final_same:
        plan["deleted"] += [str(final / w) for w in WEIGHTS if (final / w).is_file()]
    if args.yes:
        for p in plan["deleted"]:
            remove(Path(p))
        plan["pruned_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_json(VAL_ROOT / label / "PRUNED.json", plan)
    print(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
