#!/usr/bin/env python3
"""Fixed validation subsample used for epoch and LR selection (test scoring is always full size).

At most CAP items per (task, language) validation split, drawn once with random.Random(f"{SEED}:{task}:{lang}")
and stored in val_subsample.json next to this file (identical for all four architectures; its sha256 is recorded in
every score). Splits with <= CAP items stay whole. Indices are positions in the split as the runner loads it (the
sources are commit/hash-pinned); a runner asserts the split size before selecting.

Sizes (26 Sep 2026): News dev eng 472 / xho 147, SIB-200 dev 99 x 6, Intent carved dev 111/240/239/240,
MasakhaNER dev (parquet, pinned revision) tsn 499 / xho 817 / zul 836, MasakhaPOS dev 150 x 3 (450 in total),
T2X validation 460, AfriHG dev xho 1305 / zul 1777. Only NER xho/zul and AfriHG xho/zul are subsampled.
NCHLT (added 28 Sep 2026, entries above unchanged): NER dev nbl 933 / ssw 1080 / ven 848 / tso 972 (cap 500), POS dev
nbl 259 / ssw 256 / ven 260 / tso 249 sentences (cap 100: POS validation scores every token against all 26-32 tags).

  python val_subsample.py          rewrite val_subsample.json and print its sha256 (deterministic)
Runners: indices(task, lang, n) returns the positions to keep, or None when FFT_VAL_SUBSAMPLE != 1 or the split is whole.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

SEED, CAP = 20260926, 500
TASK_CAP = {"nchlt_pos": 100}
MANIFEST = Path(__file__).resolve().parent / "val_subsample.json"
SIZES = {
    "news": {"eng": 472, "xho": 147},
    "sib": dict.fromkeys(("afr", "eng", "nso", "sot", "xho", "zul"), 99),
    "intent": {"eng": 111, "sot": 240, "xho": 239, "zul": 240},
    "ner": {"tsn": 499, "xho": 817, "zul": 836},
    "pos": dict.fromkeys(("tsn", "xho", "zul"), 150),
    "t2x": {"xho": 460},
    "afrihg": {"xho": 1305, "zul": 1777},
    "nchlt_ner": {"nbl": 933, "ssw": 1080, "ven": 848, "tso": 972},
    "nchlt_pos": {"nbl": 259, "ssw": 256, "ven": 260, "tso": 249},
}


def build() -> dict:
    out = {"seed": SEED, "cap": CAP, "rule": "sorted(random.Random(f'{seed}:{task}:{lang}').sample(range(n), cap)) if n > cap else all",
           "splits": {}}
    for task, langs in SIZES.items():
        for lang, n in langs.items():
            cap = TASK_CAP.get(task, CAP)
            idx = sorted(random.Random(f"{SEED}:{task}:{lang}").sample(range(n), cap)) if n > cap else None
            out["splits"].setdefault(task, {})[lang] = {"n": n, "k": len(idx) if idx else n, "idx": idx}
    return out


def manifest_sha256() -> str:
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def indices(task: str, lang: str, n: int) -> list[int] | None:
    if os.environ.get("FFT_VAL_SUBSAMPLE") != "1":
        return None
    entry = json.loads(MANIFEST.read_text())["splits"][task][lang]
    if entry["n"] != n:
        raise RuntimeError(f"validation split {task}/{lang} has {n} items, the subsample manifest says {entry['n']}")
    return entry["idx"]


if __name__ == "__main__":
    m = build()
    assert m == build()  # deterministic
    assert all(e["k"] <= TASK_CAP.get(t, CAP) and (e["idx"] is None or len(set(e["idx"])) == TASK_CAP.get(t, CAP))
               for t, v in m["splits"].items() for e in v.values())
    MANIFEST.write_text(json.dumps(m, indent=None, separators=(",", ":")) + "\n")
    print(MANIFEST, manifest_sha256())
    print({t: {lang: e["k"] for lang, e in v.items()} for t, v in m["splits"].items()})
