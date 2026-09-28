#!/usr/bin/env python3
"""SIB-200 for siSwati, Setswana and Xitsonga, which the sealed six-language protocol leaves out (user decision 28 Sep 2026).

  sib_ext.py fetch              CPU (ada): cache Davlan/sib200 {ssw,tsn,tso}_Latn at the protocol revision and write
                                ROLLOUT/assets/sib_ext_protocol.json (copy it to sib_ext/protocol.json in the repo)
  sib_ext.py eval ARCH          GPU (l40s): score the architecture's selected Multilingual SIB checkpoint and its Multitask
                                checkpoint on the three languages. Per (checkpoint, language) the prompt (1-5) is chosen on
                                validation (ties -> lower prompt), then test is scored once with it. Cross-lingual: none of
                                these checkpoints trained on ssw/tsn/tso.
Output: runs/<arch>/ext/sib_ext.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rollout  # noqa: E402

LANGS = ("ssw", "tsn", "tso")
REVISION = "38977a667f6fc264d5c26ec57a01e16db040b358"  # sib_protocol_hex.json test.revision
ROWS = {"train": 701, "validation": 99, "test": 204}
FETCHED = Path(rollout.ROLLOUT) / "assets" / "sib_ext_protocol.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch() -> None:
    from datasets import load_dataset

    cache = Path(rollout.HF_EVAL) / "datasets"
    out = {"revision": REVISION, "source": "Davlan/sib200", "languages": {}}
    for lang in LANGS:
        subset = f"{lang}_Latn"
        ds = load_dataset("Davlan/sib200", subset, revision=REVISION, cache_dir=str(cache))
        assert {k: len(v) for k, v in ds.items()} == ROWS, (lang, {k: len(v) for k, v in ds.items()})
        arrows = {split: cache / "Davlan___sib200" / subset / "0.0.0" / REVISION / f"sib200-{split}.arrow" for split in ("validation", "test")}
        assert all(p.exists() for p in arrows.values()), arrows
        out["languages"][lang] = {"subset": subset, "arrow_sha256": sha(arrows["test"]), "validation_arrow_sha256": sha(arrows["validation"])}
    FETCHED.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def checkpoints(r) -> dict[str, Path]:
    sel = json.loads((r.out / "keep" / "sib" / "SELECTED.json").read_text())
    ckpts = {"multilingual_sib": Path(sel["checkpoint"])}
    general = sorted((r.out / "runs").glob("general-general-*-s42/RUN_DONE.json"))
    if general:
        ckpts["multitask"] = Path(json.loads(general[0].read_text())["kept"])
    return ckpts


def score(r, ckpt: Path, lang: str, split: str, prompt: int, out: Path) -> float:
    raw = out / f"{lang}_{split}_p{prompt}.json"
    if not raw.exists():
        cmd = [r.a["py"], HERE / "runners" / "sib_score.py", "--arch", r.arch, "--base", ckpt, "--split", split,
               "--langs", lang, "--prompt", str(prompt), "--out", raw]
        r.sh(cmd, out / f"{lang}_{split}_p{prompt}.log", r.env())
    return 100 * json.loads(raw.read_text())["languages"][lang]["f1"]


def evaluate(arch: str) -> None:
    r = rollout.Run(Path(rollout.ROLLOUT) / "runs" / arch)
    res = {"arch": arch, "revision": REVISION, "prompt_rule": "max validation F1 over prompts 1-5, ties -> lower prompt",
           "checkpoints": {}}
    for tag, ckpt in checkpoints(r).items():
        out = r.out / "ext" / "sib" / tag
        out.mkdir(parents=True, exist_ok=True)
        per = {}
        for lang in LANGS:
            val = {p: score(r, ckpt, lang, "validation", p, out) for p in range(1, 6)}
            best = max(val, key=lambda p: (val[p], -p))
            per[lang] = {"val_by_prompt": val, "prompt": best, "val": val[best],
                         "test": score(r, ckpt, lang, "test", best, out)}
        res["checkpoints"][tag] = {"checkpoint": str(ckpt), "per_lang": per}
    (r.out / "ext" / "sib_ext.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"fetch": lambda: fetch(), "eval": lambda: evaluate(sys.argv[2])}[sys.argv[1]]()
