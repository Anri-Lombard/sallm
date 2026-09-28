#!/usr/bin/env python3
"""Equal-recipe full fine-tuning rollout for one architecture: Base eval -> LR sweeps -> selection -> Mono/seeds -> test.

One run directory OUT per architecture holds config.json (ARCH, BASE, SMOKE), units.json (the job matrix as a DAG),
state/ (one JSON per started unit), and every artifact. Lane jobs (lane.sbatch, one L40S each) pull ready units until
none are left, so the number of lanes is the concurrency cap. See RUNBOOK.md.

  rollout.py matrix [--arch A] [--smoke] [--csv PATH]   print/write the job matrix with GPU-hour estimates (no cluster)
  rollout.py lane OUT                                    worker loop (inside a Slurm job)
  rollout.py run OUT UNIT                                run one unit now (debugging, inside a Slurm job)
  rollout.py status OUT                                  write OUT/STATUS.txt (lanes also refresh it after every unit)
  rollout.py collect OUT                                 score table with bootstrap CIs -> OUT/results/cells.csv

Protocol (identical for all four architectures; notes sallm_architecture_paper_consistency_2026-09-25.md):
full fine-tuning, AdamW (0.9, 0.95), wd 0.01, cosine, 10% warmup, effective batch 16, clip 1.0, fp32 master + bf16
autocast, label smoothing 0; 10 epochs if the train set has fewer than 5000 examples else 4; every epoch saved and
scored on validation with the paper's protocol scorers; LR grid {3e-5, 1e-4, 3e-4} swept on the Multi model (Mono for
T2X, the General model for General), edge rule (amended 27 Sep): up to 2 extra x3 points beyond an edge optimum, the swept LR reused for Mono; seeds 42/43/44
for General and Mono T2X; test scored once on the selected checkpoint; all scoring on L40S.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

# ----------------------------------------------------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
S = "/scratch/lmbanr001/masters/sallm"
SNAPS = "/scratch/lmbanr001/masters/sallm_snapshots"
ROLLOUT = f"{S}/results/fft_rollout_20260926"
SALLM = Path(os.environ.get("ROLLOUT_SALLM", f"{ROLLOUT}/code/sallm"))  # frozen copy of this repo's src/ + tokenizer/
MAIN_PY = "/home/lmbanr001/masters/sallm/.venv/bin/python"
XLSTM_PY = f"{S}/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python"
GEN = f"{SNAPS}/generation-protocol-v3-greedy-20260924"  # cwd + T2X/AfriHG caches of the paper's generation runs
V8 = f"{SNAPS}/downstream-generation-20260914-v8"  # --source: generation task definitions
BUNDLE = f"{SNAPS}/full-matrix-execution-20260916-v1"
PILOT_T2X = f"{S}/results/equal_recipe_fullft_t2x_pilot_20260925/assets/t2x_train_validation_only"
V9_RUNNER = f"{SNAPS}/full-matrix-targeted-recovery-20260916-v9/.audit/run_train_validation_only_20260914.py"
SEQ_SELECTION = f"{S}/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/selection/SELECTION.json"
SEQ_RELEASE = f"{S}/results/general_sequence_official_test_20260916_v1/release/{{}}_TEST_ACCESS_RELEASED_V1.json"
SEQ_PROTOCOL = f"{SNAPS}/general-sequence-official-test-20260916-v1/general_sequence_validation_hex_protocol_20260915_v3.json"
FM = f"{S}/results/full_matrix_execution_20260916_v1"
BASE_SEQ_PROTOCOL = FM + "/partial-ready-1/bindings/sequence_protocols/u00{}-base-gdn-{}.json"  # (1, ner), (2, pos)
BASE_SEQ_RELEASE = FM + "/partial-ready-1r1/release/{}_TEST_ACCESS_RELEASED.json"
GP = f"{S}/results/mamba_eos_fix_20260924/kombuys_snapshot"
GP_B, GP_S = f"{GP}/general-prompt-official-kombuys-20260915-v1", f"{GP}/general-prompt-correction-20260915-v8-sixfamily-tso-all-interface-fix"
HF_TRAIN, HF_EVAL = "/scratch/lmbanr001/hf", "/scratch/lmbanr001/hf-cache"
TOKENIZER_SHA = "446895905ea9b20c746317eefd0c6a3b097bcbbef71e8e44b0bf9772d664782a"  # tokenizer.json, all four bases

# ------------------------------------------------------------------------------------------------------ protocol
LRS = ("3e-5", "1e-4", "3e-4")
# Edge rule (amended 27 Sep 2026 09:06 SAST, user decision, before any extension result existed): while the best LR is at
# an edge of the evaluated grid, add the next point beyond it on this x3 ladder; at most MAX_EXT extra points per
# (arch, task); still at the edge after that -> best_at_grid_edge=true. (Old rule: one extra point, 3e-5->1e-5, 3e-4->1e-3.)
LADDER = ("3e-6", "1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3")
MAX_EXT = 2


def next_edge_lr(grid: list[str], best_lr: str, n_ext: int) -> str | None:
    """The next LR to train under the edge rule, or None (best interior, extension budget used, or ladder end)."""
    if n_ext >= MAX_EXT or len(grid) < 2 or best_lr not in (grid[0], grid[-1]):
        return None
    i = LADDER.index(best_lr) + (1 if best_lr == grid[-1] else -1)
    return LADDER[i] if 0 <= i < len(LADDER) and LADDER[i] not in grid else None
SEEDS_HEADLINE = (43, 44)  # plus seed 42 (the sweep run)
# Early stopping (user decision 26 Sep): stop once PATIENCE epochs pass without a strictly greater validation score
# (protocol scorer, fixed subsample; General: six-family mean). Minimum epochs = PATIENCE + 1. Best epoch as before.
PATIENCE = int(os.environ.get("FFT_ES_PATIENCE", "3"))
TRANSIENT_MAX = 4  # lane deaths (stale heartbeat) a unit may survive; each resumes from its last epoch checkpoint


class Diverged(RuntimeError):
    """Training diverged (non-finite loss, or > 3x the first-epoch loss for 200 steps): FAILED_DIVERGED, never retried."""
# Beam search (test only, reported beside greedy; greedy stays primary): the v1 settings of the V8 generation config
# (T2X 5 beams, length penalty 1.0; AfriHG 5 beams, length penalty 0.7; early stopping). Transformers 4.57.3 reorders
# beams through Cache.reorder_cache: FLA 0.5.1's cache layers keep their state in `.state`, so the reorder hits
# `.keys = None` (GDN, Mamba-2: AttributeError), and HF xLSTM keeps `cache_params`, which beam search never reorders
# (silently wrong beams). Those decode without the generation cache (exact, slower), Mamba-2 included.
BEAM_MODE = {"mzansilm": "cache", "gdn": "nocache", "xlstm": "nocache",
             "mamba2": "nocache"}  # user decision 26 Sep: Mamba-2 beam runs cache-free like GDN
BEAM_FACTOR = {"cache": 3.0, "nocache": 12.0}  # beam s/row over greedy s/row; provisional until the beam canary measures it
ARCHS = {
    #          Hydra model key     python     eval interface (dtype, merge_lora, tie_word_embeddings)
    "mzansilm": dict(key="llama", py=MAIN_PY, dtype="bfloat16", merge=False, tie=None),
    "mamba2": dict(key="mamba2", py=MAIN_PY, dtype="float32", merge=False, tie=False),
    "xlstm": dict(key="xlstm", py=XLSTM_PY, dtype="float32", merge=True, tie=False),
    "gdn": dict(key="gated_deltanet", py=MAIN_PY, dtype="bfloat16", merge=False, tie=None),
}
PAPER_MODEL = {"mzansilm": "MzansiLM", "mamba2": "Mamba", "xlstm": "xLSTM", "gdn": "GDN"}
# sweep regime, languages, Hydra config (Mono config gets {lang}), approximate train rows (Mono, Multi) for estimates.
FAMILIES = {
    "news": dict(sweep="multi", langs=("eng", "xho"), mono="llama_news_{}", multi="llama_news_all",
                 rows={"eng": 3309, "xho": 1032}),
    "sib": dict(sweep="multi", langs=("afr", "eng", "nso", "sot", "xho", "zul"), mono="llama_sib_{}", multi="llama_sib_all",
                rows=dict.fromkeys(("afr", "eng", "nso", "sot", "xho", "zul"), 701)),
    "intent": dict(sweep="multi", langs=("eng", "sot", "xho", "zul"), mono="llama_injongointent_{}", multi="llama_injongointent_all",
                   rows={"eng": 1046, "sot": 2000, "xho": 2000, "zul": 2000}),
    "ner": dict(sweep="multi", langs=("tsn", "xho", "zul"), mono="llama_ner_{}", multi="llama_ner_all",
                rows=dict.fromkeys(("tsn", "xho", "zul"), 1441)),  # Multi 4323 (smoke)
    "pos": dict(sweep="multi", langs=("tsn", "xho", "zul"), mono="llama_pos_{}", multi="llama_pos_all",
                rows={"tsn": 754, "xho": 752, "zul": 753}),
    "afrihg": dict(sweep="multi", langs=("xho", "zul"), mono="llama_afrihg_{}", multi="llama_afrihg_all",
                   rows={"xho": 12300, "zul": 12349}),
    "t2x": dict(sweep="mono", langs=("xho",), mono="llama_t2x_{}", multi=None, rows={"xho": 3859}),
    "general": dict(sweep="general", langs=(), mono=None, multi="llama_sa_general_examplesprop_k3000", rows={"all": 43637}),
    # NCHLT (added 28 Sep 2026) for the four languages no other task covers. Not part of the General mixture.
    # NER train is a seeded 2,000-sentence subset per language (anrilombard/nchlt-ner-sa4 `train`); POS is the full train.
    "nchlt_ner": dict(sweep="multi", langs=("nbl", "ssw", "ven", "tso"), mono="llama_nchlt_ner_{}", multi="llama_nchlt_ner_all",
                      rows=dict.fromkeys(("nbl", "ssw", "ven", "tso"), 2000)),
    "nchlt_pos": dict(sweep="multi", langs=("nbl", "ssw", "ven", "tso"), mono="llama_nchlt_pos_{}", multi="llama_nchlt_pos_all",
                      rows={"nbl": 2329, "ssw": 2307, "ven": 2344, "tso": 2245}),
}
# Per-device micro-batch (x gradient accumulation = effective batch 16), same for all four architectures. Families
# whose examples reach 2048 tokens (News, AfriHG, the General mix) or 1311 (POS) run out of L40S memory at 16 x 1
# (smoke: General OOM on the 16 x 2048 x 65539 fp32 logits; POS 28 GB after 4 steps).
MICRO = {"news": 4, "afrihg": 4, "general": 4, "pos": 8, "nchlt_pos": 8}
GENERAL_TRAIN_FAMILIES = ("news", "sib", "ner", "pos", "t2x", "afrihg")  # the six-family General mixture
LANG_FAMILY = {**dict.fromkeys(("zul", "xho", "ssw", "nbl"), "Nguni"), **dict.fromkeys(("sot", "tsn", "nso"), "Sotho-Tswana"),
               "afr": "afr", "eng": "eng", "ven": "ven", "tso": "tso"}
PAPER_TASK = {"news": "News", "sib": "SIB-200", "intent": "Intent", "ner": "NER", "pos": "POS", "t2x": "T2X",
              "afrihg": "AfriHG", "belebele": "Belebele", "afrixnli": "AfriXNLI", "afrimmlu": "AfriMMLU", "afrimgsm": "AfriMGSM",
              "nchlt_ner": "NCHLT NER", "nchlt_pos": "NCHLT POS"}
METRIC = {"news": "support_weighted_f1", "sib": "support_weighted_f1", "intent": "support_weighted_f1",
          "ner": "entity_span_f1", "pos": "token_accuracy", "t2x": "chrf", "afrihg": "chrf", "belebele": "acc_norm",
          "afrixnli": "accuracy", "afrimmlu": "accuracy", "afrimgsm": "flexible_exact_match",
          "nchlt_ner": "entity_span_f1", "nchlt_pos": "token_accuracy"}


def seq_kind(family: str) -> str:
    """'ner' / 'pos' for the MasakhaNER-X / MasakhaPOS families and their NCHLT counterparts, else the family."""
    return {"nchlt_ner": "ner", "nchlt_pos": "pos"}.get(family, family)

# Smoke test: one tiny cell per task type on a stand-in base, <= 2 lanes, few steps, ~20 scored items.
SMOKE = dict(families={"sib": ("afr",), "ner": ("tsn",), "pos": ("tsn",), "t2x": ("xho",), "general": (),
                       "nchlt_ner": ("ven",), "nchlt_pos": ("tso",)},
             lrs={"t2x": LRS}, default_lrs=("1e-4",), seeds=(43,), max_steps=4, val_limit=20, test_limit=20)

# ---------------------------------------------------------------------------------------------------- estimates
# Pilot (T2X, L40S, batch 16, ~1630 non-pad tokens/step): s/step and s/generated val row; step-time growth per extra
# 1630 tokens/step from the pilot's batch-32 runs (gdn noisy, rounded up).
STEP_S = {"mzansilm": 0.645, "mamba2": 0.361, "xlstm": 0.755, "gdn": 0.975}
STEP_GROWTH = {"mzansilm": 0.07, "mamba2": 0.34, "xlstm": 0.13, "gdn": 0.5}
# tokens per training row: measured by the smoke count (news 739, sib 130, intent 237, pos 532, afrihg 554, t2x 102)
TOK_PER_ROW = {"news": 739, "sib": 130, "intent": 237, "ner": 177, "pos": 532, "afrihg": 554, "t2x": 102, "general": 450,
               "nchlt_ner": 220, "nchlt_pos": 480}  # NCHLT: estimates, not measured
GEN_S_PER_ROW = {"mzansilm": 0.076, "mamba2": 0.35, "xlstm": 0.28, "gdn": 0.26}  # pilot T2X val (460 rows)
# Scoring minutes per split for all of a family's languages on one L40S (reselect/rescore logs, pilot; POS/NER from
# the General runs: POS test 45/124/108/97 min, NER test 11/15/11/13 min for mzansilm/mamba2/xlstm/gdn).
POS_TEST_MIN = {"mzansilm": 45, "mamba2": 124, "xlstm": 108, "gdn": 97}
NER_TEST_MIN = {"mzansilm": 11, "mamba2": 15, "xlstm": 25, "gdn": 13}
VAL_ROWS = {"t2x": 460, "afrihg": 2 * 500}  # fixed selection subsample (val_subsample.json): AfriHG dev 1305/1777 -> 500 each
TEST_ROWS = {"t2x": 378, "afrihg": 1305 + 1776}


def score_minutes(arch: str, family: str, split: str, n_langs_frac: float = 1.0) -> float:
    if family in ("t2x", "afrihg"):
        rows = (VAL_ROWS if split == "val" else TEST_ROWS)[family]
        per = GEN_S_PER_ROW[arch] * (1.5 if family == "afrihg" else 1.0)
        return (rows * per / 60 + 1.5) * n_langs_frac
    base = {"news": 4 if split == "val" else 5, "sib": 1.5 if split == "val" else 2.5, "intent": 3 if split == "val" else 8,
            "ner": NER_TEST_MIN[arch] * (1499 / 2152 if split == "val" else 1.0),  # val subsample 499+500+500 of 2152
            "pos": POS_TEST_MIN[arch] * (0.4 if split == "val" else 1.0),
            "belebele": 15, "transfer": 45 if arch != "xlstm" else 120,
            # NCHLT estimates: NER 3,832 test / 2,000 val rows vs MasakhaNER's 2,152; POS ~23k test tokens x ~29 tags
            "nchlt_ner": NER_TEST_MIN[arch] * ((2000 if split == "val" else 3832) / 2152),
            "nchlt_pos": POS_TEST_MIN[arch] * (0.4 if split == "val" else 1.0)}[family]
    slow = 1.6 if arch in ("mamba2", "xlstm") and family in ("news", "sib", "intent", "belebele") else 1.0
    return (base * slow + 1.0) * n_langs_frac


def train_hours(arch: str, family: str, rows: int, epochs: int) -> float:
    micro = MICRO.get(family, 16)
    tokens_per_micro = micro * TOK_PER_ROW[family]
    step = (16 // micro) * STEP_S[arch] * (1 + STEP_GROWTH[arch] * max(0.0, tokens_per_micro / 1630 - 1))
    return math.ceil(rows / 16) * epochs * step / 3600


# Best epochs seen in the T2X pilot (Mono xho, lr 1e-4); elsewhere assume best epoch 4 -> early stop at 4 + PATIENCE.
PILOT_BEST_EPOCH = {("mzansilm", "t2x"): 3, ("mamba2", "t2x"): 3, ("xlstm", "t2x"): 2, ("gdn", "t2x"): 4}


def expected_epochs(arch: str, family: str, planned: int) -> int:
    """Estimate only: epochs a run is expected to train under early stopping."""
    return min(planned, max(PATIENCE + 1, PILOT_BEST_EPOCH.get((arch, family), 4) + PATIENCE))


def epochs_for(rows: int) -> int:
    return 10 if rows < 5000 else 4


# ------------------------------------------------------------------------------------------------------ the DAG
def unit(uid, kind, deps=(), **kw):
    return {"id": uid, "kind": kind, "deps": list(deps), **kw}


def plan(arch: str, smoke: bool, cross_eval: bool = False, only: list[str] | None = None) -> list[dict]:
    units = [unit("prep", "prep")]
    if smoke and not only:
        units.append(unit("count", "count", ["prep"]))
    for group in ("gen", "prompt", "ner", "pos"):
        units.append(unit(f"base-{group}", "base", ["prep"], group=group))
    families = SMOKE["families"] if smoke else {f: FAMILIES[f]["langs"] for f in FAMILIES}
    if only:  # smoke of selected families only (e.g. the per-architecture T2X smoke)
        families = {f: v for f, v in families.items() if f in only}
    for fam, mono_langs in families.items():
        if fam == "general":
            continue  # one General model, below
        spec = FAMILIES[fam]
        lrs = (SMOKE["lrs"].get(fam, SMOKE["default_lrs"]) if smoke and not only else SMOKE["default_lrs"] if smoke else LRS)
        sweep_regime = spec["sweep"]
        sweep_langs = list(spec["langs"]) if sweep_regime != "general" else []
        tag = {"multi": "multi", "mono": "mono-" + "-".join(spec["langs"]), "general": "general"}[sweep_regime]
        sweep_ids = []
        for lr in lrs:
            uid = f"train-{fam}-{tag}-lr{lr}-s42"
            sweep_ids.append(uid)
            units.append(unit(uid, "train", ["prep"], family=fam, regime=sweep_regime, langs=sweep_langs, lr=lr, seed=42,
                              keep=True, test=False))
        units.append(unit(f"select-{fam}", "select", sweep_ids, family=fam, regime=sweep_regime, langs=sweep_langs,
                          lrs=list(lrs), tag=tag, edge=not smoke or fam == "t2x"))
        units.append(unit(f"test-{fam}-{tag}", "test", [f"select-{fam}"], family=fam, regime=sweep_regime, langs=sweep_langs))
        if sweep_regime == "multi":
            for lang in mono_langs:
                units.append(unit(f"train-{fam}-mono-{lang}-s42", "train", [f"select-{fam}"], family=fam, regime="mono",
                                  langs=[lang], lr=None, seed=42, keep=False, test=True))
        if fam == "t2x":
            for seed in (SMOKE["seeds"] if smoke else SEEDS_HEADLINE):
                units.append(unit(f"train-{fam}-{tag}-s{seed}", "train", [f"select-{fam}"], family=fam, regime=sweep_regime,
                                  langs=sweep_langs, lr=None, seed=seed, keep=False, test=True))
    if "general" in families:
        # General, trimmed (user decision, 26 Sep): no LR sweep and one seed. ONE model (seed 42) at the LR chosen most
        # often across this architecture's Multi selections (ties -> lower LR); per-epoch validation and epoch selection
        # on the six-family mean; test in-unit; beam below.
        multis = [f for f in families if FAMILIES[f]["sweep"] == "multi"]
        units.append(unit("train-general-general-s42", "train", ["prep"] + [f"select-{f}" for f in multis], family="general",
                          regime="general", langs=[], lr=None, seed=42, keep=False, test=True, lr_from=multis))
    units.append(unit("collect", "collect", [u["id"] for u in units if u["kind"] != "count"], light=True))
    beams = beam_units(families, smoke)  # after collect in the DAG: a failed beam unit never blocks the main results
    units += beams + [unit("collect-beam", "collect", [u["id"] for u in beams] + ["collect"], light=True)] if beams else []
    if cross_eval:
        units += cross_eval_units(families)
    for u in units:
        u["est_hours"] = round(estimate(arch, u, smoke), 3)
    return units


def beam_units(families: dict, smoke: bool) -> list[dict]:
    """Test-only beam decoding of every selected generation checkpoint (T2X, AfriHG, and the General model's two)."""
    out = []
    seeds = (42, *(SMOKE["seeds"] if smoke else SEEDS_HEADLINE))
    if "t2x" in families:
        tag = "mono-" + "-".join(FAMILIES["t2x"]["langs"])
        for seed in seeds:
            src = f"test-t2x-{tag}" if seed == 42 else f"train-t2x-{tag}-s{seed}"
            out.append(unit(f"beam-t2x-{tag}-s{seed}", "beam", [src], src=src, family="t2x", gen={"t2x": ["xho"]}, seed=seed))
    if "afrihg" in families:
        out.append(unit("beam-afrihg-multi", "beam", ["test-afrihg-multi"], src="test-afrihg-multi", family="afrihg",
                        gen={"afrihg": list(FAMILIES["afrihg"]["langs"])}, seed=42))
        for lang in families["afrihg"]:
            src = f"train-afrihg-mono-{lang}-s42"
            out.append(unit(f"beam-afrihg-mono-{lang}", "beam", [src], src=src, family="afrihg", gen={"afrihg": [lang]}, seed=42))
    if "general" in families:  # seed 42 only
        out.append(unit("beam-general-s42", "beam", ["train-general-general-s42"], src="train-general-general-s42", family="general",
                        gen={"t2x": ["xho"], "afrihg": list(FAMILIES["afrihg"]["langs"])}, seed=42))
    return out


def cross_eval_units(families: dict) -> list[dict]:
    """Optional (off by default): each Mono model's selected checkpoint on the test sets of its task's other languages."""
    out = []
    for fam, mono_langs in families.items():
        langs = FAMILIES[fam]["langs"]
        if FAMILIES[fam]["sweep"] != "multi" or len(langs) < 2:
            continue
        for lang in mono_langs:
            out.append(unit(f"xeval-{fam}-{lang}", "xeval", [f"train-{fam}-mono-{lang}-s42"], family=fam, regime="mono",
                            langs=[lang], targets=[x for x in langs if x != lang], seed=42, optional=True))
    out.append(unit("collect-xeval", "collect", [u["id"] for u in out] + ["collect"], light=True, optional=True))
    return out


def rows_for(u: dict) -> int:
    rows = FAMILIES[u["family"]]["rows"]
    return sum(rows.values()) if u["regime"] in ("multi", "general") else rows[u["langs"][0]]


def general_val_minutes(arch: str) -> float:
    return sum(score_minutes(arch, f, "val") for f in GENERAL_TRAIN_FAMILIES)


def general_test_minutes(arch: str) -> float:
    return sum(score_minutes(arch, f, "test") for f in GENERAL_TRAIN_FAMILIES + ("intent", "belebele", "transfer"))


def family_minutes(arch: str, u: dict, split: str) -> float:
    if u["family"] == "general":
        return general_val_minutes(arch) if split == "val" else general_test_minutes(arch)
    frac = len(u["langs"]) / len(FAMILIES[u["family"]]["langs"])
    return score_minutes(arch, u["family"], split, frac)


def estimate(arch: str, u: dict, smoke: bool) -> float:
    if smoke:
        return {"prep": 0.05, "count": 0.3, "base": 0.15, "train": 0.2, "select": 0.05, "test": 0.2, "collect": 0.02, "xeval": 0.1, "beam": 0.1}[u["kind"]]
    if u["kind"] == "prep":
        return 0.1
    if u["kind"] == "base":
        return {"gen": sum(TEST_ROWS.values()) * GEN_S_PER_ROW[arch] / 3600 + 0.1,
                "prompt": 1.5 if arch != "xlstm" else 4.0,
                "ner": NER_TEST_MIN[arch] / 60 * (3 if arch == "xlstm" else 1),
                "pos": POS_TEST_MIN[arch] / 60}[u["group"]]
    if u["kind"] == "train":
        rows = rows_for(u)
        ep = expected_epochs(arch, u["family"], epochs_for(rows))
        h = train_hours(arch, u["family"], rows, ep) + ep * family_minutes(arch, u, "val") / 60 + 0.1
        return h + (family_minutes(arch, u, "test") / 60 if u["test"] else 0)
    if u["kind"] == "select":  # expected cost: one extension about half the time, a second about a quarter
        probe = {"kind": "train", "family": u["family"], "regime": u["regime"], "langs": u["langs"], "test": False}
        return 0.75 * estimate(arch, probe, smoke) if u["edge"] else 0.01
    if u["kind"] == "test":
        return family_minutes(arch, u, "test") / 60 + 0.05
    if u["kind"] == "beam":
        mode = BEAM_MODE[arch]
        if mode == "unsupported":
            return 0.01
        rows = sum(TEST_ROWS["t2x"] if f == "t2x" else TEST_ROWS["afrihg"] * len(ls) / 2 for f, ls in u["gen"].items())
        return rows * GEN_S_PER_ROW[arch] * BEAM_FACTOR[mode] / 3600 + 0.05
    if u["kind"] == "xeval":
        return score_minutes(arch, u["family"], "test", len(u["targets"]) / len(FAMILIES[u["family"]]["langs"])) / 60 + 0.05
    return 0.05


def cmd_matrix(args) -> None:
    archs = [args.arch] if args.arch else list(ARCHS)
    rows = []
    for arch in archs:
        for u in plan(arch, args.smoke, cross_eval=True):
            lr = u.get("lr") or ("selected" if u["kind"] == "train" else "")
            ep = epochs_for(rows_for(u)) if u["kind"] == "train" else ""
            rows.append({"arch": arch, "unit": u["id"], "kind": u["kind"], "stage": stage_of(u), "family": u.get("family", u.get("group", "")),
                         "regime": u.get("regime", "Base" if u["kind"] == "base" else ""), "languages": "+".join(u.get("langs", [])) or "+".join(f"{f}:{l}" for f, ls in u.get("gen", {}).items() for l in ls),
                         "lr": lr, "seed": u.get("seed", ""), "epochs": ep, "train_rows_approx": rows_for(u) if u["kind"] == "train" else "",
                         "val_scoring_passes": ep, "test_after_training": u.get("test", ""), "depends_on": " ".join(u["deps"]),
                         "est_gpu_hours": u["est_hours"], "optional": bool(u.get("optional")),
                         "cross_eval_targets": "+".join(u.get("targets", []))})
    if args.simulate:
        dags = {a: plan(a, args.smoke) for a in archs}
        for n in args.simulate:
            alone = {a: round(simulate({a: dags[a]}, n), 1) for a in archs}
            print(f"{n} L40S: each architecture alone {alone} h; all {len(archs)} together {simulate(dags, n):.1f} h", file=sys.stderr)
        return
    out = open(args.csv, "w", newline="") if args.csv else sys.stdout
    w = csv.DictWriter(out, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    if args.csv:
        out.close()
    for arch in archs:
        mine = [r for r in rows if r["arch"] == arch and not r["optional"]]
        opt = [r for r in rows if r["arch"] == arch and r["optional"]]
        kinds = {}
        for r in mine:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        final_ckpts = sum(1 for r in mine if r["kind"] == "train" and r["stage"] != "lr-sweep") + sum(1 for r in mine if r["kind"] == "select")
        print(f"{arch}: {len(mine)} units {kinds}; {sum(r['est_gpu_hours'] for r in mine):.1f} GPU-h; "
              f"selected checkpoints kept {final_ckpts} (~{0.51 * final_ckpts:.1f} GB); optional cross_eval: {len(opt) - 1} units, "
              f"{sum(r['est_gpu_hours'] for r in opt):.1f} GPU-h", file=sys.stderr)


def simulate(dags: dict[str, list[dict]], gpus: int, starts: dict[str, float] | None = None) -> float:
    """Wall-clock hours for list scheduling (longest ready unit first, as the lanes do) of several DAGs on `gpus` GPUs.
    starts: hour at which each DAG's base exists (default 0). Units run for est_hours; no queueing delays."""
    import heapq
    starts = starts or {}
    pending = {(a, u["id"]): u for a, us in dags.items() for u in us if not u.get("optional")}
    done_at: dict = {}
    running: list = []  # (finish, key)
    t, free = 0.0, gpus
    while pending or running:
        ready = [k for k, u in pending.items() if t >= starts.get(k[0], 0.0) and all((k[0], d) in done_at and done_at[(k[0], d)] <= t for d in u["deps"])]
        ready.sort(key=lambda k: -pending[k]["est_hours"])
        while free and ready:
            k = ready.pop(0)
            heapq.heappush(running, (t + pending.pop(k)["est_hours"], k))
            free -= 1
        nxt = [running[0][0]] if running else []
        nxt += [v for v in starts.values() if v > t]
        t = min(nxt)
        while running and running[0][0] <= t:
            f, k = heapq.heappop(running)
            done_at[k] = f
            free += 1
    return t


def stage_of(u: dict) -> str:
    if u["kind"] in ("prep", "count", "collect"):
        return u["kind"]
    if u["kind"] == "base":
        return "base-eval"
    if u["kind"] == "select":
        return "lr-selection (+edge run)"
    if u["kind"] == "test":
        return "test-scoring"
    if u["kind"] == "xeval":
        return "cross_eval (optional, off by default)"
    if u["kind"] == "beam":
        return "beam (test only, beside greedy)"
    return "lr-sweep" if u.get("keep") else ("seed" if u.get("seed", 42) != 42 else "mono")


# ------------------------------------------------------------------------------------------------ run directory
class Run:
    def __init__(self, out: Path):
        self.out = out
        self.cfg = json.loads((out / "config.json").read_text())
        self.arch = self.cfg["arch"]
        self.smoke = bool(self.cfg.get("smoke"))
        self.a = ARCHS[self.arch]
        self.state = out / "state"
        self.logs = out / "logs"
        for d in (self.state, self.logs, out / "runs", out / "keep", out / "test", out / "base_eval", out / "results"):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def base(self) -> Path:
        return self.out / "base"

    def units(self) -> list[dict]:
        return json.loads((self.out / "units.json").read_text())

    def limit(self, split: str) -> int | None:
        if not self.smoke:
            return None
        return SMOKE["val_limit"] if split == "val" else SMOKE["test_limit"]

    # environment shared by every child process
    def env(self, extra: dict | None = None, train: bool = False) -> dict:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "SALLM_", "HF_", "TRANSFORMERS_"))}
        hf = HF_TRAIN if train else HF_EVAL
        env.update({
            "HF_HOME": hf, "HF_DATASETS_CACHE": f"{hf}/datasets", "HF_HUB_CACHE": f"{hf}/hub", "HF_METRICS_CACHE": f"{hf}/metrics",
            "HF_TOKEN_PATH": "/home/lmbanr001/.huggingface/token",  # private datasets (NCHLT); read by huggingface_hub, never copied
            "HF_HUB_DISABLE_XET": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_MODE": "disabled", "WANDB_SILENT": "true",
            "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "42", "OMP_NUM_THREADS": "1",
            "FLA_DISABLE_BACKEND_DISPATCH": "1", "MAMBA_SCAN_IMPL": "cuda", "SALLM_SKIP_MAMBA_KERNEL_CHECK": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "SALLM_AFRIHG_CACHE_DIR": f"{GEN}/data/afrihg_cache", "HYDRA_FULL_ERROR": "1", "SCRATCH": "/scratch/lmbanr001",
            # commit-pinned MasakhaPOS/InjongoIntent files (GitHub API: 60 requests/h/IP; prefilled from the Mac)
            "SALLM_SOURCE_CACHE_DIR": f"{ROLLOUT}/assets/source_cache",
            # persistent Triton kernel cache (FLA/GDN kernels otherwise recompile for minutes in every run)
            "TRITON_CACHE_DIR": f"{ROLLOUT}/triton_cache/{self.arch}",
        })
        path = [str(SALLM / "src/main")]
        if self.arch == "mamba2":
            env["SALLM_FLA_MAMBA2"] = "1"
            path.insert(0, str(HERE / "hooks"))
        env["PYTHONPATH"] = ":".join(path)
        if extra:
            for k, v in extra.items():
                if k == "PYTHONPATH_PREPEND":
                    env["PYTHONPATH"] = v + ":" + env["PYTHONPATH"]
                else:
                    env[k] = v
        return env

    def sh(self, cmd: list[str], log: Path, env: dict, cwd: str | None = None) -> float:
        log.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with log.open("a") as fh:
            fh.write(f"\n# {time.strftime('%Y-%m-%dT%H:%M:%S')} cwd={cwd or os.getcwd()}\n# {' '.join(map(str, cmd))}\n")
            fh.flush()
            rc = subprocess.run([str(c) for c in cmd], env=env, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            tail = "".join(log.read_text(errors="replace").splitlines(True)[-40:])
            raise RuntimeError(f"command failed rc={rc}: {' '.join(map(str, cmd))[:300]}\n--- {log} tail ---\n{tail}")
        return time.time() - t0


def alert(r: "Run", unit_id: str, msg: str) -> None:
    """One line in OUT/ALERTS.txt (rebalance.py copies new lines to ~/.sallm_fire/alerts.log on the Mac)."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {r.arch} {unit_id} {msg}".replace("\n", " ")
    with open(r.out / "ALERTS.txt", "a") as fh:
        fh.write(line + "\n")
    print(f"ALERT {line}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha256(root: Path) -> str:  # same definition as full_matrix/prepare_bindings.tree_sha256
    root = root.resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts)
    return hashlib.sha256("".join(f"{sha256(p)}  ./{p.relative_to(root).as_posix()}\n" for p in files).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    tmp.replace(path)


# ----------------------------------------------------------------------------------------------------- prep
def do_prep(r: Run, u: dict) -> dict:
    """Bind the pretrained weights: copy + tokenizer + checks. New bases have bos/eos/pad = 0/1/2 in their config."""
    src = Path(r.cfg["base_src"])
    if (src / "run_meta.json").exists():  # a matched-pretraining run dir: bind its final weights
        total = json.loads((src / "run_meta.json").read_text())["total_steps"]
        finals = sorted(src.glob(f"weights/step{total:06d}_tok*"))
        finals = [p for p in finals if not p.name.endswith(".tmp")]
        if len(finals) != 1:
            raise RuntimeError(f"pretraining not finished: no weights/step{total:06d}_tok* in {src}")
        src = finals[0]
    dst = r.base
    if not (dst / "BASE_READY.json").exists():
        tmp = r.out / f".base.{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        for name in ("config.json", "generation_config.json", "pytorch_model.bin", "model.safetensors"):
            if (src / name).exists():
                shutil.copy2(src / name, tmp / name)
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            shutil.copy2(SALLM / "tokenizer/sallm_bpe_tokenizer" / name, tmp / name)
        assert sha256(tmp / "tokenizer.json") != "" and json.loads((tmp / "tokenizer.json").read_text())["decoder"]["type"] == "ByteLevel"
        cfg = json.loads((tmp / "config.json").read_text())
        ids = [cfg.get("bos_token_id"), cfg.get("eos_token_id"), cfg.get("pad_token_id")]
        notes = {"source": str(src), "source_tree_sha256": tree_sha256(src), "config_special_ids": ids}
        if ids != [0, 1, 2]:
            if not r.cfg.get("fix_special_ids"):
                raise RuntimeError(f"base config bos/eos/pad = {ids}, expected [0, 1, 2]")
            # stand-in bases only (smoke): the retained MzansiLM config says bos 1 / eos 2, its tokenizer [BOS]=0 [EOS]=1 [PAD]=2
            for name in ("config.json", "generation_config.json"):
                if (tmp / name).exists():
                    c = json.loads((tmp / name).read_text())
                    c.update(bos_token_id=0, eos_token_id=1, pad_token_id=2)
                    (tmp / name).write_text(json.dumps(c, indent=2) + "\n")
            notes["special_ids_rewritten_from"] = ids
        tok = json.loads((tmp / "tokenizer.json").read_text())
        added = {t["content"]: t["id"] for t in tok["added_tokens"]}
        assert (added["[BOS]"], added["[EOS]"], added["[PAD]"]) == (0, 1, 2), added
        if r.arch == "xlstm":
            # the matched run trained with the TFLA triton kernel (mlstm_kernels); fine-tuning and evaluation use the
            # native kernels of the sealed xLSTM runtime, as every paper xLSTM run (parity: loss rel 1.8e-5).
            c = json.loads((tmp / "config.json").read_text())
            before = {k: c.get(k) for k in ("chunkwise_kernel", "sequence_kernel", "step_kernel", "mode")}
            c.update(chunkwise_kernel="chunkwise--native_autograd", sequence_kernel="native_sequence__native", step_kernel="native",
                     mode="train")
            (tmp / "config.json").write_text(json.dumps(c, indent=2) + "\n")
            notes["xlstm_kernels_rewritten_from"] = before
        (tmp / "BASE_READY.json").write_text(json.dumps(notes, indent=1) + "\n")
        os.replace(tmp, dst) if not dst.exists() else (shutil.rmtree(dst), os.replace(tmp, dst))
    # forward-pass sanity check with the evaluation loader (also proves FLA/xLSTM classes load)
    check = r.out / "base_eval" / "load_check.json"
    if not check.exists():
        code = ("import json,sys,torch\nfrom sallm.config import ModelEvalConfig\nfrom sallm.evaluation.harness import load_model_and_tokenizer\n"
                f"m,t=load_model_and_tokenizer(ModelEvalConfig(checkpoint='{dst}',dtype='{r.a['dtype']}',device='cuda:0'))\n"
                "ids=t('Molo, unjani? Ngiyabonga kakhulu.',return_tensors='pt').input_ids.to('cuda:0')\n"
                "with torch.no_grad(): out=m(input_ids=ids,labels=ids)\n"
                "c=m.config\njson.dump({'class':type(m).__module__+'.'+type(m).__name__,'loss':float(out.loss),"
                "'ids':[c.bos_token_id,c.eos_token_id,c.pad_token_id],'tok_ids':[t.bos_token_id,t.eos_token_id,t.pad_token_id],"
                "'params':sum(p.numel() for p in m.parameters())},open(sys.argv[1],'w'))\n")
        r.sh([r.a["py"], "-c", code, check], r.logs / "prep_load_check.log", r.env())
        info = json.loads(check.read_text())
        assert math.isfinite(info["loss"]) and info["loss"] < 9, info  # random init gives ~ln(65539) = 11.1
        assert info["tok_ids"] == [0, 1, 2], info
        if r.arch == "mamba2":
            assert info["class"].startswith("fla."), info
    # T2X train/validation-only assets for the fail-closed loader (own copy; the loader writes a lock file there)
    t2x = Path(ROLLOUT) / "assets" / "t2x_train_validation_only"
    if not all((t2x / n).exists() for n in ("train.data", "train.text", "valid.data", "valid.text")):
        t2x.mkdir(parents=True, exist_ok=True)
        for n in ("train.data", "train.text", "valid.data", "valid.text"):
            shutil.copy2(f"{PILOT_T2X}/{n}", t2x / n)
    write_protocols(r)
    return {"base": str(dst), "tree_sha256": tree_sha256(dst), "load_check": json.loads(check.read_text())}


def rebind_protocols(r: Run) -> None:
    """code/sallm was re-synced (a fix): rebind the sequence protocols to its current manifest."""
    proto = r.out / "protocols" / "seq_fft.json"
    if proto.exists() and json.loads(proto.read_text())["source_snapshot"]["manifest_sha256"] != snapshot_manifest_sha():
        with Lock(r.out / "protocols" / ".lock"):
            write_protocols(r)
            print("PROTOCOLS_REBOUND to the current code/sallm manifest", flush=True)


def snapshot_manifest_sha() -> str:
    return sha256(SALLM / "SNAPSHOT_MANIFEST.sha256")


def write_protocols(r: Run) -> None:
    """Sequence-runner protocols: FT (v2 runner, any checkpoint) and Base (0914 runner, this base bound by hash)."""
    arch, a = r.arch, r.a
    iface = {"dtype": a["dtype"], "merge_lora": a["merge"], "tie_word_embeddings": a["tie"], "execution_status": "runnable"}
    source = {"path": str(SALLM), "manifest_sha256": snapshot_manifest_sha()}  # NFC-normalised NER span F1
    p = json.loads(Path(SEQ_PROTOCOL).read_text())
    p["source_snapshot"] = source
    p["models"] = {arch: {**iface, "base_path": None, "adapter_path": None, "base_tree_sha256": None, "adapter_tree_sha256": None,
                          "base_files": {}, "adapter_files": {}}}
    write_json(r.out / "protocols" / "seq_fft.json", p)
    files = {n: sha256(r.base / n) for n in ("config.json", "pytorch_model.bin", "model.safetensors", "tokenizer.json",
                                            "tokenizer_config.json") if (r.base / n).exists()}
    for idx, group in ((1, "ner"), (2, "pos")):
        p = json.loads(Path(BASE_SEQ_PROTOCOL.format(idx, group)).read_text())
        p["source_snapshot"] = source
        p["models"] = {arch: {**iface, "base_path": str(r.base), "adapter_path": None, "base_tree_sha256": tree_sha256(r.base),
                              "adapter_tree_sha256": None, "base_files": files, "adapter_files": {}}}
        write_json(r.out / "protocols" / f"seq_base_{group}.json", p)


# ------------------------------------------------------------------------------------------------ scoring
def seq_env(r: Run, langs: list[str], split: str) -> dict:
    extra = {"SEQ_LANGUAGES": ",".join(langs), "PYTHONPATH_PREPEND": f"{BUNDLE}"}
    if split == "val":
        extra["SEQ_VALIDATION_PROMPTS"] = "protocol"
    if r.arch == "xlstm":
        extra["SEQ_BATCH1"] = "1"
    return r.env(extra)


def gen_spec(r: Run, model: Path, tasks: list[str], out: Path) -> Path:
    spec = [{"unit_id": out.name, "architecture": r.arch, "tasks": tasks, "base": str(model),
             "base_tree_sha256": tree_sha256(model), "adapter": None, "adapter_tree_sha256": None}]
    write_json(out.with_suffix(".spec.json"), spec)
    return out.with_suffix(".spec.json")


def score(r: Run, family: str, split: str, model: Path, langs: list[str], out: Path, beam: bool = False) -> dict:
    """Score one family on one split with the paper's protocol scorer. Returns {"per_lang": {lang: points}, ...}."""
    done = out.with_suffix(".score.json")
    if done.exists():
        return json.loads(done.read_text())
    arch, py, lim = r.arch, r.a["py"], r.limit(split)
    log = r.logs / "scoring" / (str(out.relative_to(r.out)).replace("/", "__") + ".log")
    hs = str(HERE / "runners")
    kit_split = "validation" if split == "val" else "test"
    if out.exists():
        out.rename(out.with_name(out.name + f".partial{int(time.time())}"))
    t0 = time.time()
    per, n = {}, {}
    # selection uses the fixed validation subsample (val_subsample.json); test is always full size
    os.environ["FFT_VAL_SUBSAMPLE"] = "1" if split == "val" else "0"
    os.environ["FFT_VAL_SPLIT"] = split
    if family in ("news", "sib"):
        raw = out.with_suffix(".json")
        cmd = [py, f"{hs}/{family}_score.py", "--arch", arch, "--base", model, "--split", kit_split, "--langs", ",".join(langs), "--out", raw]
        # News scores with the frozen v15 classification code; SIB with the v8-equivalent snapshot, as the kit.
        env = r.env()
        if family == "news":
            env["PYTHONPATH"] = env["PYTHONPATH"].replace(str(SALLM / "src/main"), f"{S}/results/monomulti_reselect_20260925/jobs/kit/news_v15/src/main")
        r.sh(cmd + (["--limit", lim] if lim else []), log, env)
        d = json.loads(raw.read_text())
        key = "weighted_f1" if family == "news" else "f1"
        for lang in langs:
            per[lang] = 100 * float(d["languages"][lang][key])
            n[lang] = d["languages"][lang]["n_items"]
    elif family == "intent":
        cmd = [py, f"{hs}/intent_eval.py", "--arch", arch, "--base", model, "--split", kit_split, "--langs", ",".join(langs), "--output", out]
        if split == "val":
            cmd += ["--val-source", "carve"]
        r.sh(cmd + (["--limit", lim] if lim else []), log, r.env())
        d = json.loads((out / "SUMMARY.json").read_text())
        for lang in langs:
            per[lang] = 100 * d["languages"][lang]["weighted_f1"]
            n[lang] = d["languages"][lang]["n_items"]
    elif seq_kind(family) in ("ner", "pos"):
        rebind_protocols(r)
        raw = out.with_suffix(".json")
        cmd = [py, f"{hs}/seq_eval.py", "--task", family, "--phase", kit_split, "--architecture", arch, "--checkpoint", model,
               "--protocol", r.out / "protocols" / "seq_fft.json", "--output", raw]
        if split == "test" and family in ("ner", "pos"):  # NCHLT test is gated by its own select unit only
            cmd += ["--selection", SEQ_SELECTION, "--release", SEQ_RELEASE.format(family.upper())]
        r.sh(cmd + (["--limit", lim] if lim else []), log, seq_env(r, langs, split))
        per, n = read_seq(json.loads(raw.read_text()), family)
    elif family in ("t2x", "afrihg"):
        tasks = ["t2x_xho"] if family == "t2x" else [f"afrihg_{lang}" for lang in langs]
        runner = f"{hs}/gen_direct_bs1.py" if arch == "xlstm" else f"{hs}/gen_direct.py"
        env = r.env({"SALLM_T2X_CACHE_DIR": f"{GEN}/data/t2x_cache", "PYTHONPATH_PREPEND": BUNDLE})
        cmd = [py, runner, "--spec", gen_spec(r, model, tasks, out), "--index", "0", "--source", V8, "--output", out,
               "--split", split, "--system-prompt", "drop", "--decoding", "config" if beam else "greedy"]
        if beam:
            assert split == "test" and BEAM_MODE[arch] != "unsupported", (split, arch)
            cmd += ["--no-cache"] if BEAM_MODE[arch] == "nocache" else []
        r.sh(cmd + (["--max-samples", lim] if lim else []), log, env, cwd=GEN)
        summary = json.loads((out / "evaluation_summary.json").read_text())
        for row in summary:
            lang = row["task"].split("_")[-1]
            per[lang] = float(row["metrics"][f"eval/{row['task']}/all_chrf"])
            n[lang] = sum(1 for _ in (out / row["task"] / "examples.jsonl").open())
    elif family == "belebele":
        packs = [f"belebele_{lang}" for lang in langs]
        cmd = [py, f"{hs}/prefix_eval.py", "--arch", arch, "--base", model, "--mode", "train", "--packs", *packs, "--output", out]
        r.sh(cmd + (["--limit", lim] if lim else []), log, r.env())
        for pack in packs:
            res = json.loads((out / pack / "results.json").read_text())
            for task, vals in res["results"].items():
                lang = task.split("_")[1] if task.startswith("belebele_") else None
                # prompt 1 only, as in the Base protocol; the pack also scores prompts 2-5
                if lang and task == f"belebele_{lang}_prompt_1":
                    per[lang] = 100 * float(vals["acc_norm,none"])
                    n[lang] = len(res["samples"][task])
    elif family == "transfer":  # AfriXNLI / AfriMMLU / AfriMGSM with the General protocol runner
        raw = out.with_suffix(".json")
        cmd = [py, f"{hs}/general_prompt_lm_eval.py", "--phase", "test", "--architecture", arch, "--checkpoint", model,
               "--selection", f"{GP_B}/selection/SELECTION.json", "--dtype", r.a["dtype"], "--output", raw]
        if r.a["merge"]:
            cmd += ["--merge-lora"]
        if r.a["tie"] is not None:
            cmd += ["--tie-word-embeddings", str(r.a["tie"]).lower()]
        if arch == "xlstm":
            cmd += ["--batch-size", "1", "--max-batch-size", "1"]
        env = r.env({"GENERAL_PROMPT_GROUPS": "closed_and_generation"})
        env["PYTHONPATH"] = ":".join(([str(HERE / "hooks")] if arch == "mamba2" else []) + [GP_B, f"{GP_S}/scripts", f"{GP_S}/src/main"])
        r.sh(cmd + (["--limit", lim] if lim else []), log, env)
        d = json.loads(raw.read_text())
        for task, vals in d["calls"]["closed_and_generation"]["results"].items():
            m = re.match(r"(afrixnli|afrimmlu_direct|afrimgsm)_([a-z]+)_prompt_\d+$", task)
            if not m:
                continue
            fam = {"afrixnli": "afrixnli", "afrimmlu_direct": "afrimmlu", "afrimgsm": "afrimgsm"}[m[1]]
            key = "exact_match,flexible-extract" if fam == "afrimgsm" else "acc,none"
            per[f"{fam}:{m[2]}"] = 100 * float(vals[key])
            n[f"{fam}:{m[2]}"] = len(d["calls"]["closed_and_generation"]["samples"][task])
    else:
        raise ValueError(family)
    rec = {"family": family, "split": split, "model": str(model), "per_lang": per, "n": n, "raw": str(out),
           "mean": sum(per.values()) / len(per), "secs": round(time.time() - t0, 1), "limit": lim, "gpu": gpu_name(),
           "val_subsample_sha256": hashlib.sha256((HERE / "val_subsample.json").read_bytes()).hexdigest() if split == "val" else None,
           "decoding": ("beam" + ("" if BEAM_MODE[arch] == "cache" else " (no cache)")) if beam else "greedy"}
    rec["sanity"] = run_sanity(r, rec, out)
    write_json(done, rec)
    return rec


# ------------------------------------------------------------------------------------------------ sanity checks
def majority_weighted_f1(gold: list) -> float:
    """Weighted F1 (points) of always predicting the most frequent gold label: p * 2p / (1 + p)."""
    from collections import Counter
    p = Counter(gold).most_common(1)[0][1] / len(gold)
    return 100 * p * 2 * p / (1 + p)


def loop_4gram(text: str) -> bool:
    from collections import Counter
    w = text.split()
    return any(c >= 4 for c in Counter(tuple(w[i:i + 4]) for i in range(len(w) - 3)).values())


def sanity_flags(family: str, per_lang: dict, items: dict) -> dict:
    """items[lang] = {"gold": [...], "pred": [...]} (classification, POS tags), NER gold/predicted span strings, or {"pred": [texts]} (generation)."""
    from collections import Counter
    out = {}
    for lang, it in items.items():
        flags, pred = [], it["pred"]
        if not pred:
            out[lang] = {"n": 0, "flags": ["no items"]}
            continue
        score = per_lang.get(lang)
        stats = {"n": len(pred)}
        if family in ("news", "sib", "intent", "belebele"):
            top = Counter(map(str, pred)).most_common(1)[0]
            stats["top_prediction_share"] = round(top[1] / len(pred), 4)
            if top[1] / len(pred) > 0.8:
                flags.append(f"one label ({top[0]}) is {100 * top[1] / len(pred):.0f}% of predictions")
            base = 100 * Counter(it["gold"]).most_common(1)[0][1] / len(it["gold"]) if family == "belebele" else majority_weighted_f1(it["gold"])
            stats["majority_baseline"] = round(base, 3)
            if score is not None and score <= base:
                flags.append(f"score {score:.2f} <= majority baseline {base:.2f}")
        elif seq_kind(family) in ("ner", "pos"):
            is_empty = [not x or (isinstance(x, str) and not x.strip()) for x in pred]
            empty = sum(is_empty) / len(pred)
            stats["empty_share"] = round(empty, 4)
            if seq_kind(family) == "ner":
                # an empty output is correct when the sentence has no entities: flag only missed entities
                has_ents = [bool(str(g).strip()) for g in it["gold"]]
                missed = sum(e and h for e, h in zip(is_empty, has_ents)) / max(1, sum(has_ents))
                stats["empty_share_where_gold_has_entities"] = round(missed, 4)
                if missed > 0.2:
                    flags.append(f"{100 * missed:.0f}% empty outputs on sentences that have entities")
            elif empty > 0.2:
                flags.append(f"{100 * empty:.0f}% empty or unparseable outputs")
            if seq_kind(family) == "ner":
                base = 0.0  # all-O: no entity spans
            else:
                tags = [t for g in it["gold"] for t in g]
                base = 100 * Counter(tags).most_common(1)[0][1] / len(tags)
            stats["trivial_baseline"] = round(base, 3)
            if score is not None and score <= base:
                flags.append(f"score {score:.2f} <= trivial baseline {base:.2f}")
        elif family in ("t2x", "afrihg"):
            empty = sum(1 for x in pred if not x.strip()) / len(pred)
            loops = sum(1 for x in pred if loop_4gram(x)) / len(pred)
            stats.update(empty_share=round(empty, 4), loop_share=round(loops, 4))
            if empty > 0.2:
                flags.append(f"{100 * empty:.0f}% empty outputs")
            if loops > 0.3:
                flags.append(f"{100 * loops:.0f}% of outputs repeat a 4-gram 4+ times")
        out[lang] = {**stats, "flags": flags}
    return out


def sanity_items(family: str, out: Path) -> dict:
    items: dict = {}
    if family in ("news", "sib"):
        for x in json.loads(out.with_suffix(".json").read_text())["rows"]:
            d = items.setdefault(x["language"], {"gold": [], "pred": []})
            d["gold"].append(x["gold"])
            d["pred"].append(x["prediction"])
    elif family == "intent":
        res = json.loads((out / "injongointent_all" / "results.json").read_text())
        for task, samples in res["samples"].items():
            lang = task.split("_")[1]
            choices = res["configs"][task]["doc_to_choice"]
            d = items.setdefault(lang, {"gold": [], "pred": []})
            for smp in samples:
                lls = [float(v[0]) for v in smp["filtered_resps"]]
                d["pred"].append(choices[max(range(len(lls)), key=lls.__getitem__)])
                d["gold"].append(smp["doc"]["intent"])
    elif family == "belebele":
        for res_path in out.glob("belebele_*/results.json"):
            res = json.loads(res_path.read_text())
            for task, samples in res["samples"].items():
                if task != f"belebele_{task.split('_')[1]}_prompt_1":
                    continue
                d = items.setdefault(task.split("_")[1], {"gold": [], "pred": []})
                for smp in samples:
                    lls = [float(v[0]) for v in smp["filtered_resps"]]
                    d["pred"].append(max(range(len(lls)), key=lls.__getitem__))
                    d["gold"].append(smp["target"])
    elif seq_kind(family) == "ner":
        d = json.loads(out.with_suffix(".json").read_text())
        lang_of = {name: ev["language"] for name, ev in d["task_evidence"].items()}
        for row in d["rows"]:
            it = items.setdefault(lang_of[row["task"]], {"gold": [], "pred": []})
            it["gold"].append(row["target"])
            it["pred"].append(row["prediction"])
    elif seq_kind(family) == "pos":
        for row in json.loads(out.with_suffix(".json").read_text())["rows"]:
            d = items.setdefault(row["language"], {"gold": [], "pred": []})
            d["gold"].append(row["gold"])
            d["pred"].append(row["prediction"])
    elif family in ("t2x", "afrihg"):
        for ex_path in out.glob("*/examples.jsonl"):
            lang = ex_path.parent.name.split("_")[-1]
            items.setdefault(lang, {"pred": []})["pred"] += [json.loads(line)["prediction"] for line in ex_path.open()]
    return items


def run_sanity(r: "Run", rec: dict, out: Path) -> dict:
    """Per-score sanity flags -> rec, sanity/<unit>.json (SANITY per unit) and ALERTS.txt. Never fails the unit."""
    unit_id = os.environ.get("FFT_UNIT", "manual")
    try:
        res = sanity_flags(rec["family"], rec["per_lang"], sanity_items(rec["family"], out)) if rec["family"] != "transfer" else {}
    except Exception as exc:  # noqa: BLE001
        res = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    entry = {"family": rec["family"], "split": rec["split"], "model": rec["model"], "raw": str(out), "checks": res, "time": time.time()}
    path = r.out / "sanity" / f"{unit_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with Lock(path.with_suffix(".lock")):
        prev = json.loads(path.read_text()) if path.exists() else []
        write_json(path, prev + [entry])
    for lang, v in res.items():
        for f in (v.get("flags") or []) if isinstance(v, dict) else []:
            alert(r, unit_id, f"SANITY {rec['family']}/{lang} {rec['split']} ({Path(rec['model']).name}): {f}")
    return res


def read_seq(d: dict, family: str) -> tuple[dict, dict]:
    per, n = {}, {}
    if seq_kind(family) == "ner":
        for name, ev in d["task_evidence"].items():
            per[ev["language"]] = 100 * float(d["reported_metrics"][name])
            n[ev["language"]] = sum(1 for row in d["rows"] if row["task"] == name)
    else:
        for name, m in d["reported_metrics"].items():
            lang = name.split("/")[0]
            per[lang] = 100 * m["correct"] / m["total"]
            n[lang] = m["total"]
    return per, n


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def val_score(r: Run, u: dict, model: Path, out: Path) -> dict:
    """Validation score used for selection: mean over the unit's languages; General = mean of the six family means."""
    if u["family"] == "general":
        fams = {f: score(r, f, "val", model, list(FAMILIES[f]["langs"]), out / f) for f in GENERAL_TRAIN_FAMILIES}
        return {"score": sum(v["mean"] for v in fams.values()) / len(fams), "families": {f: v["per_lang"] for f, v in fams.items()}}
    rec = score(r, u["family"], "val", model, u["langs"], out / u["family"])
    return {"score": rec["mean"], "per_lang": rec["per_lang"]}


def test_scores(r: Run, u: dict, model: Path, out: Path) -> dict:
    if u["family"] == "general":
        fams = {f: score(r, f, "test", model, list(FAMILIES[f]["langs"]), out / f) for f in GENERAL_TRAIN_FAMILIES + ("intent",)}
        fams["belebele"] = score(r, "belebele", "test", model, ["afr", "eng", "sot", "ssw", "tsn", "tso", "xho", "zul"], out / "belebele")
        fams["transfer"] = score(r, "transfer", "test", model, [], out / "transfer")
        return {f: v["per_lang"] for f, v in fams.items()}
    return {u["family"]: score(r, u["family"], "test", model, u["langs"], out / u["family"])["per_lang"]}


# ---------------------------------------------------------------------------------------------------- training
def run_id(u: dict, lr: str) -> str:
    tag = {"multi": "multi", "general": "general"}.get(u["regime"], "mono-" + "-".join(u["langs"]))
    return f"{u['family']}-{tag}-lr{lr}-s{u['seed']}"


def selected_lr(r: Run, family: str) -> str:
    return json.loads((r.out / "keep" / family / "SELECTED.json").read_text())["lr"]


def hydra_config(u: dict) -> str:
    spec = FAMILIES[u["family"]]
    return spec["mono"].format(u["langs"][0]) if u["regime"] == "mono" else spec["multi"]


def train_run(r: Run, u: dict, lr: str) -> dict:
    """Train, score every epoch on validation, keep/test the best epoch. Resumable at the RUN_DONE level only."""
    rid = run_id(u, lr)
    run = r.out / "runs" / rid
    if (run / "RUN_DONE.json").exists():
        return json.loads((run / "RUN_DONE.json").read_text())
    if (run / "DIVERGED.json").exists():
        raise Diverged(f"FAILED_DIVERGED earlier: {(run / 'DIVERGED.json').read_text()[:300]}")
    resume = sorted((run / "resume").glob("checkpoint-*")) if (run / "resume").is_dir() else []
    if run.exists() and not resume:
        shutil.move(str(run), str(r.out / "runs" / f".{rid}.aborted{int(time.time())}"))
    run.mkdir(parents=True, exist_ok=True)
    jid = os.environ.get("SLURM_JOB_ID", "local")
    shm = Path(f"/dev/shm/fft_{jid}_{rid}")
    shutil.rmtree(shm, ignore_errors=True)
    free_gb = shutil.disk_usage("/dev/shm").free / 2**30
    need = 0.6 * (10 if u["family"] != "general" else 4) + 2
    if free_gb < need:
        raise RuntimeError(f"/dev/shm has {free_gb:.0f} GB free, need {need:.0f}")
    shm.mkdir(parents=True)
    try:
        args = [
            "--config-name", f"finetune/{hydra_config(u)}",
            f"finetune.model.architecture={r.a['key']}", f"finetune.model.init_checkpoint={r.base}",
            f"finetune.tokenizer.path={r.base}", "++finetune.peft.method=none",
            f"++finetune.training.output_dir={shm}", f"++finetune.training.logging_dir={run}/logs",
            f"++finetune.training.run_name=fft-{r.arch}-{rid}", f"hydra.run.dir={run}/hydra",
            "++finetune.training.report_to=none", "++finetune.hub.enabled=false", "++finetune.hub.push_adapter=false",
            "++finetune.hub.push_merged=false", f"++finetune.training.learning_rate={lr}", "++finetune.training.optim=adamw_torch",
            "++finetune.training.adam_beta1=0.9", "++finetune.training.adam_beta2=0.95", "++finetune.training.adam_epsilon=1e-8",
            "++finetune.training.weight_decay=0.01", "++finetune.training.lr_scheduler_type=cosine", "++finetune.training.warmup_ratio=0.10",
            f"++finetune.training.per_device_train_batch_size={MICRO.get(u['family'], 16)}",
            f"++finetune.training.gradient_accumulation_steps={16 // MICRO.get(u['family'], 16)}",
            f"++finetune.training.per_device_eval_batch_size={MICRO.get(u['family'], 16)}", "++finetune.training.num_train_epochs=4",
            "++finetune.training.max_grad_norm=1.0", "++finetune.training.bf16=true", "++finetune.training.gradient_checkpointing=false",
            "++finetune.training.label_smoothing_factor=0.0",
            # no trainer eval loss: selection uses the protocol scorers, and the loss pass over large validation sets
            # (General 22k rows) cost ~10+ min per epoch in the smoke
            "++finetune.training.eval_strategy=no",
            "++finetune.training.save_strategy=epoch", "++finetune.training.save_only_model=true", "++finetune.training.save_total_limit=null",
            "++finetune.training.load_best_model_at_end=false", "++finetune.training.metric_for_best_model=null",
            "++finetune.training.greater_is_better=null", "++finetune.training.early_stopping_patience=null",
            f"++finetune.training.seed={u['seed']}", f"++finetune.training.data_seed={u['seed']}", "++finetune.training.logging_steps=10",
            f"++finetune.training.resume_from_checkpoint={resume[-1] if resume else 'null'}",
            "++finetune.training.dataloader_num_workers=2",
        ]
        if u["family"] == "t2x":
            args.append("finetune.dataset.max_seq_length=1024")  # pilot
        if r.arch == "xlstm":
            args.append("++finetune.training.pad_to_multiple_of=64")  # chunkwise kernel needs chunk multiples; pads are loss-masked
        if r.smoke:
            args.append(f"++finetune.training.max_steps={SMOKE['max_steps']}")
        # no HF offline flags: the MasakhaNER parquet cache is only found when its pinned revision resolves online
        extra = {"FFT_EPOCHS": "auto", "SALLM_DISABLE_TASK_METRICS": "1", "FFT_UNIT": os.environ.get("FFT_UNIT", u["id"]),
                 "FFT_CTRL": json.dumps({"out": str(r.out), "run": str(run), "unit": u, "patience": PATIENCE})}
        if resume:
            print(f"RESUME {rid} from {resume[-1].name}", flush=True)
        if u["family"] in ("t2x", "general"):
            extra.update(FFT_T2X_LOADER="1", SALLM_T2X_TRAIN_VALIDATION_ONLY="1",
                         SALLM_T2X_CACHE_DIR=f"{ROLLOUT}/assets/t2x_train_validation_only")
        env = r.env(extra, train=True)
        (run / "overrides.txt").write_text("\n".join(args) + "\n")
        r.sh([r.a["py"], HERE / "train_fft.py", *args, "--cfg", "job", "--resolve"], run / "resolved_config.log",
             {**env, "FFT_RUN_INFO": "/dev/null"})
        resolved = (run / "resolved_config.log").read_text()
        if re.search(r"(^|\s)(test|test_split)\s*:", resolved.split("finetune:", 1)[-1]):
            raise RuntimeError("held-out field in the resolved config")
        smi = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader", "-lms", "10000"],
                               stdout=(run / "nvidia_smi_mem.log").open("a"), stderr=subprocess.DEVNULL)
        try:
            train_s = r.sh([r.a["py"], HERE / "train_fft.py", *args], run / "train.log", {**env, "FFT_RUN_INFO": str(run / "run_info.json")})
        except RuntimeError as exc:
            log = (run / "train.log").read_text(errors="replace") if (run / "train.log").exists() else ""
            if (run / "DIVERGED.json").exists() or re.search(r"returned nan values|FAILED_DIVERGED", log):
                if not (run / "DIVERGED.json").exists():
                    write_json(run / "DIVERGED.json", {"reason": "non-finite values in backward (anomaly detection)", "time": time.time()})
                raise Diverged(f"FAILED_DIVERGED {rid}: {(run / 'DIVERGED.json').read_text()[:300]}") from exc
            raise
        finally:
            smi.terminate()
        log = (run / "train.log").read_text(errors="replace")
        if re.search(r"'loss': nan|loss=nan|non-finite|returned nan values", log):
            write_json(run / "DIVERGED.json", {"reason": "non-finite values in the training log", "time": time.time()})
            raise Diverged(f"FAILED_DIVERGED {rid}: non-finite values in training ({run}/train.log)")
        info = json.loads((run / "run_info.json").read_text())
        epochs = json.loads((run / "val_epochs.json").read_text())
        es = json.loads((run / "early_stop.json").read_text())
        best = json.loads((run / "best" / "BEST.json").read_text())
        planned = 1 if r.smoke else es["planned_epochs"]
        ran = es["epochs_run"]
        if [e["epoch"] for e in epochs] != list(range(1, ran + 1)) or (not es["stopped_early"] and ran != planned):
            raise RuntimeError(f"epoch record incomplete: ran {ran} of {planned}, scored {[e['epoch'] for e in epochs]}")
        assert best["epoch"] == es["best_epoch"], (best, es)
        best_dir = run / "best"
        tree = tree_sha256(best_dir)
        done = {"arch": r.arch, "unit": u["id"], "family": u["family"], "regime": u["regime"], "langs": u["langs"], "lr": lr,
                "seed": u["seed"], "epochs": epochs, "best_epoch": best["epoch"], "best_val": best["val"], "best_tree_sha256": tree,
                "stopped_early": es["stopped_early"], "stop_epoch": es["stop_epoch"], "epochs_run": ran, "planned_epochs": planned,
                "patience": es["patience"], "train_wall_s": round(train_s, 1), "run_info": info, "gpu": gpu_name(),
                "host": socket.gethostname(), "job": jid, "resumed": bool(resume)}
        if u.get("test"):
            t0 = time.time()
            done["test"] = test_scores(r, u, best_dir, r.out / "test" / rid)
            done["test_secs"] = round(time.time() - t0, 1)
        # best epoch kept (weights only). Sweep runs: until LR selection; every other run's best epoch is selected.
        dest = r.out / "keep" / u["family"] / f"{rid}_e{best['epoch']}"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(best_dir, dest.with_name(dest.name + ".tmp"), ignore=shutil.ignore_patterns("BEST.json"))
        os.replace(dest.with_name(dest.name + ".tmp"), dest)
        write_manifest(dest)
        done["kept"] = str(dest)
        if not u.get("keep"):
            mark_selected(dest, u["id"])
        write_json(run / "RUN_DONE.json", done)
        shutil.rmtree(run / "best", ignore_errors=True)
        shutil.rmtree(run / "resume", ignore_errors=True)
        return done
    finally:
        shutil.rmtree(shm, ignore_errors=True)


def write_manifest(ckpt: Path) -> None:
    """<ckpt>.sha256 (sha256sum -c format, relative to keep/<family>/) for the Kombuys archive check."""
    lines = [f"{sha256(p)}  {ckpt.name}/{p.relative_to(ckpt).as_posix()}\n" for p in sorted(ckpt.rglob("*")) if p.is_file()]
    ckpt.with_name(ckpt.name + ".sha256").write_text("".join(lines))


def mark_selected(ckpt: Path, unit_id: str) -> None:
    write_json(ckpt.with_name(ckpt.name + ".selected"), {"unit": unit_id, "tree_sha256": tree_sha256(ckpt), "time": time.time()})


def transferred_lr(r: Run, u: dict) -> str:
    """General LR: the LR chosen most often across the architecture's Multi selections; ties -> the lower LR."""
    from collections import Counter
    chosen = {f: selected_lr(r, f) for f in u["lr_from"]}
    if not chosen:  # smoke subsets without a Multi family
        lr = SMOKE["default_lrs"][0]
    else:
        counts = Counter(chosen.values())
        lr = max(counts, key=lambda x: (counts[x], -float(x)))
    write_json(r.out / "keep" / "general" / "LR_TRANSFER.json", {"chosen": chosen, "lr": lr,
               "rule": "most frequent Multi-selected LR (news, sib, intent, ner, pos, afrihg); ties -> lower LR"})
    return lr


def do_train(r: Run, u: dict) -> dict:
    lr = u["lr"] or (transferred_lr(r, u) if u.get("lr_from") is not None else selected_lr(r, u["family"]))
    done = train_run(r, u, lr)
    return {"run": run_id(u, lr), "lr": lr, "best_epoch": done["best_epoch"], "best_val": done["best_val"], "test": done.get("test")}


def do_select(r: Run, u: dict) -> dict:
    """Best (lr, epoch) over the sweep; while the best LR is at an edge, train the next point beyond it (<= 2) and reselect."""
    def runs(lrs):
        return {lr: json.loads((r.out / "runs" / run_id({**u, "seed": 42}, lr) / "RUN_DONE.json").read_text()) for lr in lrs}

    def best(rs):  # max validation score; ties -> smaller lr, then earlier epoch
        cands = [(e["val"], -float(lr), -e["epoch"], lr, e["epoch"]) for lr, d in rs.items() for e in d["epochs"]]
        top = max(cands)
        return top[3], top[4], top[0]

    lrs = sorted(u["lrs"], key=float)
    rs = runs(lrs)
    lr, ep, val = best(rs)
    record = {"grid": lrs, "core_best": [lr, ep, val], "extensions": []}
    while u["edge"] and (ext := next_edge_lr(lrs, lr, len(record["extensions"]))):
        record["extensions"].append(ext)
        train_run(r, {**u, "seed": 42, "keep": True, "test": False, "lr": ext}, ext)
        lrs = sorted(lrs + [ext], key=float)
        rs = runs(lrs)
        lr, ep, val = best(rs)
    sel = {**record, "final_grid": lrs, "lr": lr, "epoch": ep, "val": val, "checkpoint": rs[lr]["kept"],
           "best_at_grid_edge": bool(u["edge"] and len(lrs) > 1 and lr in (lrs[0], lrs[-1])),
           "rule": ("max validation score over (lr, epoch); ties -> smaller lr, earlier epoch; edge rule (amended 27 Sep "
                    "2026): while the best LR is at a grid edge, add the next x3 point beyond it, at most 2 extra points"),
           "note": "the selected epoch is the kept (best) epoch of its LR run",
           "by_lr": {k: [e["val"] for e in d["epochs"]] for k, d in rs.items()}}
    assert Path(sel["checkpoint"]).is_dir() and sel["checkpoint"].endswith(f"_e{ep}"), sel["checkpoint"]
    write_json(r.out / "keep" / u["family"] / "SELECTED.json", sel)
    mark_selected(Path(sel["checkpoint"]), u["id"])
    for lr_other, d in rs.items():  # unselected sweep checkpoints are not needed any more
        kept = Path(d["kept"])
        if str(kept) != sel["checkpoint"]:
            shutil.rmtree(kept, ignore_errors=True)
            kept.with_name(kept.name + ".sha256").unlink(missing_ok=True)
    return sel


def do_test(r: Run, u: dict) -> dict:
    sel = json.loads((r.out / "keep" / u["family"] / "SELECTED.json").read_text())
    rid = run_id({**u, "seed": 42}, sel["lr"])
    res = test_scores(r, u, Path(sel["checkpoint"]), r.out / "test" / rid)
    write_json(r.out / "runs" / rid / "TEST_DONE.json", {"selection": sel, "test": res, "gpu": gpu_name()})
    return {"run": rid, "test": res}


def do_beam(r: Run, u: dict) -> dict:
    src = json.loads((r.state / f"{u['src']}.json").read_text())["result"]
    rid = src["run"]
    ckpt = Path(json.loads((r.out / "keep" / u["family"] / "SELECTED.json").read_text())["checkpoint"]) if u["src"].startswith("test-") \
        else Path(json.loads((r.out / "runs" / rid / "RUN_DONE.json").read_text())["kept"])
    if BEAM_MODE[r.arch] == "unsupported":
        res = {"status": "not supported by the implementation", "reason": "FLA 0.5.1 cache cannot be reordered across beams"}
    else:
        res = {f: score(r, f, "test", ckpt, langs, r.out / "test" / f"beam-{rid}" / f, beam=True)["per_lang"] for f, langs in u["gen"].items()}
    write_json(r.out / "runs" / rid / "BEAM_DONE.json", {"unit": u["id"], "checkpoint": str(ckpt), "mode": BEAM_MODE[r.arch],
                                                         "gen": u["gen"], "test": res})
    return {"run": rid, "test": res}


def do_xeval(r: Run, u: dict) -> dict:
    rid = run_id({**u, "seed": 42}, selected_lr(r, u["family"]))
    done = json.loads((r.out / "runs" / rid / "RUN_DONE.json").read_text())
    ckpt = Path(done["kept"])
    res = score(r, u["family"], "test", ckpt, u["targets"], r.out / "test" / f"xeval-{rid}" / u["family"])
    write_json(r.out / "runs" / rid / "XEVAL_DONE.json", {"train_language": u["langs"][0], "targets": u["targets"],
                                                          "test": res["per_lang"], "checkpoint": str(ckpt)})
    return {"run": rid, "test": res["per_lang"]}


# ------------------------------------------------------------------------------------------------------- Base
def do_base(r: Run, u: dict) -> dict:
    g, arch, py = u["group"], r.arch, r.a["py"]
    out = r.out / "base_eval" / g
    hs = HERE / "runners"
    if g == "gen":
        return score_base_gen(r, out)
    lim = 20 if r.smoke else None
    if g in ("ner", "pos"):
        raw = out.with_suffix(".json")
        rebind_protocols(r)
        if not raw.exists():
            env = r.env({"PYTHONPATH_PREPEND": BUNDLE, **({"SEQ_BATCH1": "1"} if arch == "xlstm" else {})})
            r.sh([py, hs / "seq_eval_base.py", "--task", g, "--phase", "test", "--architecture", arch, "--checkpoint", r.base,
                  "--protocol", r.out / "protocols" / f"seq_base_{g}.json", "--selection", SEQ_SELECTION,
                  "--release", BASE_SEQ_RELEASE.format(g.upper()), "--output", raw] + (["--limit", lim] if lim else []),
                 r.logs / f"base_{g}.log", env)
        per, n = read_seq(json.loads(raw.read_text()), g)
        return {"per_lang": per, "n": n, "raw": str(raw)}
    # prompt: the frozen official Base unit of this architecture, rebound to the new base
    raw = out.with_suffix(".json")
    if not raw.exists():
        inv = r.out / "base_eval" / "inventory"
        units = json.loads((Path(BUNDLE) / "frozen-inventory/official_units.json").read_text())
        index = next(i for i, x in enumerate(units) if x["architecture"] == arch and x["regime"] == "Base" and x["task_group"] == "prompt")
        units[index]["binding"]["base"] = {"kind": "path", "path": str(r.base), "expected_tree_sha256": tree_sha256(r.base)}
        write_json(inv / "official_units.json", units)
        rid = lambda ref: hashlib.sha256(json.dumps(ref, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()  # noqa: E731
        entries = {rid(units[index]["binding"]["base"]): {"path": str(r.base), "tree_sha256": tree_sha256(r.base)},
                   rid({"kind": "none"}): {"path": None}}
        write_json(inv / "BINDINGS.json", {"entries": entries})
        write_json(inv / "RELEASE.json", {"authorized": True, "no_score_based_retry": True, "selected_bindings": {}})
        env = r.env({"PYTHONPATH_PREPEND": BUNDLE, "LM_EVAL_BATCH": "1" if arch == "xlstm" else "8",
                     **({"LM_EVAL_LIMIT": str(lim)} if lim else {})})
        r.sh([py, hs / "lm_eval_unit.py", "--mode", "official", "--index", index, "--inventory", inv, "--bindings", inv / "BINDINGS.json",
              "--source", SALLM, "--release", inv / "RELEASE.json", "--output", raw], r.logs / "base_prompt.log", env)
    return {"raw": str(raw)}


def score_base_gen(r: Run, out: Path) -> dict:
    """Untuned T2X/AfriHG generation on test (greedy, no system prompt), as generation-protocol-v2.csv's Base rows."""
    t2x = score(r, "t2x", "test", r.base, ["xho"], out / "t2x")
    hg = score(r, "afrihg", "test", r.base, ["xho", "zul"], out / "afrihg")
    return {"t2x": t2x["per_lang"], "afrihg": hg["per_lang"]}


def do_count(r: Run, u: dict) -> dict:
    """Smoke only: tokenize every training config once (FFT_COUNT_ONLY) -> rows and tokens for the estimates."""
    res = {}
    configs = {f"{f}-multi": FAMILIES[f]["multi"] for f in FAMILIES if FAMILIES[f]["multi"]}
    configs["t2x-xho"] = FAMILIES["t2x"]["mono"].format("xho")
    for name, cfg in configs.items():
        info_path = r.out / "count" / f"{name}.json"
        if not info_path.exists():
            fam = name.split("-")[0]
            extra = {"FFT_COUNT_ONLY": "1", "FFT_RUN_INFO": str(info_path), "SALLM_DISABLE_TASK_METRICS": "1"}
            if fam in ("t2x", "general"):
                extra.update(FFT_T2X_LOADER="1", SALLM_T2X_TRAIN_VALIDATION_ONLY="1", SALLM_T2X_CACHE_DIR=f"{ROLLOUT}/assets/t2x_train_validation_only")
            info_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                r.sh([r.a["py"], HERE / "train_fft.py", "--config-name", f"finetune/{cfg}", f"finetune.model.architecture={r.a['key']}",
                      f"finetune.model.init_checkpoint={r.base}", f"finetune.tokenizer.path={r.base}", "finetune.peft.method=none",
                      f"++finetune.training.output_dir=/dev/shm/count_{os.getpid()}", f"hydra.run.dir={r.out}/count/hydra/{name}",
                      "++finetune.training.report_to=none", "++finetune.hub.enabled=false"] + (["finetune.dataset.max_seq_length=1024"] if fam == "t2x" else []),
                     r.logs / "count" / f"{name}.log", r.env(extra, train=True))
            except RuntimeError as exc:
                res[name] = {"error": str(exc)[-600:]}
                continue
        d = json.loads(info_path.read_text())
        res[name] = {k: d.get(k) for k in ("train_rows", "val_rows", "train_tokens_per_epoch", "train_loss_tokens_per_epoch", "train_max_tokens")}
    write_json(r.out / "count" / "SUMMARY.json", res)
    return res


# ------------------------------------------------------------------------------------------------- lane worker
KINDS = {"prep": do_prep, "count": do_count, "base": do_base, "train": do_train, "select": do_select, "test": do_test, "xeval": do_xeval, "beam": do_beam}


class Lock:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.fh = self.path.open("a")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def read_states(r: Run) -> dict:
    out = {}
    for p in r.state.glob("*.json"):
        try:
            out[p.stem] = json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return out


STALE_S = 20 * 60
TERMINAL = ("done", "failed", "blocked")


def claim(r: Run, units: list[dict], hours_left: float, lane: str) -> tuple[dict | None, str]:
    with Lock(r.state / ".lock"):
        states = read_states(r)
        now = time.time()
        for uid, st in states.items():  # a lane killed mid-unit leaves a stale heartbeat
            if st["state"] == "running":
                hb = r.state / f"{uid}.hb"
                if now - (hb.stat().st_mtime if hb.exists() else st["t0"]) > STALE_S:
                    # the lane died (node failure, preemption, wall time): not the unit's fault, so it does not use up
                    # the one retry; training units resume from their last epoch checkpoint (runs/<run>/resume)
                    st["transient"] = st.get("transient", 0) + 1
                    st["attempts"] = max(0, st.get("attempts", 1) - 1)
                    failed = st["transient"] > TRANSIENT_MAX
                    st.update(state="failed" if failed else "pending", msg=f"stale heartbeat (lane {st.get('lane')}, job {st.get('job')})")
                    write_json(r.state / f"{uid}.json", st)
                    alert(r, uid, f"{'FAILED after' if failed else 'requeued after'} lane death #{st['transient']} (job {st.get('job')})")
        states = read_states(r)
        by_id = {u["id"]: u for u in units}
        changed = True
        while changed:  # propagate failures
            changed = False
            for u in units:
                if states.get(u["id"], {}).get("state") in TERMINAL + ("running",):
                    continue
                bad = [d for d in u["deps"] if states.get(d, {}).get("state") in ("failed", "blocked")]
                if bad:
                    states[u["id"]] = {"state": "blocked", "msg": f"dependency failed: {bad}", "t0": now}
                    write_json(r.state / f"{u['id']}.json", states[u["id"]])
                    changed = True
        ready = [u for u in units if states.get(u["id"], {}).get("state") in (None, "pending")
                 and all(states.get(d, {}).get("state") == "done" for d in u["deps"])
                 and all(states.get(d, {}).get("state") in TERMINAL for d in u.get("after", ()))]
        pending = [u for u in units if states.get(u["id"], {}).get("state") not in TERMINAL]
        if not pending:
            return None, "finished"
        fits = [u for u in ready if u["est_hours"] * 1.5 + 0.1 < hours_left]
        if not fits:
            return None, ("resubmit" if ready else "wait")
        pick = max(fits, key=lambda u: (u["est_hours"], -units.index(u)))  # longest first
        prev = states.get(pick["id"], {})
        rec = {"state": "running", "lane": lane, "job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
               "t0": now, "attempts": prev.get("attempts", 0) + 1}
        write_json(r.state / f"{pick['id']}.json", rec)
        (r.state / f"{pick['id']}.hb").touch()
        return by_id[pick["id"]], "run"


def heartbeat(path: Path, stop: threading.Event) -> None:
    while not stop.wait(60):
        path.touch()


def execute(r: Run, u: dict, lane: str) -> None:
    rec = json.loads((r.state / f"{u['id']}.json").read_text())
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(r.state / f"{u['id']}.hb", stop), daemon=True).start()
    t0 = time.time()
    os.environ["FFT_UNIT"] = u["id"]  # sanity records of this unit (also inherited by the training subprocess)
    try:
        if u["kind"] == "collect":
            result = collect(r)
        else:
            result = KINDS[u["kind"]](r, u)
        rec.update(state="done", result=result)
    except Exception as exc:  # noqa: BLE001 - recorded, lanes continue with other units
        import traceback
        # a Python error is retried once, then FAILED; divergence is FAILED_DIVERGED at once (never retried)
        retry = rec.get("attempts", 1) < 2 and not isinstance(exc, (AssertionError, KeyError, Diverged))
        rec.update(state="pending" if retry else "failed", msg=f"{type(exc).__name__}: {exc}"[-3000:], trace=traceback.format_exc()[-4000:])
        if isinstance(exc, Diverged):
            rec["failure"] = "FAILED_DIVERGED"
        if not retry:
            alert(r, u["id"], f"{rec.get('failure', 'FAILED')}: {type(exc).__name__}: {str(exc)[:300]}")
        print(f"UNIT_FAILED {u['id']}: {exc}", flush=True)
    finally:
        stop.set()
    rec.update(t1=time.time(), hours=round((time.time() - t0) / 3600, 3))
    with Lock(r.state / ".lock"):
        write_json(r.state / f"{u['id']}.json", rec)
    print(f"UNIT_{rec['state'].upper()} {u['id']} {rec['hours']} h", flush=True)


def ensure_plan(out: Path) -> None:
    with Lock(out / ".plan.lock"):
        if (out / "units.json").exists():
            return
        cfg = json.loads((out / "config.json").read_text())
        write_json(out / "units.json", plan(cfg["arch"], bool(cfg.get("smoke")), only=cfg.get("smoke_families")))


def ensure_cross_eval(out: Path) -> None:
    """touch OUT/CROSS_EVAL (then start lanes) to append the optional cross-lingual units to a run's DAG."""
    if not (out / "CROSS_EVAL").exists():
        return
    with Lock(out / ".plan.lock"):
        units = json.loads((out / "units.json").read_text())
        if any(u["kind"] == "xeval" for u in units):
            return
        cfg = json.loads((out / "config.json").read_text())
        extra = [u for u in plan(cfg["arch"], bool(cfg.get("smoke")), cross_eval=True, only=cfg.get("smoke_families")) if u.get("optional")]
        write_json(out / "units.json", units + extra)


def cmd_lane(args) -> None:
    out = Path(args.out)
    ensure_plan(out)
    ensure_cross_eval(out)
    r = Run(out)
    units = r.units()
    rebind_protocols(r)
    lane = os.environ.get("LANE", "0")
    import signal
    # scancel / wall time: leave the unit "running" (a stale heartbeat or relane.sh requeues it), not "failed"
    signal.signal(signal.SIGTERM, lambda *_: os._exit(143))
    hours = float(os.environ.get("LANE_HOURS", "47.5"))
    start = time.time()
    idle_exit = float(os.environ.get("IDLE_EXIT_MIN", "480")) * 60
    idle_since = None
    print(f"LANE {lane} job={os.environ.get('SLURM_JOB_ID')} host={socket.gethostname()} gpu={gpu_name()} arch={r.arch} out={out}", flush=True)
    while True:
        cap = (out / "MAX_LANES").read_text().strip() if (out / "MAX_LANES").exists() else None
        stop_txt = (out / "STOP").read_text() if (out / "STOP").exists() else None
        # "jobs <id> <id> ..." in STOP stops only those lane jobs (retiring lanes that run older code); any other STOP stops all
        stop_me = stop_txt is not None and (not stop_txt.startswith("jobs ") or os.environ.get("SLURM_JOB_ID", "") in stop_txt.split()[1:])
        if stop_me or (cap is not None and int(lane) >= int(cap)):
            print("LANE_STOP (STOP file or MAX_LANES)", flush=True)
            break
        u, why = claim(r, units, hours - (time.time() - start) / 3600, lane)
        if u is not None:
            idle_since = None
            print(f"UNIT_START {u['id']} est={u['est_hours']} h", flush=True)
            execute(r, u, lane)
            write_status(r)
            continue
        if why == "finished":
            print("LANE_FINISHED", flush=True)
            break
        if why == "resubmit":
            resubmit(out, lane)
            break
        # nothing runnable (a barrier such as select-<family>): free the GPU after IDLE_EXIT_MIN; rebalance.sh (Mac)
        # hands free GPUs to the architecture with the most work left once units become ready again
        idle_since = idle_since or time.time()
        write_status(r)
        if time.time() - idle_since > idle_exit:
            print("LANE_IDLE_EXIT", flush=True)
            break
        time.sleep(60)
    write_status(r)


def resubmit(out: Path, lane: str) -> None:
    cfg = json.loads((out / "config.json").read_text())
    cmd = ["sbatch", "--parsable", f"--job-name={os.environ.get('SLURM_JOB_NAME', 'fft-lane')}", f"--time={cfg.get('lane_time', '48:00:00')}", f"--nice={cfg.get('nice', 0)}",
           f"--export=ALL,OUT={out},LANE={lane},LANE_HOURS={os.environ.get('LANE_HOURS', '47.5')}", str(HERE / "lane.sbatch")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"LANE_RESUBMIT {res.stdout.strip()} {res.stderr.strip()}", flush=True)


def write_status(r: Run) -> None:
    units, states = r.units(), read_states(r)
    lines = [f"{time.strftime('%Y-%m-%d %H:%M:%S')} arch={r.arch} out={r.out}"]
    counts = {}
    for u in units:
        s = states.get(u["id"], {}).get("state", "waiting")
        counts[s] = counts.get(s, 0) + 1
    ready = sum(1 for u in units if states.get(u["id"], {}).get("state") in (None, "pending")
                and all(states.get(d, {}).get("state") == "done" for d in u["deps"])
                and all(states.get(d, {}).get("state") in TERMINAL for d in u.get("after", ())))
    lines.append(" ".join(f"{k}={v}" for k, v in sorted(counts.items())) + f" ready={ready}" +
                 f" | est GPU-h left {sum(u['est_hours'] for u in units if states.get(u['id'], {}).get('state') != 'done'):.1f}")
    for u in units:
        st = states.get(u["id"], {})
        s = st.get("state", "waiting")
        extra = ""
        if s == "running":
            extra = f"lane {st.get('lane')} job {st.get('job')} {st.get('host')} since {time.strftime('%H:%M', time.localtime(st['t0']))}"
        elif s == "done":
            res = st.get("result") or {}
            extra = f"{st.get('hours')} h " + (f"lr={res.get('lr')} best_epoch={res.get('best_epoch', res.get('epoch'))} val={res.get('best_val', res.get('val'))}"
                                             if isinstance(res, dict) and ("lr" in res) else "")
        elif s in ("failed", "blocked", "pending"):
            extra = (st.get("msg") or "")[:300].replace("\n", " ")
        lines.append(f"{s:8s} {u['id']:45s} est {u['est_hours']:6.2f} h  {extra}")
    (r.out / "STATUS.txt").write_text("\n".join(lines) + "\n")


def cmd_run(args) -> None:
    out = Path(args.out)
    ensure_plan(out)
    r = Run(out)
    u = next(x for x in r.units() if x["id"] == args.unit)
    with Lock(r.state / ".lock"):
        write_json(r.state / f"{u['id']}.json", {"state": "running", "lane": "manual", "t0": time.time(), "attempts": 1,
                                                  "job": os.environ.get("SLURM_JOB_ID")})
    execute(r, u, "manual")
    write_status(r)


def cmd_status(args) -> None:
    r = Run(Path(args.out))
    write_status(r)
    print((r.out / "STATUS.txt").read_text())


# ------------------------------------------------------------------------------------------------------ collect
def bootstrap(values: list[float], stat, n_boot: int = 1000, seed: int = 20260926) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]


def items_for(rec: dict, lang: str) -> tuple[list, object] | None:
    """Per-item units and the statistic that reproduces the reported score from them (for bootstrap CIs)."""
    fam, raw = rec["family"], Path(rec["raw"])
    from collections import Counter

    def weighted_f1(pairs):
        gold, pred = Counter(g for g, _ in pairs), Counter(p for _, p in pairs)
        tp = Counter(g for g, p in pairs if g == p)
        return 100 * sum(s * 2 * tp[l] / (gold[l] + pred[l]) for l, s in gold.items() if gold[l] + pred[l]) / len(pairs)

    if fam in ("news", "sib"):
        d = json.loads(raw.with_suffix(".json").read_text())
        return [(x["gold"], x["prediction"]) for x in d["rows"] if x["language"] == lang], weighted_f1
    if fam in ("t2x", "afrihg"):
        from sacrebleu.metrics import CHRF
        import unicodedata
        task = "t2x_xho" if fam == "t2x" else f"afrihg_{lang}"
        ex = [json.loads(line) for line in (raw / task / "examples.jsonl").open()]
        items = [(unicodedata.normalize("NFC", e["prediction"]), [unicodedata.normalize("NFC", x) for x in e["debug"]["references"]]) for e in ex]

        def chrf(its):
            width = max(len(r) for _, r in its)
            return CHRF().corpus_score([h for h, _ in its], [[r[i] if i < len(r) else r[0] for _, r in its] for i in range(width)]).score
        return items, chrf
    if seq_kind(fam) == "pos":
        d = json.loads(raw.with_suffix(".json").read_text())
        rows = [x for x in d["rows"] if x.get("language") == lang or str(x.get("task", "")).startswith(lang)]
        if rows and "correct" in rows[0]:
            return [(sum(x["correct"]), len(x["gold"])) for x in rows], lambda its: 100 * sum(c for c, _ in its) / sum(t for _, t in its)
    return None  # NER span F1, Intent and the lm-eval tasks: point estimate only (bootstrap needs their row parsers)


def collect(r: Run) -> dict:
    """One row per (task, language, regime, model run): score, n items, bootstrap 95% CI where rows allow."""
    rows = []
    model = PAPER_MODEL[r.arch]
    for done in sorted((r.out / "runs").glob("*/RUN_DONE.json")) + sorted((r.out / "runs").glob("*/TEST_DONE.json")):
        d = json.loads(done.read_text())
        test = d.get("test")
        if not test:
            continue
        meta = json.loads((done.parent / "RUN_DONE.json").read_text())
        regime = {"mono": "Mono", "multi": "Multi", "general": "General"}[meta["regime"]]
        for fam in test:
            rec_path = r.out / "test" / done.parent.name / f"{fam}.score.json"
            rec = json.loads(rec_path.read_text())
            for lang, pts in rec["per_lang"].items():
                task_fam, lg = (lang.split(":") if ":" in lang else (fam, lang))
                ci = ("", "")
                try:
                    got = items_for(rec, lg) if fam not in ("transfer", "belebele", "intent", "ner", "nchlt_ner") else None
                    if got and got[0]:
                        ci = tuple(round(x, 4) for x in bootstrap(got[0], got[1]))
                except Exception as exc:  # noqa: BLE001
                    ci = (f"error: {exc}"[:80], "")
                rows.append({"model": model, "task": PAPER_TASK[task_fam], "language": lg, "family": LANG_FAMILY[lg], "regime": regime,
                             "train_language": "+".join(meta["langs"]) if regime == "Mono" else "",
                             "metric": METRIC[task_fam], "score_points": f"{pts:.6f}", "ci95_low": ci[0], "ci95_high": ci[1],
                             "n_items": rec["n"].get(lang), "lr": meta["lr"], "seed": meta["seed"], "epoch": meta["best_epoch"],
                             "run": done.parent.name, "gpu": rec.get("gpu"), "limit": rec.get("limit"),
                             "decoding": "greedy" if fam in ("t2x", "afrihg") else "", "note": ""})
    for b in sorted((r.out / "runs").glob("*/BEAM_DONE.json")):  # test-only beam decoding, beside greedy
        d, meta = json.loads(b.read_text()), json.loads((b.parent / "RUN_DONE.json").read_text())
        regime = {"mono": "Mono", "multi": "Multi", "general": "General"}[meta["regime"]]
        common = {"model": model, "regime": regime, "train_language": "+".join(meta["langs"]) if regime == "Mono" else "", "metric": "chrf",
                  "lr": meta["lr"], "seed": meta["seed"], "epoch": meta["best_epoch"], "run": b.parent.name}
        if "status" in d["test"]:
            for fam, langs in d["gen"].items():
                for lg in langs:
                    rows.append({**common, "task": PAPER_TASK[fam], "language": lg, "family": LANG_FAMILY[lg], "score_points": "",
                                 "ci95_low": "", "ci95_high": "", "n_items": "", "gpu": "", "limit": "", "decoding": "beam",
                                 "note": d["test"]["status"]})
            continue
        for fam in d["test"]:
            rec = json.loads((r.out / "test" / f"beam-{b.parent.name}" / f"{fam}.score.json").read_text())
            for lg, pts in rec["per_lang"].items():
                ci = ("", "")
                got = items_for(rec, lg)
                if got and got[0]:
                    ci = tuple(round(x, 4) for x in bootstrap(got[0], got[1]))
                rows.append({**common, "task": PAPER_TASK[fam], "language": lg, "family": LANG_FAMILY[lg], "score_points": f"{pts:.6f}",
                             "ci95_low": ci[0], "ci95_high": ci[1], "n_items": rec["n"].get(lg), "gpu": rec.get("gpu"),
                             "limit": rec.get("limit"), "decoding": "beam", "note": rec.get("decoding", "")})
    base = r.out / "base_eval"
    for fam in ("t2x", "afrihg"):
        p = base / "gen" / f"{fam}.score.json"
        if p.exists():
            rec = json.loads(p.read_text())
            for lang, pts in rec["per_lang"].items():
                rows.append({"model": model, "task": PAPER_TASK[fam], "language": lang, "family": LANG_FAMILY[lang], "regime": "Base",
                             "train_language": "", "metric": "chrf",
                             "score_points": f"{pts:.6f}", "ci95_low": "", "ci95_high": "", "n_items": rec["n"][lang], "lr": "",
                             "seed": "", "epoch": "", "run": "base-gen", "gpu": rec.get("gpu"), "limit": rec.get("limit"),
                             "decoding": "greedy", "note": ""})
    for x in sorted((r.out / "runs").glob("*/XEVAL_DONE.json")):  # optional cross-lingual transfer matrix
        d, meta = json.loads(x.read_text()), json.loads((x.parent / "RUN_DONE.json").read_text())
        rec = json.loads((r.out / "test" / f"xeval-{x.parent.name}" / f"{meta['family']}.score.json").read_text())
        for lang, pts in d["test"].items():
            rows.append({"model": model, "task": PAPER_TASK[meta["family"]], "language": lang, "family": LANG_FAMILY[lang],
                         "regime": "Mono-crosslingual", "train_language": d["train_language"], "metric": METRIC[meta["family"]],
                         "score_points": f"{pts:.6f}", "ci95_low": "", "ci95_high": "", "n_items": rec["n"].get(lang), "lr": meta["lr"],
                         "seed": meta["seed"], "epoch": meta["best_epoch"], "run": x.parent.name, "gpu": rec.get("gpu"),
                         "limit": rec.get("limit"), "decoding": "greedy" if meta["family"] in ("t2x", "afrihg") else "", "note": ""})
    out = r.out / "results" / "cells.csv"
    if rows:
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return {"cells": len(rows), "csv": str(out)}


def cmd_collect(args) -> None:
    print(json.dumps(collect(Run(Path(args.out))), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("matrix")
    m.add_argument("--arch", choices=sorted(ARCHS))
    m.add_argument("--smoke", action="store_true")
    m.add_argument("--csv")
    m.add_argument("--simulate", type=int, nargs="*", help="print list-scheduling wall-clock for these GPU counts")
    for name in ("lane", "status", "collect"):
        sub.add_parser(name).add_argument("out")
    rr = sub.add_parser("run")
    rr.add_argument("out")
    rr.add_argument("unit")
    args = ap.parse_args()
    {"matrix": cmd_matrix, "lane": cmd_lane, "run": cmd_run, "status": cmd_status, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
