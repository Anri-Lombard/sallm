#!/usr/bin/env python3
"""Checkpoint re-selection driver: per-epoch validation scoring, selection, and General-protocol test scoring.

  reselect.py val  <label>                      score VALIDATION for every checkpoint-* (resumable), then select
  reselect.py test <label> [--adapter PATH] [--out PATH]

Scorers run as subprocesses with the task's own runtime (kit/news_score.py, kit/sib_score.py, kit/intent_eval.py,
kit/run_seq_wrapped.py around the frozen NER v2 runner). Roots can be redirected with RESELECT_TRAIN_ROOT,
RESELECT_VAL_ROOT, RESELECT_TEST_ROOT (used by the smoke tests)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HOST = os.environ.get("RESELECT_HOST", "hex" if Path("/scratch/lmbanr001").is_dir() else "kombuys")
ROOT = Path({"hex": "/scratch/lmbanr001/masters/sallm/results/monomulti_reselect_20260925",
             "kombuys": "/scratch/alombard/sallm/results/monomulti_reselect_20260925"}[HOST])
KIT = ROOT / "jobs" / "kit"
TRAIN_ROOT = Path(os.environ.get("RESELECT_TRAIN_ROOT", ROOT / "train"))
VAL_ROOT = Path(os.environ.get("RESELECT_VAL_ROOT", ROOT / "val"))
TEST_ROOT = Path(os.environ.get("RESELECT_TEST_ROOT", ROOT / "test"))

S = "/scratch/lmbanr001/masters/sallm"
SNAP = "/scratch/lmbanr001/masters/sallm_snapshots"
V8 = f"{SNAP}/downstream-generation-20260914-v8" if HOST == "hex" else "/scratch/alombard/sallm_snapshots/downstream-generation-20260914-v8"
HF = "/scratch/lmbanr001/hf-cache" if HOST == "hex" else "/scratch/alombard/sallm/hf"
BUNDLE = f"{SNAP}/general-sequence-official-test-20260916-v1"
REPAIR = f"{S}/results/general_sequence_official_test_20260917_v2/control"
SEQ_PROTOCOL = Path(f"{BUNDLE}/general_sequence_validation_hex_protocol_20260915_v3.json")
SEQ_PROTOCOL_EOSFIX = Path(f"{S}/results/mamba_eos_fix_20260924/protocols/general_ner_protocol_eosfix.json")
SEQ_SELECTION = Path(f"{S}/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/selection/SELECTION.json")
SEQ_RELEASE = Path(f"{S}/results/general_sequence_official_test_20260916_v1/release/NER_TEST_ACCESS_RELEASED_V1.json")
# Kombuys: one venv (patched xLSTM backend 5f20208d, as the HEX xLSTM runtime venv) and the Kombuys bases of protocol.md;
# every base must have the same tree sha256 prefix as on HEX (BASE_TREES).
PY = {"hex": {"default": "/home/lmbanr001/masters/sallm/.venv/bin/python",
              "xlstm": f"{S}/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python"},
      "kombuys": {"default": "/scratch/alombard/sallm/.venv/bin/python", "xlstm": "/scratch/alombard/sallm/.venv/bin/python"}}[HOST]
BASES = {"hex": {
    "gdn": f"{S}/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model",
    "mzansilm": f"{S}/results/downstream_standardized_20260913_v1/bases/mzansilm",
    "mamba2": f"{S}/results/downstream_standardized_20260913_v1/bases/mamba2",
    "xlstm": f"{S}/results/downstream_standardized_20260913_v1/bases/xlstm",
}, "kombuys": {
    "gdn": "/scratch/alombard/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model",
    "mzansilm": "/scratch/alombard/sallm/results/retained_standardized_20260913_v1/bases/mzansilm",
    "mamba2": "/scratch/alombard/sallm/results/retained_standardized_20260913_v1/bases/mamba2",
    "xlstm": "/scratch/alombard/sallm/results/retained_standardized_20260913_v1/bases/xlstm",
}}[HOST]
BASE_TREES = {"gdn": "5ceca7d0", "mzansilm": "aab89ea6", "mamba2": "5db6860f", "xlstm": "ff2e7c99", "mamba2_eosfix": "d069732e"}
MAMBA_NER_BASE = f"{S}/results/mamba_eos_fix_20260924/bases/mamba2_eosfix"
LANGS = {"news": ("eng", "xho"), "sib": ("afr", "eng", "nso", "sot", "xho", "zul"), "intent": ("eng", "sot", "xho", "zul"), "ner": ("tsn", "xho", "zul")}
PROMPTS = {"news": {"eng": "p2", "xho": "p4"}, "sib": {"afr": "p4", "eng": "p3", "nso": "p4", "sot": "p3", "xho": "p5", "zul": "p5"},
           "intent": {"eng": "p4", "sot": "p1", "xho": "p2", "zul": "p2"}, "ner": {"tsn": "P2", "xho": "P5", "zul": "P5"}}
METRIC = {"news": "support_weighted_f1", "sib": "support_weighted_f1", "intent": "lm_eval_weighted_f1", "ner": "micro_span_f1"}
KIT_SHA = {
    "news_v15/scripts/run_mamba_news_common_validation.py": "a620218a",
    "news_v15/scripts/run_mamba_news_interface_diagnostic.py": "28d9706a",
    "news_v15/src/main/sallm/evaluation/classification_metrics.py": "8dc06c3f",
    "sib/run_sib_natural_first_token_eval.py": "b247505a",
    "sib/run_sib_balanced_code_eval.py": "866e9d82",
}
# Feb-era Mono Intent retrains (recipe cf0ed06) train on the full upstream train split, so the carve-from-train rows are
# inside their training data; they are validated on the upstream dev.jsonl instead. Every other Intent label uses the carve.
UPSTREAM_DEV_LABELS = {f"{a}_intent_mono_{l}" for a in ("mamba2", "mzansilm") for l in ("sot", "xho", "zul")}
FEB_VAL_SPEC = ROOT / "tmp" / "mm_recipe" / "feb_intent_val.json"
SEQ_FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "chat_template.jinja", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(root: Path) -> str:
    root = root.resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts)
    return hashlib.sha256("".join(f"{sha(p)}  ./{p.relative_to(root).as_posix()}\n" for p in files).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=False) + "\n")
    tmp.replace(path)


def parse_label(label: str) -> dict:
    parts = label.split("_")
    arch, task, regime = parts[0], parts[1], parts[2]
    assert arch in BASES and task in LANGS and regime in ("mono", "multi"), label
    langs = list(LANGS[task]) if regime == "multi" else [parts[3]]
    assert all(lang in LANGS[task] for lang in langs), label
    assert HOST == "hex" or task != "ner", "NER is scored on HEX only"
    base = MAMBA_NER_BASE if (arch == "mamba2" and task == "ner") else BASES[arch]
    return {"label": label, "arch": arch, "task": task, "regime": regime, "langs": langs, "base": os.environ.get("RESELECT_BASE", base)}


def val_source(info: dict) -> str:
    task = info["task"]
    if task == "intent":
        return "upstream_dev" if info["label"] in UPSTREAM_DEV_LABELS else "carve"
    return {"news": "masakhanews_dev_tsv_fa3b5fff", "sib": "sib200_validation_38977a66", "ner": "masakhaner2_validation_v2_runner"}[task]


def check_val_source(info: dict, source: str) -> None:
    """Rows must match their manifest; for upstream_dev also cross-check the other agent's spec when it exists."""
    if info["task"] != "intent":
        return
    rows_dir = KIT / ("intent_val" if source == "carve" else "intent_val_upstream_dev")
    manifest = json.loads((rows_dir / "MANIFEST.json").read_text())
    for lang in info["langs"]:
        entry = manifest["languages"][lang]
        assert sha(rows_dir / f"{lang}.jsonl") == entry["file_sha256"], rows_dir / f"{lang}.jsonl"
        if source == "upstream_dev" and FEB_VAL_SPEC.exists():
            spec = FEB_VAL_SPEC.read_text()
            assert entry["source_sha256"] in spec, f"{FEB_VAL_SPEC} does not list dev sha {entry['source_sha256']} for {lang}"


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def env_for(task: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    work = ROOT / "tmp" / "work" / f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    env.update({"HF_HOME": HF, "HF_DATASETS_CACHE": f"{HF}/datasets", "HF_HUB_CACHE": f"{HF}/hub", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_XET": "1", "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false", "PYTHONHASHSEED": "42",
                "PYTHONDONTWRITEBYTECODE": "1", "FLA_DISABLE_BACKEND_DISPATCH": "1", "MAMBA_SCAN_IMPL": "cuda", "TMPDIR": str(work)})
    if task == "news":
        env.update({"PYTHONPATH": str(KIT / "news_v15/src/main"), "SALLM_SKIP_MAMBA_KERNEL_CHECK": "1"})
    elif task in ("sib", "intent"):
        env["PYTHONPATH"] = f"{V8}/src/main"
    else:
        env.update({"PYTHONPATH": f"{V8}/src/main:{BUNDLE}:{REPAIR}", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True"})
        # The v2 runner's load_dataset(anrilombard/masakhaner-x-parquet, data_files=..., revision=...) only finds its cached
        # config when the revision resolves; General NER (eval_unit.sbatch) ran without the offline flags, so match that.
        for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
            env.pop(key)
    if os.environ.get("RESELECT_MEMCAP"):
        env["PYTHONPATH"] = str(KIT.parent / "memcap") + ":" + env["PYTHONPATH"]
    return env


def check_kit(info: dict) -> None:
    for rel, prefix in KIT_SHA.items():
        assert sha(KIT / rel).startswith(prefix), rel
    if "RESELECT_BASE" not in os.environ:
        key = "mamba2_eosfix" if info["base"] == MAMBA_NER_BASE else info["arch"]
        assert tree_sha256(Path(info["base"])).startswith(BASE_TREES[key]), info["base"]


def seq_protocol(info: dict, adapter: Path, out: Path) -> Path:
    src_path = SEQ_PROTOCOL_EOSFIX if info["arch"] == "mamba2" else SEQ_PROTOCOL
    src = json.loads(src_path.read_text())
    p = copy.deepcopy(src)
    m = p["models"][info["arch"]]
    assert Path(m["base_path"]) == Path(info["base"]), (m["base_path"], info["base"])
    m["adapter_path"] = str(adapter)
    m["adapter_files"] = {f: sha(adapter / f) for f in SEQ_FILES if (adapter / f).is_file()}
    m["adapter_tree_sha256"] = tree_sha256(adapter)
    # As Phase 2 (monomulti_rescore_20260924/protocols/mzansilm_ner_mono_*.json): an adapter with modules_to_save
    # (embed_tokens/lm_head on a resized vocab) cannot load unmerged in lm-eval, so it is merged.
    info["merge_lora_override"] = bool(json.loads((adapter / "adapter_config.json").read_text()).get("modules_to_save")) and not m["merge_lora"]
    if info["merge_lora_override"]:
        m["merge_lora"] = True
    nonmodel = lambda d: json.dumps({k: v for k, v in d.items() if k != "models"}, sort_keys=True)  # noqa: E731
    assert nonmodel(p) == nonmodel(src)
    assert all(p["models"][a] == src["models"][a] for a in p["models"] if a != info["arch"])
    write_json(out, p)
    return out


def run_scorer(info: dict, adapter: Path, split: str, raw: Path, langs: list[str], log: Path, source: str = "carve") -> tuple[Path, float]:
    """Run the task scorer once; returns (raw output path, wall seconds). Reuses a complete raw output."""
    task, arch = info["task"], info["arch"]
    py = PY["xlstm"] if arch == "xlstm" else PY["default"]
    env = env_for(task)
    if task == "news":
        out = raw.with_suffix(".json")
        cmd = [py, str(KIT / "news_score.py"), "--arch", arch, "--base", info["base"], "--adapter", str(adapter), "--split", split,
               "--langs", ",".join(langs), "--out", str(out)]
        done = out
    elif task == "sib":
        out = raw.with_suffix(".json")
        cmd = [py, str(KIT / "sib_score.py"), "--arch", arch, "--base", info["base"], "--adapter", str(adapter), "--split", split,
               "--langs", ",".join(langs), "--out", str(out)]
        done = out
    elif task == "intent":
        out = raw
        cmd = [py, str(KIT / "intent_eval.py"), "--arch", arch, "--base", info["base"], "--adapter", str(adapter), "--split", split,
               "--langs", ",".join(langs), "--output", str(out)]
        if split == "validation":
            cmd += ["--val-source", source]
        done = out / "SUMMARY.json"
    else:
        out = raw.with_suffix(".json")
        protocol = seq_protocol(info, adapter, raw.with_suffix(".protocol.json"))
        cmd = [py, str(KIT / "run_seq_wrapped.py"), "--task", "ner", "--phase", split, "--architecture", arch, "--checkpoint", info["base"],
               "--adapter", str(adapter), "--protocol", str(protocol), "--output", str(out)]
        if split == "validation":
            env.update({"SEQ_VALIDATION_PROMPTS": "protocol", "SEQ_LANGUAGES": ",".join(langs)})
        else:
            cmd += ["--selection", str(SEQ_SELECTION), "--release", str(SEQ_RELEASE)]
        done = out
    if done.exists():
        return out, 0.0
    if out.exists():
        out.rename(out.with_name(f"{out.name}.partial-{int(time.time())}"))
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("a") as handle:
        handle.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(cmd)}\n")
        handle.flush()
        rc = subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
    shutil.rmtree(env["TMPDIR"], ignore_errors=True)
    if rc != 0 or not done.exists():
        raise SystemExit(f"scorer failed rc={rc}; see {log}")
    return out, time.monotonic() - started


def pred_stats(preds: list[str]) -> dict:
    top = Counter(preds).most_common(1)[0]
    return {"distinct_predicted_labels": len(set(preds)), "top_prediction": top[0], "top_prediction_share": top[1] / len(preds)}


def read_scores(info: dict, out: Path, langs: list[str]) -> dict:
    """Per-language {score (0-100), n_items, ...diagnostics} from a scorer output."""
    task = info["task"]
    res = {}
    if task in ("news", "sib"):
        d = json.loads(out.read_text())
        key = "weighted_f1" if task == "news" else "f1"
        for lang in langs:
            preds = [r["prediction"] for r in d["rows"] if r["language"] == lang]
            res[lang] = {"score": 100 * float(d["languages"][lang][key]), "n_items": len(preds), **pred_stats(preds)}
    elif task == "intent":
        d = json.loads((out / "SUMMARY.json").read_text())
        for lang in langs:
            v = d["languages"][lang]
            res[lang] = {"score": 100 * v["weighted_f1"], "n_items": v["n_items"], "distinct_predicted_labels": v["distinct_predicted_labels"],
                         "top_prediction": v["top_prediction"], "top_prediction_share": v["top_prediction_share"]}
    else:
        d = json.loads(out.read_text())
        for name, ev in d["task_evidence"].items():
            lang = ev["language"]
            if lang not in langs:
                continue
            rows = [r for r in d["rows"] if r["task"] == name]
            res[lang] = {"score": 100 * float(d["reported_metrics"][name]), "n_items": len(rows), "task": name,
                         "pct_rows_loop": 100 * sum(len(r["raw_response"]) > 600 for r in rows) / len(rows),
                         "pct_blank": 100 * sum(not r["prediction"].strip() for r in rows) / len(rows)}
    assert set(res) == set(langs), (sorted(res), langs)
    return res


def training_complete(label: str) -> bool:
    return (TRAIN_ROOT / label / "final_adapter" / "adapter_config.json").is_file()


def checkpoints(label: str) -> list[tuple[int, Path]]:
    """While training runs, only checkpoints with trainer_state.json (the Trainer writes it after the weights) count."""
    complete = training_complete(label)
    found = []
    for p in (TRAIN_ROOT / label).glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and p.is_dir() and (p / "adapter_config.json").is_file() and (complete or (p / "trainer_state.json").is_file()):
            found.append((int(m.group(1)), p))
    return sorted(found)


def epoch_of(ckpt: Path):
    state = ckpt / "trainer_state.json"
    return json.loads(state.read_text()).get("epoch") if state.is_file() else None


def cmd_val(label: str) -> None:
    info = parse_label(label)
    check_kit(info)
    vdir = VAL_ROOT / label
    source = val_source(info)
    check_val_source(info, source)
    ckpts = checkpoints(label)
    if not ckpts:
        raise SystemExit(f"no checkpoints under {TRAIN_ROOT / label}")
    if not training_complete(label) and os.environ.get("RESELECT_ALLOW_PARTIAL") != "1":
        raise SystemExit(f"{label}: final_adapter missing; set RESELECT_ALLOW_PARTIAL=1 to score the finished checkpoints so far")
    gpu = gpu_name()
    for step, ckpt in ckpts:
        target = vdir / f"{ckpt.name}.json"
        if target.exists():
            prior = json.loads(target.read_text()).get("val_source")
            if prior != source:
                raise SystemExit(f"{target} was scored on val_source={prior}, expected {source}")
            print(f"skip {ckpt.name} (scored)", flush=True)
            continue
        out, secs = run_scorer(info, ckpt, "validation", vdir / "raw" / ckpt.name, info["langs"], vdir / "logs" / f"{ckpt.name}.log", source)
        per = read_scores(info, out, info["langs"])
        record = {"checkpoint": ckpt.name, "step": step, "epoch": epoch_of(ckpt),
                  "per_lang": {k: v["score"] for k, v in per.items()},
                  "score": sum(v["score"] for v in per.values()) / len(per),
                  "n_items": {k: v["n_items"] for k, v in per.items()},
                  "diagnostics": {k: {x: y for x, y in v.items() if x not in ("score", "n_items")} for k, v in per.items()},
                  "label": label, "task": info["task"], "split": "validation", "val_source": source, "metric": METRIC[info["task"]],
                  "prompts": {k: PROMPTS[info["task"]][k] for k in info["langs"]}, "adapter": str(ckpt), "base": info["base"],
                  "raw_output": str(out), "scorer_wall_seconds": secs, "gpu": gpu, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "merge_lora_override": info.get("merge_lora_override", False), "host": HOST}
        write_json(target, record)
        print(f"VAL {label} {ckpt.name} epoch={record['epoch']} score={record['score']:.4f} {json.dumps(record['per_lang'])} {secs:.0f}s", flush=True)
    select(label, info, ckpts)


def select(label: str, info: dict, ckpts: list[tuple[int, Path]]) -> None:
    vdir = VAL_ROOT / label
    records = [json.loads((vdir / f"{c.name}.json").read_text()) for _, c in ckpts]
    records.sort(key=lambda r: r["step"])
    best = records[0]
    for r in records[1:]:
        if r["score"] > best["score"]:
            best = r
    complete = training_complete(label)
    if complete:
        # the checkpoint list may have been taken while training was still running: require every saved epoch to be scored
        now = [c.name for _, c in checkpoints(label)]
        last = json.loads((TRAIN_ROOT / label / now[-1] / "trainer_state.json").read_text())
        if now != [r["checkpoint"] for r in records] or last.get("global_step") != last.get("max_steps"):
            complete = False
            print(f"incomplete scoring: scored={[r["checkpoint"] for r in records]} on_disk={now}", flush=True)
    selection = {
        "label": label, "task": info["task"], "regime": info["regime"], "languages": info["langs"], "metric": METRIC[info["task"]],
        "rule": ("highest validation score; Multi = unweighted mean over the task's languages, Mono = own language; "
                 "exact ties -> earlier epoch (lower step)"),
        "training_complete": complete, "val_source": records[0]["val_source"],
        "selected": {"checkpoint": best["checkpoint"], "step": best["step"], "epoch": best["epoch"], "score": best["score"],
                     "per_lang": best["per_lang"], "adapter": str(TRAIN_ROOT / label / best["checkpoint"])},
        "scores_by_epoch": [{"checkpoint": r["checkpoint"], "step": r["step"], "epoch": r["epoch"], "score": r["score"], "per_lang": r["per_lang"]}
                            for r in records],
        "n_checkpoints": len(records),
    }
    write_json(vdir / ("selection.json" if complete else "selection.partial.json"), selection)
    print(f"SELECT {label} -> {best['checkpoint']} epoch={best['epoch']} score={best['score']:.4f} (training_complete={complete})", flush=True)
    if not complete:
        print("final_adapter missing: wrote selection.partial.json only", flush=True)


def cmd_test(label: str, adapter: str | None, out_path: str | None) -> None:
    info = parse_label(label)
    check_kit(info)
    selection = None
    if adapter is None:
        sel_path = VAL_ROOT / label / "selection.json"
        selection = json.loads(sel_path.read_text())
        adapter_path = Path(selection["selected"]["adapter"])
    else:
        adapter_path = Path(adapter)
    tree = tree_sha256(adapter_path)
    tag = label if adapter is None else f"{label}__{tree[:12]}"
    target = Path(out_path) if out_path else (TEST_ROOT / f"{label}.json" if adapter is None else TEST_ROOT / "adapter_override" / f"{tag}.json")
    if target.exists():
        raise SystemExit(f"{target} exists")
    score_langs = list(LANGS["ner"]) if info["task"] == "ner" else info["langs"]
    out, secs = run_scorer(info, adapter_path, "test", TEST_ROOT / "raw" / tag, score_langs, TEST_ROOT / "logs" / f"{tag}.log")
    per = read_scores(info, out, info["langs"])
    for lang, v in per.items():
        v["prompt"] = PROMPTS[info["task"]][lang]
    record = {"label": label, "task": info["task"], "regime": info["regime"], "arch": info["arch"], "split": "test", "metric": METRIC[info["task"]],
              "languages": per, "score_mean": sum(v["score"] for v in per.values()) / len(per),
              "adapter": str(adapter_path), "adapter_tree_sha256": tree, "adapter_override": adapter is not None,
              "selection": None if selection is None else {"path": str(VAL_ROOT / label / "selection.json"), **selection["selected"]},
              "base": info["base"], "raw_output": str(out), "log": str(TEST_ROOT / "logs" / f"{tag}.log"),
              "scorer_wall_seconds": secs, "gpu": gpu_name(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "merge_lora_override": info.get("merge_lora_override", False), "host": HOST}
    write_json(target, record)
    print(f"TEST {label} {json.dumps({k: round(v['score'], 4) for k, v in per.items()})} -> {target}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("val", "test"))
    ap.add_argument("label")
    ap.add_argument("--adapter")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.stage == "val":
        cmd_val(args.label)
    else:
        cmd_test(args.label, args.adapter, args.out)


if __name__ == "__main__":
    sys.exit(main())
