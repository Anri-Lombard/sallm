#!/usr/bin/env python3
"""Bookkeeping for the equal-recipe full fine-tuning T2X pilot (no model code here)."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/lmbanr001/masters/sallm_snapshots/full-matrix-execution-20260916-v1")
from prepare_bindings import tree_sha256  # noqa: E402

ROOT = Path("/scratch/lmbanr001/masters/sallm/results/equal_recipe_fullft_t2x_pilot_20260925")
ARCHS = ["mzansilm", "mamba2", "xlstm", "gdn"]
CORE = ["1e-5", "3e-5", "1e-4"]
LOW_EXT, HIGH_EXT = "3e-6", "3e-4"
TRAIN_ROWS = 3859
CHRF = "eval/t2x_xho/all_chrf"


def score(unit_dir: Path, rows_expected: int | None = None) -> dict:
    summary = json.loads((unit_dir / "evaluation_summary.json").read_text())
    rows = [json.loads(line) for line in (unit_dir / "t2x_xho" / "examples.jsonl").open()]
    if rows_expected is not None and len(rows) != rows_expected:
        raise RuntimeError(f"{unit_dir}: {len(rows)} rows != {rows_expected}")
    return {
        "chrf": summary[0]["metrics"][CHRF],
        "rows": len(rows),
        "empty": sum(bool(r["debug"]["empty_prediction"]) for r in rows),
        "hit_max_new_tokens": sum(bool(r["debug"]["hit_max_new_tokens"]) for r in rows),
        "no_eos": sum(not r["debug"]["ended_with_eos"] for r in rows),
        "max_new_tokens": rows[0]["debug"]["max_new_tokens"],
        "max_prompt_tokens": max(r["debug"]["input_token_count"] for r in rows),
        "generated_tokens": sum(r["debug"]["generated_token_count"] for r in rows),
    }


def timings(path: Path, prefix: str) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == prefix:
                out[parts[1]] = int(parts[2].split("=")[1])
    return out


def gen_tokens(run: Path, unit_id: str):
    f = run / "val" / unit_id / "greedy" / "t2x_xho" / "examples.jsonl"
    return sum(json.loads(line)["debug"]["generated_token_count"] for line in f.open()) if f.exists() else ""


def tokens(arch: str, run_info: dict) -> dict:
    if "train_tokens_per_epoch" in run_info:
        return run_info
    p = ROOT / "runs" / arch / "tokens.json"  # count-only pass for runs started before token accounting existed
    return json.loads(p.read_text()) if p.exists() else {}


def best(entries: list[dict]) -> dict:
    # max val chrF; ties -> smaller lr, then earlier epoch
    return min(entries, key=lambda e: (-e["val_chrf"], float(e["lr"]), e["epoch"]))


def cmd_tree(path: str) -> None:
    print(tree_sha256(Path(path)))


def cmd_valspec(arch: str, lr: str, out: str, *ckpts: str) -> None:
    units = []
    for epoch, ckpt in enumerate(sorted(ckpts, key=lambda c: int(c.rsplit("-", 1)[1])), 1):
        seed, bs = os.environ.get("SEED", "42"), os.environ.get("BS", "16")
        b = "" if bs == "16" else f"-b{bs}"
        units.append({"unit_id": f"val-fft-{arch}-s{seed}{b}-lr{lr}-e{epoch}", "architecture": arch,
                      "tasks": ["t2x_xho"], "base": ckpt, "base_tree_sha256": tree_sha256(Path(ckpt)),
                      "adapter": None, "adapter_tree_sha256": None, "lr": lr, "epoch": epoch})
    Path(out).write_text(json.dumps(units, indent=1) + "\n")


def cmd_rundone(arch: str, lr: str, precision: str, seed: str, run_dir: str, spec: str, state: str, info: str,
                disk: str) -> None:
    run = Path(run_dir)
    units = json.loads(Path(spec).read_text())
    log = json.loads(Path(state).read_text())["log_history"]
    losses = [h["loss"] for h in log if "loss" in h]
    grads = [h["grad_norm"] for h in log if "grad_norm" in h]
    # the final train_runtime entry is logged after the last checkpoint's trainer_state.json is written
    import ast, re
    final = [ast.literal_eval(m) for m in re.findall(r"\{'train_runtime': [^}]*\}", (run / "train.log").read_text())][-1]
    epochs = []
    for u in units:
        s = score(run / "val" / u["unit_id"] / "greedy", 460)
        epochs.append({"lr": lr, "epoch": u["epoch"], "val_chrf": s["chrf"], "val": s,
                       "checkpoint": u["base"], "tree_sha256": u["base_tree_sha256"]})
    sel = best(epochs)
    done = {
        "arch": arch, "lr": lr, "precision": precision, "seed": int(seed),
        "batch": int(os.environ.get("BS", "16")), "epochs": epochs, "best_epoch": sel["epoch"],
        "steps": max(h.get("step", 0) for h in log) if log else None,
        "global_step": json.loads(Path(state).read_text())["global_step"],
        "examples_seen": round(final["epoch"] * TRAIN_ROWS),
        "train_runtime_s": final["train_runtime"],
        "sec_per_step": final["train_runtime"] / json.loads(Path(state).read_text())["global_step"],
        "train_samples_per_second": final["train_samples_per_second"],
        "eval_loss_by_epoch": [h["eval_loss"] for h in log if "eval_loss" in h],
        "first_loss": losses[0] if losses else None, "last_loss": losses[-1] if losses else None,
        "max_grad_norm": max(grads) if grads else None,
        "nonfinite_logged": any(not math.isfinite(x) for x in losses + grads),
        "run_info": json.loads(Path(info).read_text()),
        "disk_bytes_per_checkpoint": json.loads(Path(disk).read_text()),
    }
    (run / "RUN_DONE.json").write_text(json.dumps(done, indent=1) + "\n")
    print(json.dumps({k: done[k] for k in ("arch", "lr", "best_epoch", "global_step", "train_runtime_s")}),
          [round(e["val_chrf"], 2) for e in epochs])
    print(f"BEST_CKPT {sel['checkpoint']}")


def load_runs(arch: str) -> list[dict]:  # seed-42 sweep only (seed runs live in s<seed>_lr* dirs)
    runs = [json.loads(p.read_text()) for p in sorted((ROOT / "runs" / arch).glob("lr*/RUN_DONE.json"))]
    return [e for r in runs for e in r["epochs"]]


def cmd_final(test_spec: str) -> None:
    entries = {a: load_runs(a) for a in ARCHS}
    core_best = {a: best([e for e in entries[a] if e["lr"] in CORE]) for a in ARCHS}
    grid = list(CORE)
    if any(w["lr"] == CORE[-1] for w in core_best.values()):
        grid.append(HIGH_EXT)
    if any(w["lr"] == CORE[0] for w in core_best.values()):
        grid.insert(0, LOW_EXT)
    units, selection = [], {"rule": "max greedy val chrF over (lr, epoch); ties -> smaller lr, earlier epoch",
                            "core_grid": CORE, "final_grid": grid,
                            "core_best": {a: [w["lr"], w["epoch"], w["val_chrf"]] for a, w in core_best.items()}}
    for a in ARCHS:
        sel = best([e for e in entries[a] if e["lr"] in grid])
        run = json.loads((ROOT / "runs" / a / f"lr{sel['lr']}" / "RUN_DONE.json").read_text())
        keep = ROOT / "keep" / a / f"lr{sel['lr']}_e{sel['epoch']}"
        tree = tree_sha256(keep)
        if tree != sel["tree_sha256"]:
            raise RuntimeError(f"{keep} tree {tree} != {sel['tree_sha256']}")
        selection[a] = {"lr": sel["lr"], "epoch": sel["epoch"], "val_chrf": sel["val_chrf"], "checkpoint": str(keep)}
        units.append({"unit_id": f"u-fft-mono-{a}-t2x-xho", "architecture": a, "tasks": ["t2x_xho"],
                      "base": str(keep), "base_tree_sha256": tree, "adapter": None, "adapter_tree_sha256": None,
                      "model_tree_sha256": tree, "lr": float(sel["lr"]), "epoch": sel["epoch"],
                      "val_chrf": sel["val_chrf"], "precision": run["precision"],
                      "n_trainable_params": run["run_info"]["n_trainable_params"],
                      "recipe": "equal-recipe full fine-tuning pilot 20260925 (see recipe.md)"})
    Path(test_spec).write_text(json.dumps(units, indent=1) + "\n")
    (ROOT / "SEEDS_PLAN.tsv").write_text("".join(f"{a}\t{selection[a]['lr']}\n" for a in ARCHS))
    (ROOT / "SELECTION.json").write_text(json.dumps(selection, indent=1) + "\n")
    print(json.dumps(selection, indent=1))


def cmd_extratest(arch: str, tag: str, lr: str, spec: str) -> None:
    """Extra seed or batch-size run (runs/<arch>/<tag>) at the selected lr: best epoch on val -> test unit."""
    run = json.loads((ROOT / "runs" / arch / tag / "RUN_DONE.json").read_text())
    sel = best(run["epochs"])
    keep = ROOT / "keep" / arch / f"{tag}_e{sel['epoch']}"
    tree = tree_sha256(keep)
    if tree != sel["tree_sha256"]:
        raise RuntimeError(f"{keep} tree {tree} != {sel['tree_sha256']}")
    suffix = tag.rsplit("_lr", 1)[0].replace("_", "-")
    unit = {"unit_id": f"u-fft-mono-{arch}-t2x-xho-{suffix}", "architecture": arch, "tasks": ["t2x_xho"],
            "base": str(keep), "base_tree_sha256": tree, "adapter": None, "adapter_tree_sha256": None,
            "model_tree_sha256": tree, "seed": run["seed"], "batch": run.get("batch", 16), "lr": float(lr),
            "epoch": sel["epoch"], "val_chrf": sel["val_chrf"], "precision": run["precision"],
            "n_trainable_params": run["run_info"]["n_trainable_params"],
            "recipe": f"equal-recipe full fine-tuning pilot 20260925 (see recipe.md); extra run {suffix} at the selected lr"}
    Path(spec).write_text(json.dumps([unit], indent=1) + "\n")


def cmd_report(test_spec: str | None = None) -> None:
    test_spec = test_spec or str(ROOT / "test_units.json")
    sel = json.loads((ROOT / "SELECTION.json").read_text()) if (ROOT / "SELECTION.json").exists() else None
    with (ROOT / "sweep.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "seed", "batch", "lr", "epoch", "val_chrf", "steps", "examples_seen", "train_minutes",
                    "sec_per_step", "gpu_type", "peak_mem_gb", "precision", "in_final_grid", "selected",
                    "val_empty", "val_hit_max_new_tokens", "eval_loss", "max_grad_norm", "checkpoint_gb",
                    "trainable_params", "train_tokens_processed", "train_loss_tokens_processed", "train_tokens_per_s",
                    "train_samples_per_s", "val_wall_s", "val_generated_tokens", "val_generated_tokens_per_s", "run_id"])
        for a in ARCHS:
            for p in sorted((ROOT / "runs" / a).glob("*/RUN_DONE.json"),
                            key=lambda p: (not p.parent.name.startswith("lr"), p.parent.name.split("lr")[0],
                               float(p.parent.name.split("lr")[1]))):
                r = json.loads(p.read_text())
                spe = r["global_step"] / 4
                tok = tokens(a, r["run_info"])
                sps = r["train_samples_per_second"]
                vt = timings(p.parent / "val_timing.txt", "VAL_DONE")
                units = {u["epoch"]: u["unit_id"] for u in json.loads((p.parent / "val_units.json").read_text())}
                disk = r["disk_bytes_per_checkpoint"]
                seed_sel = best(r["epochs"])["epoch"]
                for e in r["epochs"]:
                    ck = [v for k, v in disk.items() if k.endswith(f"-{round(spe * e['epoch'])}")]
                    if p.parent.name.startswith("lr"):
                        chosen = bool(sel) and sel[a]["lr"] == r["lr"] and sel[a]["epoch"] == e["epoch"]
                        in_grid = (r["lr"] in sel["final_grid"]) if sel else ""
                    else:
                        chosen, in_grid = e["epoch"] == seed_sel, True
                    w.writerow([a, r.get("seed", 42), r.get("batch", 16), r["lr"], e["epoch"], round(e["val_chrf"], 3), round(spe * e["epoch"]),
                                round(TRAIN_ROWS * e["epoch"]), round(r["train_runtime_s"] / 60, 2),
                                round(r["sec_per_step"], 4), r["run_info"].get("gpu"),
                                r["run_info"].get("peak_mem_allocated_gb"), r["precision"], in_grid, chosen,
                                e["val"]["empty"], e["val"]["hit_max_new_tokens"],
                                # Mamba-2: the Trainer's eval loss runs transformers' eval-mode Mamba2 (gated RMSNorm over all
                                # channels, not per group as in the trained fused kernel / FLA), so it is not reported
                                "" if a == "mamba2" else round(r["eval_loss_by_epoch"][e["epoch"] - 1], 4)
                                if len(r["eval_loss_by_epoch"]) >= e["epoch"] else "",
                                r["max_grad_norm"], round(ck[0] / 1e9, 3) if ck else "",
                                r["run_info"]["n_trainable_params"],
                                tok.get("train_tokens_per_epoch", 0) * e["epoch"] or "",
                                tok.get("train_loss_tokens_per_epoch", 0) * e["epoch"] or "",
                                round(tok["train_tokens_per_epoch"] * 4 / r["train_runtime_s"], 1) if tok else "",
                                sps, vt.get(units[e["epoch"]], ""), gen_tokens(p.parent, units[e["epoch"]]),
                                round(gen_tokens(p.parent, units[e["epoch"]]) / vt[units[e["epoch"]]], 1)
                                if units[e["epoch"]] in vt and gen_tokens(p.parent, units[e["epoch"]]) != "" else "",
                                f"{a}/{p.parent.name}"])
    units = []
    for spec in [Path(test_spec)] + sorted(ROOT.glob("test_units_*_*.json")):
        if spec.exists():
            units += json.loads(spec.read_text())
    rows = []
    for u in units:
        d = ROOT / "test" / u["unit_id"] / "greedy"
        if not (d / "UNIT_DONE.json").exists():
            continue
        s = score(d)
        wall = timings(ROOT / "test_timing.txt", "TEST_DONE").get(u["unit_id"])
        rows.append([u["architecture"], u.get("seed", 42), u.get("batch", 16), u["lr"], u["epoch"], round(u["val_chrf"], 3),
                     round(s["chrf"], 3), s["rows"], s["empty"], s["hit_max_new_tokens"], s["no_eos"],
                     s["max_new_tokens"], u["unit_id"], wall or "", s["generated_tokens"],
                     round(s["generated_tokens"] / wall, 1) if wall else "", u["model_tree_sha256"]])
    with (ROOT / "test.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "seed", "batch", "lr", "epoch", "val_chrf", "test_chrf", "rows", "empty", "hit_max_new_tokens",
                    "no_eos", "max_new_tokens", "unit_id", "test_wall_s", "generated_tokens",
                    "generated_tokens_per_s", "model_tree_sha256"])
        w.writerows(sorted(rows, key=lambda r: (ARCHS.index(r[0]), r[2], r[1])))
    with (ROOT / "test_seeds_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "lr", "test_chrf_seed42_primary", "test_chrf_seed43", "test_chrf_seed44", "mean", "std_ddof1", "n"])
        for a in ARCHS:
            by_seed = {r[1]: r[6] for r in rows if r[0] == a and r[2] == 16}
            vals = [by_seed[k] for k in sorted(by_seed)]
            w.writerow([a, next((r[3] for r in rows if r[0] == a), ""), by_seed.get(42, ""), by_seed.get(43, ""),
                        by_seed.get(44, ""), round(statistics.mean(vals), 3) if vals else "",
                        round(statistics.stdev(vals), 3) if len(vals) > 1 else "", len(vals)])
    with (ROOT / "batch_sensitivity.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "lr", "batch", "seed", "selected_epoch", "val_chrf", "test_chrf", "unit_id"])
        for r in sorted(rows, key=lambda r: (ARCHS.index(r[0]), r[2])):
            if r[1] == 42:
                w.writerow([r[0], r[3], r[2], 42, r[4], r[5], r[6], r[12]])
    print((ROOT / "test.csv").read_text())
    print((ROOT / "batch_sensitivity.csv").read_text())
    print((ROOT / "test_seeds_summary.csv").read_text())


if __name__ == "__main__":
    globals()[f"cmd_{sys.argv[1]}"](*sys.argv[2:])
