#!/usr/bin/env python3
"""Mac: every ~10 min, hand free L40S one at a time to the architecture with the most estimated GPU-hours left per lane.

  nohup caffeinate -i python3 rebalance.py > ~/.sallm_fire/rebalance.log 2>&1 &

Budget: at most MAX_GPUS (10, the per-user QOS limit) L40S across ALL of this user's queued and running jobs, pretraining
included. Nothing is added while any mp-full-* job is pending (pretraining first). An architecture whose pretraining
has finished but whose rollout has not been fired yet keeps 2 GPUs reserved for fire_when_ready.sh.
Readiness is computed here from units.json + state/*.json (the head node only runs cat/squeue/sbatch): a unit is
ready when it has not started and its deps are done (and its ordering-only `after` units are terminal).
New lines of each run's ALERTS.txt are appended to ~/.sallm_fire/alerts.log.
It never cancels anything and never lowers MAX_LANES: lanes free their GPU by themselves (LANE_IDLE_EXIT after
IDLE_EXIT_MIN minutes with nothing runnable, or LANE_FINISHED). Lanes are added with addlane.sh, NICE=0 (pretraining finished; NICE=1000 only mattered while mp-full-* jobs were queued).
A run with work left but no lane at all (e.g. every lane idled out while a stale unit waited) gets one lane first.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

R = "/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926"
ARCHS = ("mzansilm", "mamba2", "xlstm", "gdn")
MAX_GPUS, RESERVE, PERIOD = 10, 2, 600
TERMINAL = ("done", "failed", "blocked")
DRY = "--dry-run" in sys.argv


def hex_(cmd: str, timeout: int = 120) -> str:
    p = subprocess.run(["ssh", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", "hex", cmd], capture_output=True, text=True,
                       timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"ssh rc={p.returncode}: {p.stderr.strip()[-200:]}")
    return p.stdout


def gpus(tres: str) -> int:
    # "gres/gpu:l40s:2", "gres:gpu:l40s:1", "N/A"
    if "gpu" not in tres:
        return 0
    last = tres.rsplit(":", 1)[-1]
    return int(last) if last.isdigit() else 1


ALERT_LOG = Path.home() / ".sallm_fire" / "alerts.log"
ALERT_SEEN = Path.home() / ".sallm_fire" / "alerts.seen"


def forward_alerts(lines: list[str]) -> None:
    """Append rollout ALERTS.txt lines not seen before to ~/.sallm_fire/alerts.log (the one file to watch)."""
    seen = set(ALERT_SEEN.read_text().splitlines()) if ALERT_SEEN.exists() else set()
    new = [x for x in lines if x not in seen]
    if new:
        with ALERT_LOG.open("a") as fh:
            fh.write("".join(x + "\n" for x in new))
        with ALERT_SEEN.open("a") as fh:
            fh.write("".join(x + "\n" for x in new))
        print(f"{time.strftime('%F %T')} {len(new)} new alert(s) -> {ALERT_LOG}", flush=True)


def snapshot() -> tuple[list[tuple[str, str, int]], dict]:
    script = (f'squeue -u $USER -h -o "%j|%T|%b"; echo "@@RUNS"; '
              f'for a in {" ".join(ARCHS)}; do O={R}/runs/$a; [ -e $O/units.json ] || continue; '
              f'echo "@@RUN $a"; cat $O/units.json; echo "@@STATES"; '
              f'for f in $O/state/*.json; do [ -e "$f" ] || continue; echo "@@S $(basename $f .json)"; cat "$f"; done; echo "@@END"; done; '
              f'for a in {" ".join(ARCHS)}; do [ -s {R}/runs/$a/ALERTS.txt ] && sed "s/^/@@ALERT /" {R}/runs/$a/ALERTS.txt; done; true')
    out = hex_(script)
    jobs, runs = [], {}
    alerts = [line[len("@@ALERT "):] for line in out.splitlines() if line.startswith("@@ALERT ")]
    forward_alerts(alerts)
    out = "\n".join(line for line in out.splitlines() if not line.startswith("@@ALERT "))
    head, _, rest = out.partition("@@RUNS")
    for line in head.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3:
            jobs.append((parts[0], parts[1], gpus(parts[2])))
    for block in rest.split("@@RUN ")[1:]:
        name, _, body = block.partition("\n")
        units_txt, _, states_txt = body.partition("@@STATES")
        states = {}
        for s in states_txt.split("@@S ")[1:]:
            uid, _, js = s.partition("\n")
            js = js.replace("@@END", "").strip()
            try:
                states[uid.strip()] = json.loads(js).get("state")
            except json.JSONDecodeError:
                states[uid.strip()] = "running"  # being written: treat as busy
        runs[name.strip()] = (json.loads(units_txt), states)
    return jobs, runs


def summarize(units: list[dict], states: dict) -> dict:
    main = [u for u in units if not u.get("optional")]
    ready = [u for u in main if states.get(u["id"]) in (None, "pending")
             and all(states.get(d) == "done" for d in u["deps"]) and all(states.get(d) in TERMINAL for d in u.get("after", ()))]
    left = sum(u["est_hours"] for u in main if states.get(u["id"]) != "done")
    open_ = sum(1 for u in main if states.get(u["id"]) not in TERMINAL)
    running = sum(1 for u in main if states.get(u["id"]) == "running")
    return {"ready": len(ready), "left": round(left, 1), "open": open_, "running": running}


def step() -> None:
    jobs, runs = snapshot()
    live = [(j, st, g) for j, st, g in jobs if st in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING")]
    used = sum(g for _, _, g in live)
    if any(j.startswith("mp-full-") and st == "PENDING" for j, st, _ in live):
        print(f"{time.strftime('%F %T')} used={used}: an mp-full-* job is pending; adding nothing", flush=True)
        return
    reserve = sum(RESERVE for a in ARCHS if a not in runs and not any(j == f"mp-full-{a}" for j, _, _ in live))
    free = MAX_GPUS - used - reserve
    info = {}
    for a, (units, states) in runs.items():
        s = summarize(units, states)
        s["lanes"] = sum(1 for j, _, _ in live if j.startswith(f"fft-{a}-"))
        s["pending_lanes"] = sum(1 for j, st, _ in live if j.startswith(f"fft-{a}-") and st == "PENDING")
        info[a] = s
    print(f"{time.strftime('%F %T')} used={used} reserve={reserve} free={free} " +
          " ".join(f"{a}:lanes={s['lanes']},ready={s['ready']},run={s['running']},left={s['left']}h" for a, s in info.items()), flush=True)
    plan = []
    for a, s in info.items():  # a run with work left and no lane at all
        if free > 0 and s["lanes"] == 0 and s["open"] > 0:
            plan.append((a, 1))
            free -= 1
            s["lanes"] += 1
            s["pending_lanes"] += 1
    while free > 0:
        need = {a: s["ready"] - s["pending_lanes"] for a, s in info.items() if s["ready"] - s["pending_lanes"] > 0}
        if not need:
            break
        # one GPU at a time to the run that would finish last: most estimated hours left per lane
        a = max(need, key=lambda x: info[x]["left"] / (info[x]["lanes"] + 1))
        plan.append((a, 1))
        free -= 1
        info[a]["pending_lanes"] += 1
        info[a]["lanes"] += 1
    for a, k in plan:
        print(f"{time.strftime('%F %T')} add {k} lane(s) to {a} (left {info[a]['left']} h, ready {info[a]['ready']})", flush=True)
        if not DRY:
            print(hex_(f"NICE=0 bash {R}/code/fft_rollout/addlane.sh {a} {k}").strip(), flush=True)


def main() -> None:
    while True:
        try:
            step()
        except Exception as exc:  # noqa: BLE001  (ssh hiccup, partial file): retry next period
            print(f"{time.strftime('%F %T')} error: {exc}", flush=True)
        if DRY:
            return
        time.sleep(PERIOD)


if __name__ == "__main__":
    main()
