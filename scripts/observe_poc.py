#!/usr/bin/env python3
"""Tiny SALLM observability proof of concept.

Writes:
- sallm_memory/observability/latest.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "sallm_memory" / "observability"
SNAPSHOT_PATH = OUT_DIR / "latest.json"
ARTIFACTS_DIR = ROOT / "sallm_memory" / "artifacts"
NOTES_DIR = ROOT / "sallm_memory" / "notes"

REMOTE_ARTIFACTS = [
    "/scratch/lmbanr001/masters/sallm/results/diagnostics/task_head_matrix_test_20260626/llama_pos_zul.json",
    "/scratch/lmbanr001/masters/sallm/checkpoints/sallm-gated-deltanet-125m",
]

ERROR_RE = re.compile(
    r"Traceback|Error|Exception|AssertionError|ValueError|CUDA out of memory|OOM|"
    r"FAILED|JobArrayTaskLimit|MissingConfigException|No such file",
    re.IGNORECASE,
)
DICT_RE = re.compile(
    r"\{[^{}\n]*(?:loss|eval_loss|train_runtime|mean_token_accuracy)[^{}\n]*\}"
)
PROGRESS_RE = re.compile(
    r"(?P<pct>\d+)%\|.*?\|\s*(?P<step>\d+)/(?P<total>\d+).*?\[(?P<elapsed>[^,<\]]+)<(?P<eta>[^,\]]+)"
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def tail_text(path: Path, max_bytes: int = 40000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def remote_probe_script() -> str:
    artifact_checks = "\n".join(
        (
            f"[ -e {sh_quote(path)} ] && printf 'present|%s\\n' "
            f"{sh_quote(path)} || printf 'missing|%s\\n' {sh_quote(path)}"
        )
        for path in REMOTE_ARTIFACTS
    )
    sacct_fields = (
        "JobIDRaw,JobName,State,Partition,AllocTRES,ElapsedRaw,Start,End,ExitCode"
    )
    return f"""/scratch/slurm/bin/purequota
printf '\\n__SALLM_SECTION__SQUEUE\\n'
squeue -u "$USER" -h -o '%i|%j|%T|%M|%l|%D|%R|%P|%b' 2>&1 || true
printf '\\n__SALLM_SECTION__SACCT\\n'
SINCE=$(date -d '7 days ago' +%F 2>/dev/null || date +%F)
sacct -u "$USER" -S "$SINCE" -P --noheader -o {sacct_fields} 2>&1 || true
printf '\\n__SALLM_SECTION__ARTIFACTS\\n'
{artifact_checks}
printf '\\n__SALLM_SECTION__TAILS\\n'
ls -t "$HOME"/masters/sallm/slurm-*.out 2>/dev/null | head -n 8 \
  | while IFS= read -r f; do
  [ -e "$f" ] || continue
  printf '\\n__SALLM_LOG__%s\\n' "$f"
  tail -n 80 "$f" 2>/dev/null || true
done
"""


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def split_sections(output: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"QUOTA": []}
    current = "QUOTA"
    for line in output.splitlines():
        if line.startswith("__SALLM_SECTION__"):
            current = line.replace("__SALLM_SECTION__", "", 1).strip()
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {key: "\n".join(lines).strip() for key, lines in sections.items()}


def run_ssh_probe(host: str, timeout: int, connect_timeout: int) -> dict[str, Any]:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        host,
        remote_probe_script(),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"ssh probe timed out after {timeout}s",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "sections": {},
        }

    output = "\n".join(part for part in [proc.stdout, proc.stderr] if part)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "error": ""
        if proc.returncode == 0
        else (proc.stderr or proc.stdout).strip()[-500:],
        "sections": split_sections(output),
    }


def parse_squeue(text: str) -> list[dict[str, str]]:
    jobs = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 9:
            continue
        jobs.append(
            {
                "job_id": parts[0],
                "name": parts[1],
                "state": parts[2],
                "elapsed": parts[3],
                "limit": parts[4],
                "nodes": parts[5],
                "reason": parts[6],
                "partition": parts[7],
                "gres": parts[8],
            }
        )
    return jobs


def parse_sacct(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 9 or not parts[0] or "." in parts[0]:
            continue
        elapsed_raw = int(parts[5]) if parts[5].isdigit() else 0
        gpus = gpu_count(parts[4])
        rows.append(
            {
                "job_id": parts[0],
                "name": parts[1],
                "state": parts[2],
                "partition": parts[3],
                "alloc_tres": parts[4],
                "elapsed_seconds": elapsed_raw,
                "gpu_count": gpus,
                "gpu_hours": round(elapsed_raw * gpus / 3600, 2),
                "start": parts[6],
                "end": parts[7],
                "exit": parts[8],
            }
        )
    return rows


def gpu_count(text: str) -> int:
    for pattern in (
        r"gres/gpu(?::[\w-]+)?=(\d+)",
        r"gpu:[\w-]+:(\d+)",
        r"gpu:(\d+)",
    ):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return 0


def parse_artifacts(text: str) -> list[dict[str, str]]:
    rows = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        status, path = line.split("|", 1)
        rows.append({"status": status, "path": path})
    return rows


def parse_quota(text: str) -> dict[str, Any]:
    quota: dict[str, Any] = {"raw": text}
    scratch = re.search(r"(?i)(/scratch).*?(\d+(?:\.\d+)?)%", text, re.S)
    home = re.search(r"(?i)(/home).*?(\d+(?:\.\d+)?)%", text, re.S)
    if scratch:
        quota["scratch_percent"] = float(scratch.group(2))
    if home:
        quota["home_percent"] = float(home.group(2))
    return quota


def extract_logs(remote_tails: str) -> list[dict[str, str]]:
    logs: list[dict[str, str]] = []
    chunks = remote_tails.split("__SALLM_LOG__")
    for chunk in chunks[1:]:
        path, _, body = chunk.partition("\n")
        logs.append({"path": path.strip(), "tail": body.strip()[-12000:]})

    local_logs = sorted(
        ARTIFACTS_DIR.rglob("*.out"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )[:12]
    for path in local_logs:
        logs.append(
            {"path": str(path.relative_to(ROOT)), "tail": tail_text(path)[-12000:]}
        )
    return logs


def extract_training_events(logs: list[dict[str, str]]) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    progress: list[dict[str, Any]] = []

    for log in logs:
        source = log["path"]
        for match in DICT_RE.finditer(log["tail"]):
            try:
                record = ast.literal_eval(match.group(0))
            except (SyntaxError, ValueError):
                continue
            if isinstance(record, dict):
                record["source"] = source
                metrics.append(record)

        for line in log["tail"].splitlines():
            if ERROR_RE.search(line):
                errors.append({"source": source, "line": strip_ansi(line)[-500:]})
            match = PROGRESS_RE.search(strip_ansi(line))
            if match:
                progress.append(
                    {
                        "source": source,
                        "pct": int(match.group("pct")),
                        "step": int(match.group("step")),
                        "total": int(match.group("total")),
                        "elapsed": match.group("elapsed"),
                        "eta": match.group("eta"),
                    }
                )

    return {
        "metrics": metrics[-80:],
        "errors": errors[-50:],
        "progress": progress[-20:],
    }


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def recent_notes(limit: int = 8) -> list[dict[str, str]]:
    notes = sorted(
        NOTES_DIR.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True
    )[:3]
    entries: list[dict[str, str]] = []
    for note in notes:
        text = note.read_text(errors="replace")
        chunks = re.split(r"(?m)^###\s+", text)
        for chunk in chunks[1:]:
            title, _, body = chunk.partition("\n")
            entries.append(
                {
                    "file": str(note.relative_to(ROOT)),
                    "title": title.strip(),
                    "body": "\n".join(body.strip().splitlines()[:7]),
                }
            )
    return entries[-limit:]


def local_result_summaries(limit: int = 16) -> list[dict[str, Any]]:
    paths = []
    for base in [ROOT / "results", ARTIFACTS_DIR]:
        if base.exists():
            paths.extend(base.rglob("*.json"))
    paths = sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)[:limit]

    summaries = []
    for path in paths:
        try:
            data = json.loads(path.read_text(errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        numbers: list[tuple[str, float]] = []
        collect_numbers(data, "", numbers, 10)
        summaries.append(
            {
                "path": str(path.relative_to(ROOT)),
                "top_keys": list(data)[:8] if isinstance(data, dict) else [],
                "numbers": numbers[:8],
            }
        )
    return summaries


def collect_numbers(
    value: Any, prefix: str, out: list[tuple[str, float]], limit: int
) -> None:
    if len(out) >= limit:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            collect_numbers(item, f"{prefix}.{key}" if prefix else str(key), out, limit)
    elif isinstance(value, list):
        for idx, item in enumerate(value[:5]):
            collect_numbers(item, f"{prefix}[{idx}]", out, limit)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out.append((prefix, round(float(value), 6)))


def build_stats(sacct_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gpu_hours = sum(row["gpu_hours"] for row in sacct_rows)
    by_day: dict[str, float] = defaultdict(float)
    for row in sacct_rows:
        start = row.get("start", "")
        if start and start not in {"Unknown", "None", "N/A"}:
            by_day[start[:10]] += row["gpu_hours"]
    streak = 0
    if by_day:
        day = datetime.fromisoformat(max(by_day)).date()
        while by_day.get(day.isoformat(), 0) > 0:
            streak += 1
            day -= timedelta(days=1)
    return {
        "gpu_hours_7d": round(gpu_hours, 2),
        "training_streak_days": streak,
        "days": dict(sorted(by_day.items())),
    }


def collect_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    ssh = {"ok": False, "skipped": True, "sections": {}}
    if not args.offline:
        ssh = run_ssh_probe(args.host, args.timeout, args.connect_timeout)

    sections = ssh.get("sections", {})
    jobs = parse_squeue(sections.get("SQUEUE", ""))
    sacct_rows = parse_sacct(sections.get("SACCT", ""))
    remote_artifacts = parse_artifacts(sections.get("ARTIFACTS", ""))
    logs = extract_logs(sections.get("TAILS", ""))
    training = extract_training_events(logs)

    return {
        "generated_at": utc_now(),
        "host": args.host,
        "ssh": {key: value for key, value in ssh.items() if key != "sections"},
        "quota": parse_quota(sections.get("QUOTA", "")),
        "jobs": jobs,
        "history": sacct_rows,
        "stats": build_stats(sacct_rows),
        "remote_artifacts": remote_artifacts,
        "training": training,
        "results": local_result_summaries(),
        "notes": recent_notes(),
    }


def write_outputs(snapshot: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="hex")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--connect-timeout", type=int, default=10)
    args = parser.parse_args()

    snapshot = collect_snapshot(args)
    write_outputs(snapshot)

    print(f"snapshot: {SNAPSHOT_PATH}")
    print("app: http://127.0.0.1:5177/")
    print(f"ssh: {'ok' if snapshot['ssh'].get('ok') else 'not ok/offline'}")
    print(f"live jobs: {len(snapshot['jobs'])}")
    print(f"metrics parsed: {len(snapshot['training']['metrics'])}")
    print(f"errors parsed: {len(snapshot['training']['errors'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
