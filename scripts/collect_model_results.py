#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path
from typing import Any
from urllib import parse, request

DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1Ph_zVcSuLZy0dUBnDPF4tkLqAfEVCwK9JVybB2_8x6U/edit?usp=sharing"
)

SHEET_TABS: dict[str, dict[str, Any]] = {
    "Transformer Results": {
        "family": "transformer",
        "model_name": "transformer_baseline",
        "variant_columns": {
            "Base Model": "base",
            "Monolingual Tuned Model": "monolingual",
            "Multilingual Tuned Model": "multilingual",
            "General Instruction Tuned Model": "general_instruction",
        },
    },
    "Mamba Results": {
        "family": "mamba",
        "model_name": "mamba",
        "variant_columns": {
            "Base Model": "base",
            "Monolingual Tuned Model": "monolingual",
            "Multilingual Tuned Model": "multilingual",
            "General Instruction Tuned Model": "general_instruction",
        },
    },
    "XLSTM Results": {
        "family": "xlstm",
        "model_name": "xlstm",
        "variant_columns": {
            "Base Model": "base",
            "Fine Tuned Model": "fine_tuned",
            "General Instruction Tuned Model": "general_instruction",
        },
    },
}

DISPLAY_TASKS = {
    "afrihg": "AfriHG",
    "injongointent": "Injongo Intent",
    "ner": "MasakhaNER",
    "news": "MasakhaNews",
    "pos": "MasakhaPOS",
    "sib": "SIB200",
    "t2x": "Text2Text",
    "sa_general": "SA General",
}

DISPLAY_LANGS = {
    "afr": "Afr",
    "eng": "Eng",
    "nso": "Nso",
    "sot": "Sot",
    "tsn": "Tsn",
    "tso": "Tso",
    "xho": "Xho",
    "zul": "Zul",
    "all": "All",
}

LABELED_SCORE_RE = re.compile(
    r"^(?P<label>[^:]+):\s*(?P<score>\d+(?:\.\d+)?)\s*(?P<metric>[A-Za-z0-9_.\" -]+)?$"
)
PLAIN_SCORE_RE = re.compile(
    r"^(?P<score>\d+(?:\.\d+)?)\s*(?P<metric>[A-Za-z0-9_.\" -]+)?$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect tracked experiment baselines and live model run status."
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("outputs/final_submissions"),
        help="Directory containing *_final_submit_*.csv manifests",
    )
    parser.add_argument(
        "--sheet-url",
        default=DEFAULT_SHEET_URL,
        help="Google Sheet URL with Transformer/Mamba/XLSTM result tabs",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/model_results.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("reports/model_results.md"),
        help="Markdown output path",
    )
    parser.add_argument(
        "--run-tag",
        action="append",
        default=[],
        help="Limit manifest import to one or more run tags",
    )
    parser.add_argument(
        "--skip-sheet",
        action="store_true",
        help="Skip importing the Google Sheet baselines",
    )
    parser.add_argument(
        "--skip-manifests",
        action="store_true",
        help="Skip importing local final-run manifests",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def extract_run_tag(path: Path, prefix: str) -> str:
    stem = path.stem
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected manifest name: {path.name}")
    return stem[len(prefix) :]


def discover_manifests(
    manifest_dir: Path, run_tags: set[str]
) -> dict[tuple[str, str], dict[str, Path]]:
    runs: dict[tuple[str, str], dict[str, Path]] = {}

    for path in sorted(manifest_dir.glob("*_final_submit_*.csv")):
        match = re.match(
            r"(?P<model>[a-z0-9]+)_final_submit_(?P<run_tag>.+)\.csv", path.name
        )
        if not match:
            continue
        model_name = match.group("model")
        run_tag = match.group("run_tag")
        if run_tags and run_tag not in run_tags:
            continue
        runs.setdefault((model_name, run_tag), {})["ft"] = path

    for path in sorted(manifest_dir.glob("*_final_eval_submit_*.csv")):
        match = re.match(
            r"(?P<model>[a-z0-9]+)_final_eval_submit_(?P<run_tag>.+)\.csv", path.name
        )
        if not match:
            continue
        model_name = match.group("model")
        run_tag = match.group("run_tag")
        if run_tags and run_tag not in run_tags:
            continue
        runs.setdefault((model_name, run_tag), {})["eval"] = path

    return runs


def find_checkpoint(ft_row: dict[str, str]) -> Path | None:
    output_dir = ft_row.get("output_dir", "")
    if not output_dir:
        return None
    base = Path(output_dir)
    adapter = base / "final_adapter"
    merged = base / "final_merged_model"
    if adapter.exists():
        return adapter
    if merged.exists():
        return merged
    return None


def load_summary(eval_output_dir: str) -> tuple[Path | None, list[dict[str, Any]]]:
    if not eval_output_dir:
        return None, []
    summary_path = Path(eval_output_dir) / "evaluation_summary.json"
    if not summary_path.exists():
        return None, []
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return summary_path, []
    if not isinstance(payload, list):
        return summary_path, []
    return summary_path, [item for item in payload if isinstance(item, dict)]


def summarize_payload(payload: list[dict[str, Any]]) -> tuple[str, str]:
    task_names: list[str] = []
    metric_chunks: list[str] = []

    for item in payload:
        task_name = str(
            item.get("task_pack") or item.get("task") or item.get("id") or ""
        )
        if task_name:
            task_names.append(task_name)

        if isinstance(item.get("metrics"), dict) and item["metrics"]:
            metric_chunks.append(json.dumps(item["metrics"], sort_keys=True))
        elif isinstance(item.get("results"), dict) and item["results"]:
            metric_chunks.append(json.dumps(item["results"], sort_keys=True))

    return ";".join(task_names), " | ".join(metric_chunks)


def family_for_model(model_name: str) -> str:
    return "transformer" if model_name == "llama" else model_name


def parse_config_stem(model_name: str, config_name: str) -> tuple[str, str]:
    stem = config_name.split("/")[-1]
    prefix = f"{model_name}_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]

    if stem.startswith("sa_general_"):
        return DISPLAY_TASKS["sa_general"], DISPLAY_LANGS.get(
            stem.split("_")[-1], stem.split("_")[-1].title()
        )

    parts = stem.split("_")
    if len(parts) >= 2:
        lang = parts[-1]
        task = "_".join(parts[:-1])
        return DISPLAY_TASKS.get(task, task.title()), DISPLAY_LANGS.get(
            lang, lang.title()
        )

    return stem.title(), ""


def build_manifest_rows(
    runs: dict[tuple[str, str], dict[str, Path]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for (model_name, run_tag), paths in sorted(runs.items()):
        ft_rows = read_csv_rows(paths["ft"]) if "ft" in paths else []
        eval_rows = read_csv_rows(paths["eval"]) if "eval" in paths else []
        eval_by_config = {row["config"]: row for row in eval_rows if row.get("config")}

        for ft_row in ft_rows:
            config = ft_row.get("config", "")
            eval_row = eval_by_config.get(config, {})
            checkpoint = find_checkpoint(ft_row)
            summary_path, payload = load_summary(eval_row.get("output_dir", ""))
            tasks_blob, metrics_blob = summarize_payload(payload)
            task_display, lang_display = parse_config_stem(model_name, config)

            if payload:
                status = "results_ready"
            elif eval_row.get("eval_job_id"):
                status = "eval_submitted"
            elif checkpoint is not None:
                status = "checkpoint_ready"
            else:
                status = "training_incomplete"

            rows.append(
                {
                    "source_kind": "cluster_manifest",
                    "source_name": model_name,
                    "source_url": "",
                    "source_tab": "",
                    "family": family_for_model(model_name),
                    "model_name": model_name,
                    "run_tag": run_tag,
                    "task": task_display,
                    "language": lang_display,
                    "variant": ft_row.get("category", ""),
                    "setting": "",
                    "score": "",
                    "metric": "",
                    "raw_value": metrics_blob,
                    "status": status,
                    "date_updated": "",
                    "config": config,
                    "finetune_job_id": ft_row.get("job_id", ""),
                    "eval_job_id": eval_row.get("eval_job_id", ""),
                    "eval_output_dir": eval_row.get("output_dir", ""),
                    "summary_path": str(summary_path) if summary_path else "",
                    "checkpoint_path": str(checkpoint) if checkpoint else "",
                    "comparisons": "",
                    "notes": tasks_blob,
                }
            )

    return rows


def sheet_export_url(sheet_url: str, tab_name: str) -> str:
    parsed = parse.urlparse(sheet_url)
    parts = [part for part in parsed.path.split("/") if part]
    if "d" not in parts:
        raise ValueError(f"Could not parse Google Sheet URL: {sheet_url}")
    sheet_id = parts[parts.index("d") + 1]
    return (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq"
        f"?tqx=out:csv&sheet={parse.quote(tab_name)}"
    )


def fetch_sheet_rows(sheet_url: str, tab_name: str) -> list[dict[str, str]]:
    url = sheet_export_url(sheet_url, tab_name)
    with request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def expand_metric_cell(cell_value: str) -> list[dict[str, str]]:
    cleaned = cell_value.strip()
    if not cleaned:
        return []

    parsed_rows: list[dict[str, str]] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.lstrip("-").strip()
        match = LABELED_SCORE_RE.match(line)
        setting = ""
        metric = ""
        score = ""

        if match:
            label = (match.group("label") or "").strip().strip('"')
            score = match.group("score") or ""
            metric = (match.group("metric") or "").strip().strip('"')

            if "shot" in label.lower():
                setting = label
            elif metric:
                metric = f"{label} {metric}".strip()
            else:
                metric = label
        else:
            plain_match = PLAIN_SCORE_RE.match(line)
            if not plain_match:
                continue
            score = plain_match.group("score") or ""
            metric = (plain_match.group("metric") or "").strip().strip('"')

        parsed_rows.append(
            {
                "setting": setting,
                "score": score,
                "metric": metric,
                "raw_value": line,
                "status": "baseline_result",
            }
        )

    if parsed_rows:
        return parsed_rows

    return [
        {
            "setting": "",
            "score": "",
            "metric": "",
            "raw_value": cleaned,
            "status": "baseline_note",
        }
    ]


def build_sheet_rows(sheet_url: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for tab_name, meta in SHEET_TABS.items():
        sheet_rows = fetch_sheet_rows(sheet_url, tab_name)
        current_task = ""

        for row in sheet_rows:
            task = (row.get("Task") or "").strip()
            language = (row.get("Language") or "").strip()
            date_updated = (row.get("Date Updated") or "").strip()
            comparisons = (row.get("Closest Comparisons") or "").strip()
            notes = "\n".join(
                value.strip()
                for key, value in row.items()
                if key in {"Notes", "Extra notes"} and value and value.strip()
            ).strip()

            if task:
                current_task = task
            elif not current_task:
                continue

            for column_name, variant in meta["variant_columns"].items():
                cell_value = (row.get(column_name) or "").strip()
                if not cell_value:
                    continue

                for parsed in expand_metric_cell(cell_value):
                    rows.append(
                        {
                            "source_kind": "google_sheet",
                            "source_name": "masters_datasets_sheet",
                            "source_url": sheet_url,
                            "source_tab": tab_name,
                            "family": str(meta["family"]),
                            "model_name": str(meta["model_name"]),
                            "run_tag": "",
                            "task": current_task,
                            "language": language,
                            "variant": variant,
                            "setting": parsed["setting"],
                            "score": parsed["score"],
                            "metric": parsed["metric"],
                            "raw_value": parsed["raw_value"],
                            "status": parsed["status"],
                            "date_updated": date_updated,
                            "config": "",
                            "finetune_job_id": "",
                            "eval_job_id": "",
                            "eval_output_dir": "",
                            "summary_path": "",
                            "checkpoint_path": "",
                            "comparisons": comparisons,
                            "notes": notes,
                        }
                    )

    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_kind",
        "source_name",
        "source_url",
        "source_tab",
        "family",
        "model_name",
        "run_tag",
        "task",
        "language",
        "variant",
        "setting",
        "score",
        "metric",
        "raw_value",
        "status",
        "date_updated",
        "config",
        "finetune_job_id",
        "eval_job_id",
        "eval_output_dir",
        "summary_path",
        "checkpoint_path",
        "comparisons",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline_rows = [row for row in rows if row["source_kind"] == "google_sheet"]
    live_rows = [row for row in rows if row["source_kind"] == "cluster_manifest"]
    scored_rows = [row for row in baseline_rows if row["score"]]
    completed_live = sum(row["status"] == "results_ready" for row in live_rows)
    in_flight_live = sum(row["status"] != "results_ready" for row in live_rows)

    lines = [
        "# Model Results",
        "",
        "## Summary",
        "",
        f"- Baseline rows imported from Google Sheets: {len(baseline_rows)}",
        f"- Scored baseline rows: {len(scored_rows)}",
        f"- Live experiment rows from manifests: {len(live_rows)}",
        f"- Live rows with completed results: {completed_live}",
        f"- Live rows still in flight: {in_flight_live}",
        "",
        "## Baseline Scores",
        "",
        "| Family | Task | Lang | Variant | Setting | Score | Metric | Updated |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in scored_rows:
        lines.append(
            f"| {row['family']} | {row['task']} | {row['language']} | "
            f"{row['variant']} | {row['setting']} | {row['score']} | "
            f"{row['metric']} | {row['date_updated']} |"
        )

    lines.extend(
        [
            "",
            "## Live Runs",
            "",
            "| Family | Model | Run Tag | Task | Lang | Variant | Status | Eval Job |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in live_rows:
        lines.append(
            f"| {row['family']} | {row['model_name']} | {row['run_tag']} | "
            f"{row['task']} | {row['language']} | {row['variant']} | "
            f"{row['status']} | {row['eval_job_id']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []

    if not args.skip_sheet:
        rows.extend(build_sheet_rows(args.sheet_url))

    if not args.skip_manifests:
        runs = discover_manifests(args.manifest_dir, set(args.run_tag))
        rows.extend(build_manifest_rows(runs))

    rows.sort(
        key=lambda row: (
            row["family"],
            row["model_name"],
            row["task"],
            row["language"],
            row["variant"],
            row["run_tag"],
            row["setting"],
        )
    )

    write_csv(args.out_csv, rows)
    write_markdown(args.out_md, rows)


if __name__ == "__main__":
    main()
