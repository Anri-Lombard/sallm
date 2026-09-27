#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import fmean

ARCHITECTURES = ("mzansilm", "mamba2", "xlstm", "gdn")
LANGUAGES = ("afr", "eng", "nso", "sot", "xho", "zul")
TASK = re.compile(r"^sallm_sib_(afr|eng|nso|sot|xho|zul)_val_prompt_([1-5])$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    values = {language: {prompt: {} for prompt in range(1, 6)} for language in LANGUAGES}
    sources = {}
    for architecture in ARCHITECTURES:
        path = args.validation_root / architecture / "sib_all_val" / "results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources[architecture] = {"path": str(path), "sha256": sha256(path)}
        found = 0
        for task, metrics in payload["results"].items():
            match = TASK.match(task)
            if not match:
                continue
            language, prompt = match.group(1), int(match.group(2))
            values[language][prompt][architecture] = float(metrics["f1,none"])
            found += 1
        if found != 30:
            raise ValueError(f"{architecture}: expected 30 SIB validation tasks, got {found}")

    selected = {}
    for language in LANGUAGES:
        means = {}
        for prompt in range(1, 6):
            scores = values[language][prompt]
            if tuple(scores) != ARCHITECTURES:
                raise ValueError(f"{language} prompt {prompt}: incomplete architecture coverage")
            means[prompt] = fmean(scores.values())
        winner = max(range(1, 6), key=lambda prompt: (means[prompt], -prompt))
        selected[language] = {
            "prompt": winner,
            "validation_mean_f1": means[winner],
            "architecture_f1": values[language][winner],
            "all_prompt_means": means,
            "test_task": f"sib_{language}_prompt_{winner}",
        }

    payload = {
        "schema": "sallm.retained_sib_prompt_selection/v1",
        "selection_split": "validation",
        "architectures": list(ARCHITECTURES),
        "aggregation": "unweighted mean F1 across the four architectures",
        "tie_break": "lowest listed prompt number",
        "apply_chat_template": False,
        "runtime_projection": {
            "path": str(args.projection),
            "sha256": sha256(args.projection),
            "value": json.loads(args.projection.read_text(encoding="utf-8")),
        },
        "bindings": {"path": str(args.bindings), "sha256": sha256(args.bindings)},
        "snapshot_manifest": {
            "path": str(args.snapshot_manifest),
            "sha256": sha256(args.snapshot_manifest),
        },
        "launcher": {"path": str(args.launcher), "sha256": sha256(args.launcher)},
        "sources": sources,
        "selected": selected,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(sha256(args.output))


if __name__ == "__main__":
    main()
