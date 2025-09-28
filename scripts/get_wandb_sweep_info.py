import math
import sys
from argparse import ArgumentParser

import pandas as pd
import wandb

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--project", default="anri-lombard/sallm-ft")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def is_empty(v):
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, list | dict) and len(v) == 0:
        return True
    return False


def to_list(v):
    return v if isinstance(v, list) else [v]


args = parse_args()
project_path = args.project
target_name = args.name
out_csv = args.out or f"{target_name}.csv"

api = wandb.Api()
runs = list(api.runs(project_path, filters={"display_name": target_name}))
iterator = tqdm(runs, total=len(runs), desc="Scanning runs") if tqdm else runs
selected = []
for r in iterator:
    if r.name == target_name:
        selected.append(r)
if not selected:
    sys.exit(f"No run found with name {target_name!r} in {project_path}.")
run = sorted(selected, key=lambda r: r.created_at, reverse=True)[0]
config = {k: v for k, v in run.config.items() if not str(k).startswith("_")}
summary = dict(run.summary)
cfg_df = pd.json_normalize(config, sep=".")
cfg_df.columns = [f"config.{c}" for c in cfg_df.columns]
sum_df = pd.json_normalize(summary, sep=".")
sum_df.columns = [f"summary.{c}" for c in sum_df.columns]
row_df = pd.concat([cfg_df, sum_df], axis=1)
row = row_df.iloc[0].to_dict()
non_empty_cols = {k: v for k, v in row.items() if not is_empty(v)}
pd.DataFrame([non_empty_cols]).to_csv(out_csv, index=False)
print(f"Saved non-empty fields to {out_csv}")
for col, val in sorted(non_empty_cols.items()):
    print(f"{col}: {to_list(val)}")
