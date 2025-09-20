import math
import sys

import pandas as pd
import wandb

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

PROJECT_PATH = "anri-lombard/sallm-ft"
TARGET_NAME = "absurd-sweep-7"
OUT_CSV = f"{TARGET_NAME}.csv"


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


api = wandb.Api()
runs = list(api.runs(PROJECT_PATH, filters={"display_name": TARGET_NAME}))
iterator = tqdm(runs, total=len(runs), desc="Scanning runs") if tqdm else runs
selected = []
for r in iterator:
    if r.name == TARGET_NAME:
        selected.append(r)
if not selected:
    sys.exit(f"No run found with name {TARGET_NAME!r} in {PROJECT_PATH}.")
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
pd.DataFrame([non_empty_cols]).to_csv(OUT_CSV, index=False)
print(f"Saved non-empty fields to {OUT_CSV}")
for col, val in sorted(non_empty_cols.items()):
    print(f"{col}: {to_list(val)}")
