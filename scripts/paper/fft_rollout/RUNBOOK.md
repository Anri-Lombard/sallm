# Full fine-tuning rollout: runbook

Post-pretraining pipeline for one architecture: Base 0-shot eval, full fine-tuning LR sweeps, LR selection
(with the edge rule), Mono and seed runs at the selected LR, test scoring, collection. One command per
architecture. Protocol: `rollout.py` docstring and the notes file
`~/Desktop/Masters/Notes/sallm_architecture_paper_consistency_2026-09-25.md` (final protocol section).

HEX root: `R=/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926`

| path | what |
|---|---|
| `$R/code/fft_rollout/` | this directory (synced by `sync_to_hex.sh`, checked against `$R/code/CODE.sha256` by every lane) |
| `$R/code/sallm/` | frozen copy of this repo's `src/` + `tokenizer/` (NFC chrF and NER span F1), `SNAPSHOT_MANIFEST.sha256` |
| `$R/runs/<name>/` | one architecture: `config.json`, `units.json` (the DAG), `state/`, `STATUS.txt`, `base/`, `runs/`, `keep/`, `test/`, `base_eval/`, `results/`, `logs/` |
| `$R/logs/` | lane job logs `fft-<name>-<lane>-<jobid>.out` |
| `$R/assets/` | T2X train/validation-only files for the fail-closed loader |

## 0. Before the first launch (once, from the Mac)

```bash
cd ~/Desktop/Masters/sallm && git checkout paper/architecture-comparison-2026-09
bash scripts/paper/fft_rollout/sync_to_hex.sh     # rsyncs code/ and writes CODE.sha256
```
Re-run it after any code change; lanes that are already running keep the code they started with
(the Python processes they spawn read the new files, so restart lanes with `relane.sh` after a fix).

## 1. Fire an architecture (HEX head node; bash + sbatch only)

```bash
R=/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926
# when the pretraining run has finished (BASE = the run dir; prep binds weights/step<total_steps>_tok*):
bash $R/code/fft_rollout/launch.sh gdn /scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/runs/full_gdn_wsd_lr<LR> 4
# or fire it now and let Slurm hold the lanes until the pretraining job ends (zero idle time):
AFTER=<pretraining jobid> bash $R/code/fft_rollout/launch.sh gdn <run dir> 4
```
Arguments: `ARCH BASE_PATH [CAP=4] [NAME=ARCH]`. `CAP` = number of lane jobs = concurrent L40S for this
architecture. The per-user QOS limit is 10 GPUs across everything (pretraining included), so keep the sum of
all CAPs + pretraining GPUs <= 10. BASE_PATH may also be a plain weights dir (config.json + pytorch_model.bin).
The launcher refuses a base whose config does not say bos/eos/pad = 0/1/2 (prep checks again, plus the
tokenizer ids, a forward pass, and for Mamba-2 that FLA's class loads).

With `AFTER`: if the pretraining run needs another (resumed) job, the lanes start when the first job ends, prep
finds no final weights, fails twice and blocks everything; re-fire with `NAME` unchanged after
`rm -rf $R/runs/<name>` or use `relane.sh` once the final weights exist. Use the id of the job that will
actually finish the run.

## 2. Monitor

```bash
/scratch/slurm/bin/purequota
cat $R/runs/<name>/STATUS.txt                     # every unit: waiting/running/done/failed/blocked, est. hours, lr/epoch/val
squeue -u $USER -o "%.10i %.22j %.8T %.10M %R" | grep fft-
tail -n 40 $R/logs/fft-<name>-<lane>-<jobid>.out  # UNIT_START / UNIT_DONE / UNIT_FAILED lines
cat $R/runs/<name>/state/<unit>.json              # failure message + traceback tail
cat $R/runs/<name>/keep/<family>/SELECTED.json    # chosen lr/epoch, whether the edge rule fired
tail -n 30 $R/runs/<name>/runs/<run>/train.log
```
A unit that fails is retried once, then marked failed; its dependants become `blocked`. Lanes keep working
on everything else and exit when nothing is runnable.

Change the cap: `echo 2 > $R/runs/<name>/MAX_LANES` (lanes with index >= 2 stop after their current unit);
to add lanes, raise MAX_LANES and `sbatch --export=ALL,OUT=$R/runs/<name>,LANE=<i> $R/code/fft_rollout/lane.sbatch`.
Stop everything gracefully: `touch $R/runs/<name>/STOP`. Restart after a fix (re-runs every unit not done):
`bash $R/code/fft_rollout/relane.sh <name> <N>`.

Lanes run at most 48 h; a lane that cannot fit the next unit in its remaining time resubmits itself.
Idle GPU time is limited to the DAG's barriers (a lane waits while every runnable unit is taken).

## 3. What runs (per architecture; `rollout.py matrix` prints it, `job_matrix.csv` is the committed copy)

| stage | units |
|---|---|
| prep | copy weights + tokenizer, checks, protocols |
| Base 0-shot | `base-gen` (untuned T2X/AfriHG test generation), `base-prompt` (News, Intent, SIB, Belebele, AfriXNLI, AfriMMLU, AfriMGSM official unit), `base-ner`, `base-pos` |
| LR sweep | 3 LRs {3e-5, 1e-4, 3e-4} x 8 families: News/SIB/Intent/NER/POS/AfriHG on the Multi model, T2X on Mono xho, General on the six-family mixture |
| selection | `select-<family>`: best (lr, epoch) on validation; if the best LR is 3e-5 or 3e-4, trains 1e-5 or 1e-3 and reselects |
| Mono / seeds | 20 Mono runs at the family's selected LR (seed 42); seeds 43, 44 for T2X Mono and General |
| test | `test-<family>-...` on the selected sweep checkpoint; Mono and seed runs test their own best epoch in-unit |

Every training run: epochs = 10 if the tokenized train set has < 5000 rows else 4, one checkpoint per epoch in
node-local `/dev/shm`, each scored on validation with the paper's protocol scorer; only the best epoch survives
(sweep runs keep it in `keep/`, deleted when not selected). All scoring is on the lane's L40S.

## 3a. Checkpoints (kept) and the Kombuys archive

Every selected fine-tuned model is kept, weights only (~0.51 GB fp32): each Mono and seed run's best epoch, and
each sweep's selected (lr, epoch). Per architecture: 20 Mono + 6 Multi + T2X (3 seeds) + General (3 seeds) = 32
checkpoints, ~16.3 GB; all four ~65 GB. Sweep runs keep their best epoch only until `select-<family>` has run.
Layout: `runs/<name>/keep/<family>/<run>_e<epoch>/` plus `<run>_e<epoch>.sha256` (manifest, written in-job) and
`<run>_e<epoch>.selected` (marker). HEX /scratch had ~63 GB free on 26 Sep (pretraining also writes there), so
archive as architectures finish, from the Mac:

```bash
bash scripts/paper/fft_rollout/archive_to_kombuys.sh <name>               # copy + sha256sum -c on Kombuys
bash scripts/paper/fft_rollout/archive_to_kombuys.sh <name> --delete-hex  # also drop verified HEX copies
```
Kombuys: `/scratch/alombard/sallm/results/fft_rollout_20260926/<name>/keep/` (560 GB free on 26 Sep). Keep the
Mono checkpoints on HEX until the optional cross_eval stage has run (it scores on L40S), or copy them back.

## 3b. Optional stage: cross_eval (off by default)

Scores every Mono model's selected checkpoint on the test sets of its task's other languages (News, SIB-200,
Intent, NER, POS, AfriHG; 19 units per architecture), same protocol scorers, on L40S. In `job_matrix.csv` with
`optional=True`. Enable for a run (before or after the main pipeline):

```bash
touch $R/runs/<name>/CROSS_EVAL
# if the run's lanes have exited: sbatch --export=ALL,OUT=$R/runs/<name>,LANE=0 $R/code/fft_rollout/lane.sbatch
```
The next lane appends `xeval-<family>-<lang>` units and `collect-xeval`; rows land in `results/cells.csv` with
regime `Mono-crosslingual` and `train_language`, and `collect_fft.py` writes them to `data/fft-crosslingual.csv`.

## 4. Results into the paper

The `collect` unit (last) writes `$R/runs/<name>/results/cells.csv` (one row per cell and run: score, n items,
bootstrap 95% CI where rows allow, lr, seed, epoch). Fetch per architecture (rsync is a plain file copy):

```bash
for a in mzansilm mamba2 xlstm gdn; do
  rsync -a --include='results/***' --include='base_eval/' --include='base_eval/*.json' --exclude='*' \
    hex:$R/runs/$a/ ~/Desktop/sa-architecture-comparison-paper/data/fft_raw/$a/
done
cd ~/Desktop/sa-architecture-comparison-paper
/opt/homebrew/bin/python3 scripts/collect_fft.py data/fft_raw/{mzansilm,mamba2,xlstm,gdn}
/opt/homebrew/bin/python3 scripts/build_results.py
```

| rollout output | paper file | read by |
|---|---|---|
| `base_eval/prompt.json`, `ner.json`, `pos.json` (same schema as the official units) | `data/base-official.csv` (152) | `build_results.py` (Base) |
| cells.csv Mono/Multi rows of News, SIB-200, Intent, NER, POS (seed 42) | `data/monomulti.csv` (144, source `fft`) | `build_results.py`; replaces `collect_monomulti.py`'s rescore/retrain/reselect merge |
| cells.csv T2X/AfriHG rows: Base (`base-gen`), Mono, Multi, General | `data/generation-protocol-v2.csv` (44) | `build_results.py`; replaces `extract_generation_v2.py` (scores are recomputed on HEX with NFC) |
| cells.csv General rows (41 per model) | `data/general-fft.csv` | `build_results.py` (block after the adapter-era General files) |
| all rows + CIs; seeds of General and Mono T2X | `data/fft-cells.csv`, `data/fft-seeds.csv` | text/tables (mean, std over seeds 42/43/44) |

`collect_fft.py` refuses smoke (limited) results and needs all four architectures. Every output carries a language
`family` column (Nguni: zul, xho, ssw, nbl; Sotho-Tswana: sot, tsn, nso; afr, eng, ven, tso on their own).
