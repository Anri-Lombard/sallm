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
Or from the Mac, fire each architecture the moment its final weights exist (polls every 5 min; `AFTER` is fragile
because `mp-full-*` jobs resubmit themselves under new ids):

```bash
# detached from the terminal (setsid) and kept awake (caffeinate); a copy of the script so later edits cannot touch it
mkdir -p ~/.sallm_fire && cp ~/Desktop/Masters/sallm/scripts/paper/fft_rollout/fire_when_ready.sh ~/.sallm_fire/
for ac in mzansilm:2 xlstm:3 gdn:3 mamba2:2; do a=${ac%:*}; c=${ac#*:}
  perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' -- nohup caffeinate -i bash ~/.sallm_fire/fire_when_ready.sh $a $c \
    < /dev/null > ~/.sallm_fire/$a.log 2>&1 &
done
pgrep -fl fire_when_ready; tail ~/.sallm_fire/*.log
```
The loops wait while any `mp-full-*` job is pending. Caps 2/3/3/2 keep pretraining + lanes <= 10 L40S at every point
(the first three bases free 6 GPUs while Mamba-2 still pretrains on 2). When an architecture's lanes go idle at the end,
hand its GPUs to a slower one: `echo <n> > $R/runs/<name>/MAX_LANES` plus extra lanes (section 2). The Mac must stay
awake and online (caffeinate stops idle sleep, not a closed lid).

GPU budget: the per-user QOS limit is 10 L40S and the matched-pretraining runs (`mp-full-*`, 2 GPUs each, auto-resuming)
have priority. Pick `CAP` so that pretraining GPUs + all rollout lanes <= 10 (e.g. the first base finishes -> its 2 GPUs
-> `CAP=2`), and pass `NICE=1000` so a pending pretraining resubmit outranks pending lanes of the same user:

```bash
NICE=1000 bash $R/code/fft_rollout/launch.sh <arch> <run dir or weights dir> 2
```
Raise the cap later with `echo <n> > $R/runs/<name>/MAX_LANES` plus extra `sbatch ... lane.sbatch` (section 2).

Arguments: `ARCH BASE_PATH [CAP=4] [NAME=ARCH]`. `CAP` = number of lane jobs = concurrent L40S for this
architecture. The per-user QOS limit is 10 GPUs across everything (pretraining included), so keep the sum of
all CAPs + pretraining GPUs <= 10. BASE_PATH may also be a plain weights dir (config.json + pytorch_model.bin).
The launcher refuses a base whose config does not say bos/eos/pad = 0/1/2 (prep checks again, plus the
tokenizer ids, a forward pass, and for Mamba-2 that FLA's class loads).

With `AFTER`: if the pretraining run needs another (resumed) job, the lanes start when the first job ends, prep
finds no final weights, fails twice and blocks everything; re-fire with `NAME` unchanged after
`rm -rf $R/runs/<name>` or use `relane.sh` once the final weights exist. Use the id of the job that will
actually finish the run.

### Automatic lane rebalancing (Mac)

```bash
cp ~/Desktop/Masters/sallm/scripts/paper/fft_rollout/rebalance.py ~/.sallm_fire/
perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' -- nohup caffeinate -i /opt/homebrew/bin/python3 -u ~/.sallm_fire/rebalance.py \
  < /dev/null >> ~/.sallm_fire/rebalance.log 2>&1 &
```
Every 10 min: counts this user's queued + running L40S (pretraining included), keeps the total <= 10, adds nothing
while an `mp-full-*` job is pending, keeps 2 GPUs for an architecture whose pretraining ended but is not fired yet,
and gives free GPUs to the architecture with the most estimated GPU-h left that has ready units (`addlane.sh`,
NICE=1000). It never cancels anything. GPUs are freed by lanes themselves: a lane with nothing runnable for
`IDLE_EXIT_MIN` (15) minutes exits (`LANE_IDLE_EXIT`), and finished runs' lanes exit. Lanes started before 22:35 on
26 Sep run the older code without idle exit until they resubmit. `python3 rebalance.py --dry-run` prints one decision.
Stop it by its exact PID.

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

Kombuys RTX 5090: not used by the lanes. Every training unit scores each epoch on validation in the same job (node-local
checkpoints), and all scoring must be on L40S, so a 5090 lane would need train/score split units plus ~5 GB of epoch
checkpoints per run shipped Kombuys -> Mac -> HEX login node. Not worth it at ~0.5 GB/epoch; kept out on purpose.

## 3. What runs (per architecture; `rollout.py matrix` prints it, `job_matrix.csv` is the committed copy)

| stage | units |
|---|---|
| prep | copy weights + tokenizer, checks, protocols |
| Base 0-shot | `base-gen` (untuned T2X/AfriHG test generation), `base-prompt` (News, Intent, SIB, Belebele, AfriXNLI, AfriMMLU, AfriMGSM official unit), `base-ner`, `base-pos` |
| LR sweep | 3 LRs {3e-5, 1e-4, 3e-4} x 7 families: News/SIB/Intent/NER/POS/AfriHG on the Multi model, T2X on Mono xho (General: no sweep, LR transferred, see below) |
| selection | `select-<family>`: best (lr, epoch) on validation; if the best LR is 3e-5 or 3e-4, trains 1e-5 or 1e-3 and reselects |
| Mono / seeds | 20 Mono runs at the family's selected LR (seed 42); seeds 43, 44 for T2X Mono; one General model (seed 42) |
| test | `test-<family>-...` on the selected sweep checkpoint; Mono and seed runs test their own best epoch in-unit |

Every training run: epochs = 10 if the tokenized train set has < 5000 rows else 4, one checkpoint per epoch in
node-local `/dev/shm`, each scored on validation with the paper's protocol scorer; only the best epoch survives
(sweep runs keep it in `keep/`, deleted when not selected). All scoring is on the lane's L40S.

Selection (best epoch and LR) uses a FIXED validation subsample: at most 500 items per (task, language), drawn once
with seed 20260926 and stored in `val_subsample.json` (sha256 in every validation score record; `val_subsample.py`
rebuilds it byte for byte). Only NER xho/zul (817/836 -> 500) and AfriHG xho/zul (1305/1777 -> 500) are cut; every
other validation split has <= 500 items and stays whole. The General model's per-epoch validation uses the same
subsets. Test scoring is always full size.

Beam stage (test only, beside greedy; greedy stays the primary all-architecture number): every selected T2X/AfriHG
checkpoint (T2X Mono seeds 42/43/44, AfriHG Multi + Mono xho/zul, General seeds 42/43/44) is decoded again with the v1
beam settings (5 beams; length penalty 1.0 T2X, 0.7 AfriHG; early stopping), scored with the same NFC chrF, rows with
`decoding=beam` in cells.csv and `data/fft-beam.csv`. MzansiLM uses the generation cache; GDN and xLSTM decode without
it, because FLA 0.5.1's cache cannot be reordered across beams and transformers 4.57.3 never reorders xLSTM's
`cache_params`; xLSTM runs at batch 1. Mamba-2 also decodes cache-free (the default in `BEAM_MODE`; no environment
variable needed). The `collect-beam` unit re-collects after the beam units; `collect` does not wait for them.

Early stopping (user decision 26 Sep 2026, recorded before any unit ran with it; goes into the paper): after each
epoch's validation scoring with the protocol scorer on the fixed validation subsample, training stops once 3
consecutive epochs pass without a strictly greater selection score (ties are not improvement; General: the
six-family mean). So at least 4 epochs run (patience + 1). The best epoch is selected exactly as before (maximum
score, ties to the earlier epoch). Each run's RUN_DONE.json and `runs/<run>/early_stop.json` record `stopped_early`,
`stop_epoch`, `epochs_run`, `planned_epochs` and `best_epoch`. Mechanism: a callback in `train_fft.py` (the training
subprocess) scores each epoch checkpoint in-process through `rollout.val_score` while training waits (CUDA cache
emptied first), keeps the best epoch's weights in `runs/<run>/best/` and sets `should_training_stop`; the Trainer's
eval loss is not used. Units that were already running when this was synced (22:45, 26 Sep) finished with full epochs.

Safeguards (no protocol change):
- Sanity checks after every validation/test score: `sanity/<unit>.json` (SANITY per unit); failures go to
  `ALERTS.txt` in the run and, through `rebalance.py`, to `~/.sallm_fire/alerts.log` on the Mac (one file to watch).
  Classification (News/SIB/Intent/Belebele): one predicted label > 80% of predictions, or score <= the majority
  baseline (weighted F1 of always predicting the majority class; Belebele: majority-answer accuracy). NER/POS: > 20%
  empty outputs, or score <= a trivial baseline (NER all-O = 0; POS majority tag accuracy). T2X/AfriHG: > 20% empty,
  or > 30% of outputs repeating one 4-gram 4+ times.
- Divergence abort in training: a non-finite logged loss, or a loss above 3x the first-epoch mean for 200
  consecutive steps (anomaly-detection NaNs in backward count too) -> `runs/<run>/DIVERGED.json`, unit FAILED_DIVERGED,
  alert, never retried.
- Resume: every epoch's full checkpoint (with optimizer/scheduler/RNG) replaces `runs/<run>/resume/` on scratch; a lane
  that dies (stale heartbeat after 20 min) puts its unit back to pending without using up the retry, and the next lane
  resumes training from that checkpoint (already-scored epochs and the best weights persist). After 4 lane deaths the
  unit is FAILED and alerted. A Python error is retried once, then FAILED and alerted.
- Retiring lanes that run older code: `echo "jobs <id> <id>" > $R/runs/<name>/STOP` stops only those lane jobs after
  their current unit; any other STOP content stops every lane. `python3 selfcheck.py` checks the rules offline.

General regime, trimmed (user decision 26 Sep 2026, recorded before any Multi result was read): no General LR
sweep and no General seeds 43/44 (nor their beam units). Each architecture trains ONE General model, seed 42, at the
LR chosen most often across that architecture's Multi selections (News, SIB-200, Intent, NER, POS, AfriHG, including
any edge-extension LR that won); ties go to the lower LR. `train-general-general-s42` therefore depends on all six
`select-<family>` units and writes `keep/general/LR_TRANSFER.json` (the six chosen LRs and the result). Its per-epoch
validation and epoch selection use the six-family mean on the fixed subsample; test runs in-unit, then
`beam-general-s42`. Mono T2X seeds 43/44 stay.

Speed settings (27 Sep 2026, synced 07:59 SAST; disclose in the paper). Training units that START after this sync use
fused AdamW (`adamw_torch_fused`: same AdamW, betas 0.9/0.95, eps 1e-8, wd 0.01, cosine schedule, clip 1.0) and TF32
matmuls (`torch.backends.cuda.matmul.allow_tf32` and `cudnn.allow_tf32` on), for all four architectures. Applied in
`train_fft.py` (the per-unit training subprocess), so running lanes pick it up at their next unit. Units already
started keep the old settings (foreach AdamW, TF32 matmul off), also when resumed after a lane death: each run writes
`runs/<run>/train_settings.json` at first start and a resumed run without it counts as old. `run_info.json` (and so
RUN_DONE.json's `run_info`) records `train_settings` (`settings` legacy | new-20260927, `optimizer_impl`, `tf32`,
`train_kernel`) and `optimizer` (class, fused, betas, eps, weight decay per group, TF32 flags as the Trainer saw them);
train.log has `FFT_TRAIN_SETTINGS` / `FFT_OPTIMIZER` lines. Old-settings runs: every unit done or running at the sync
(for xLSTM: the POS Multi sweep and the AfriHG Multi sweep; see the notes file for the full list).
Checks: fused AdamW was bit-identical to foreach in the matched-pretraining parity (max param diff 0.0, all four);
a 20-step xLSTM loop with the recipe (tfla_check.py loop, RTX 5090) gives final losses 4.05781 vs 4.05739 (16x256) and
3.60121 vs 3.60132 (4x1024 x 4 accum), with no measurable xLSTM speedup (noise from a shared GPU).

xLSTM training kernel: unchanged (transformers' native chunkwise kernel, as every earlier paper xLSTM run). The TFLA
Triton kernels of mlstm_kernels 2.0.2 (chunkwise--triton_xl_chunk, the matched base's pretraining kernel, and
triton_limit_chunk) intermittently give a wrong forward (rel. error 0.07-0.11 instead of 0.006) or near-orthogonal
gradients (cosine 0.02-0.05) on identical repeated calls at this model's head dims (qk 92, v 184; also 96/192); at
64/128 and 128/256 every call agrees (RTX 5090; model level one first call gave grad norm 1.05e6 vs 2.73).
`tfla.py` pads q/k to 128 and v to 256 (scale 1/sqrt(92) passed to the kernel, outputs sliced): on the 5090 every
repeated call then agrees (12 x 16x256), forward rel. error 0.006 (bf16 native 0.011), input-gradient cosine vs
native_custbw fp32 >= 0.99997 for q/k/v/i and 0.9998/0.9999/0.9990/0.9985 for the forget gate at
16x256/16x192/4x1024/12x2048 (bf16 native_custbw: 0.9991/0.9996/0.9977/0.9955). Model level vs transformers' native
kernel: loss rel. diff <= 1.3e-4, grad cosine 0.9998/0.9981/0.9904 at 16x256/4x1024/4x2048, where the pure-torch
native_custbw kernel gets 0.9993/0.9982/0.9858 (custom-backward kernels treat the stabiliser differently from
autograd). 20-step recipe loop: loss within 0.02% of native, 0.163 -> 0.095 s/step (16x256) and 0.98 -> 0.43 s/step
(4x1024 x 4 accum). The strict gate (every gradient cosine >= 0.999) fails at long sequences, so padded TFLA is NOT
enabled; wiring it in is `use_tfla(model)` in train_fft.py's model hook plus `train_kernel=tfla_padded128`.
`tfla_check.sbatch` repeats the kernel and parity checks on one L40S (not yet run: all 10 were busy);
`pydeps_tfla/` on HEX holds mlstm_kernels 2.0.2 + einops 0.8.2. The matched xLSTM pretraining used the unpadded
kernel on L40S at 92/184.

## 3a. Checkpoints (kept) and the Kombuys archive

Every selected fine-tuned model is kept, weights only (~0.51 GB fp32): each Mono and seed run's best epoch, and
each sweep's selected (lr, epoch). Per architecture: 20 Mono + 6 Multi + T2X (3 seeds) + General (3 seeds) = 32
checkpoints, ~16.3 GB; all four ~65 GB. Sweep runs keep their best epoch only until `select-<family>` has run.
Layout: `runs/<name>/keep/<family>/<run>_e<epoch>/` plus `<run>_e<epoch>.sha256` (manifest, written in-job) and
`<run>_e<epoch>.selected` (marker). HEX /scratch quota (300 GB) had only ~22 GB free at 14:40 on 26 Sep, and pretraining still writes there
(~4 GB per run to go). Peak rollout use is ~25 GB per architecture, so free space before firing (largest
candidates: `masters/sallm_recovery` 30 GB, `results/eval` 18 GB, the T2X pilot 8 GB) and archive as families finish:

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
cd ~/Desktop/sa-architecture-comparison-paper
for a in mzansilm mamba2 xlstm gdn; do
  rsync -a --include='results/***' --include='base_eval/' --include='base_eval/*.json' --exclude='*' \
    hex:$R/runs/$a/ data/fft_raw/$a/
  # compact per-item predictions for the benchmark-standard columns (accuracy, ROUGE, chrF++/BLEU, XTREME-UP NER F1)
  # and the Belebele prompt-1 guard; stdlib only, on a compute node (the raw test outputs are too big for the login node)
  ssh hex srun --account=l40sfree --partition=l40s --cpus-per-task=2 --time=00:30:00 \
    /home/lmbanr001/masters/sallm/.venv/bin/python - $R/runs/$a < scripts/hex/extract_items.py | gzip -n > data/fft_raw/$a/items.jsonl.gz
done
~/Desktop/Masters/sallm/.venv/bin/python scripts/collect_fft.py data/fft_raw/{mzansilm,mamba2,xlstm,gdn}
/opt/homebrew/bin/python3 scripts/build_results.py
```
`collect_fft.py` needs the sallm venv (rouge_score, numpy, sacrebleu) for the extra columns. It recomputes every
Belebele score as prompt 1's acc_norm from `items.jsonl.gz` and warns where `cells.csv` differs (lanes started before
the prompt-1 fix, 297eec6). Metric definitions: the paper repo's `literature/metric_comparability.md`.

| rollout output | paper file | read by |
|---|---|---|
| `base_eval/prompt.json`, `ner.json`, `pos.json` (same schema as the official units) | `data/base-official.csv` (152) | `build_results.py` (Base) |
| cells.csv Mono/Multi rows of News, SIB-200, Intent, NER, POS (seed 42) | `data/monomulti.csv` (144, source `fft`) | `build_results.py`; replaces `collect_monomulti.py`'s rescore/retrain/reselect merge |
| cells.csv T2X/AfriHG rows: Base (`base-gen`), Mono, Multi, General | `data/generation-protocol-v2.csv` (44) | `build_results.py`; replaces `extract_generation_v2.py` (scores are recomputed on HEX with NFC) |
| cells.csv General rows (41 per model) | `data/general-fft.csv` | `build_results.py` (block after the adapter-era General files) |
| all rows + CIs; seeds of General and Mono T2X | `data/fft-cells.csv`, `data/fft-seeds.csv` | text/tables (mean, std over seeds 42/43/44) |

`collect_fft.py` refuses smoke (limited) results and needs all four architectures. Every output carries a language
`family` column (Nguni: zul, xho, ssw, nbl; Sotho-Tswana: sot, tsn, nso; afr, eng, ven, tso on their own).

## 5. Smoke test (stand-in base, tiny cells)

```bash
SMOKE=1 FIX_SPECIAL_IDS=1 LANE_TIME=00:50:00 bash $R/code/fft_rollout/launch.sh mzansilm \
  /scratch/lmbanr001/masters/sallm/checkpoints/sallm-llama-125m/final_model 1 smoke-mzansilm
# per architecture, T2X + General only (every scorer of that architecture):
SMOKE=1 SMOKE_FAMILIES="t2x general" LANE_TIME=00:50:00 bash $R/code/fft_rollout/launch.sh gdn <base> 1 smoke-gdn
```
Smoke = 4 optimizer steps per run (one checkpoint), 20 items per language for every validation and test scorer,
SIB/NER/POS/T2X/General cells (+ a tokenization-only count of every training config). 50-minute lanes backfill
single free GPUs and resubmit themselves. Delete smoke checkpoints afterwards: `rm -rf $R/runs/smoke-*/keep/*/*_e* $R/runs/smoke-*/base`.
While any `mp-full-*` job is pending, hold smoke jobs (`scontrol hold <id>`); pretraining has priority.
