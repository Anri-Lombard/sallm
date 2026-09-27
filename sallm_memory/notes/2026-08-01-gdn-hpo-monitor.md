# GDN HPO monitor — 2026-08-01

## 09:10 SAST frozen test closeout and canonical promotion

- Official held-out job `1151321` completed cleanly (`0:0`) in `00:14:12`. Artifact root: `/scratch/lmbanr001/masters/sallm/results/final/gdn_ner_xho_hpo_tyvese0z_seed42_test_20260801/`. Each of five fixed prompts contains exactly 1,000 official held-out examples.
- Prompt F1 values P1-P5 are `0.5192250373`, `0.6269095182`, `0.6238586156`, `0.6156052783`, `0.6286711253`. Canonical headline is explicitly **best prompt P5 F1 `0.6286711253`**; mean `0.6028539149`; range `0.5192250373-0.6286711253`. This best-of-five test-prompt value is descriptive, not an unbiased estimate.
- Validation selection remained `eval/all_f1` only. Frozen representative checkpoint: seed-42 `tyvese0z`, `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260731/ner_xho/tyvese0z/checkpoint-552`; this represents the three-seed winner without selecting the lucky highest seed. Adapter SHA-256 `11c640ea243d14a42e3deea03bddb270676c99eca993cdc2f8ed3b012e57d8af`; trainer-state SHA-256 `77288e185c0f69fefaf276ad88dba15778b9fdbf772e5d6df226a158baa1ae73`.
- Summary SHA-256 `3436f346c6f9fe268d74bd196c1a7161522f2fdb729459d794fe83fbd35e5f21`; raw results SHA-256 `f0e96598b400294ed8d98c1ec0262e80bfb913f86c18675a746b6e96473b3eb8`.
- Re-read exact source and dependent cells, then promoted only `GatedDeltaNet Results!C4,E4,I4:J4`. Re-read confirms `Comparison Data!I44=0.6287`, coverage `O44=4/4`, and LLaMA remains row winner at `0.7200`; GDN is now 87.3% of the row leader. Full prompt provenance, warning, hashes, artifact, and checkpoint are retained in the `E4` note.
- HEX currently has no active SALLM GPU job. Scratch remains capped at `100 GB`, `93 GB` used (`93.7%`); requested `300 GB` expansion is not active. No broad next HPO wave launched at this occupancy. Held jobs `1117467/1117468` remain untouched.

## 00:06 SAST sweep state

- Reused the active validation-only W&B sweep `yrhn2494`; no duplicate job was
  submitted. HEX A100 job `1150968` remains healthy on `srvrocgpu010` after
  `01:36:09`, with four A100-40GB GPUs, exit state `0:0`, and peak batch RSS
  about `11.5 GiB`. No L40S job, traceback, CUDA OOM, or disk error is present.
- Progress is `6/24` completed or Hyperband-pruned, `4/24` active, and `14/24`
  not yet started. Completed/pruned run IDs are `5rneugxh`, `mb1gmc5p`,
  `ikbh3jki`, `by1rha7z`, `5e2399ui`, and `ystsai87`. Active roots include
  `oxlksi6t`, `u93p2qj7`, `s1df0m4x`, and `u3nzfv3d` under
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260731/ner_xho/`.
- The provisional validation leader is now `oxlksi6t` with task-native NER F1
  `0.5362485616`; its best checkpoint is currently step `207` with a later
  step-230 checkpoint retained while the run continues. Its sampled recipe is
  effective batch `64`, LoRA rank/alpha `64/128`, LR
  `2.1502251418e-4`, constant-with-warmup, dropout `0`, warmup `0.1`, and
  weight decay `0.0798773673`.
- Other observed validation bests include `5e2399ui` `0.4250363901` at step
  276, `u93p2qj7` `0.3828447767` at step 276, `by1rha7z` `0.3095238095` at
  step 138, and `ikbh3jki` `0.2214621633` at step 276. These values are
  provisional validation results and must not be compared directly with
  held-out-test headlines.
- The leader is materially above the previous mono-Xhosa validation behavior,
  supporting optimization/capacity deficiency as an important explanation for
  the former mono-versus-multi gap. That conclusion remains provisional until
  the sweep completes, the top two are repeated across three seeds, and a
  matched exposure/control analysis is performed.
- Sweep artifacts occupy `974 MiB`. HEX scratch is 100 GB total / 91 GB used
  (`91.3%`); the 300 GB quota increase is not active. Active and pruned trial
  artifacts were preserved because final ranking and provenance are not yet
  complete. Held jobs `1117467/1117468` were untouched.
- Selection remains validation-only `eval/all_f1`. After the 24-run screen,
  select the top two recipes for three-seed stability runs, choose by mean and
  stability, freeze one validation-selected checkpoint, then run the official
  held-out test once. No canonical-sheet cell was read or changed this cycle.

Estimated remaining sweep time is roughly 3–6 hours if the observed Hyperband
pruning rate continues.

## 01:09 SAST sweep state

- Reused validation-only W&B sweep `yrhn2494`; no duplicate job was submitted.
  HEX A100 job `1150968` remains healthy on `srvrocgpu010` after `02:38:00`
  with four A100-40GB GPUs. Slurm reports `RUNNING`, exit status `0:0`, and no
  traceback, CUDA OOM, disk-full, or quota error. No L40S work is active.
- Progress is `10/24` completed or Hyperband-pruned, `4/24` active, and `10/24`
  not yet started. Active run IDs are `ipg3wnuv`, `qqublsn7`, `9glbzuc5`, and
  `wmjmrnnx`. Trial roots remain isolated under
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260731/ner_xho/`.
- The current validation leaders remain `oxlksi6t` at task-native NER F1
  `0.5935708946` (`checkpoint-345`) and `u93p2qj7` at `0.5786350148`
  (`checkpoint-690`). The leaders are too close to select from one seed; both
  recipes remain candidates for the required three-seed stability stage.
- Other verified trainer-state bests are `5e2399ui` `0.4250363901`,
  `s1df0m4x` `0.3587581449`, `u3nzfv3d` `0.3482684406`, `by1rha7z`
  `0.3095238095`, `ikbh3jki` `0.2214621633`, and active `ipg3wnuv`
  `0.2152414194`. The three newest active runs do not yet have retained trainer
  states. All values are provisional validation metrics, not test headlines.
- Sweep artifacts occupy `1.2 GiB`. HEX scratch remains 100 GB total / 91 GB
  used (`91.6%`); the requested 300 GB expansion is still not active. No
  artifacts were moved or removed because sweep ranking/provenance is not yet
  complete. Held jobs `1117467/1117468` were untouched.
- Selection remains validation-only `eval/all_f1`. After the 24-run screen,
  rerun the top two recipes across three seeds, choose by mean validation F1
  with stability considered, and freeze one checkpoint before the single
  official held-out-test evaluation. The canonical sheet remains unchanged.

Estimated remaining screen time is roughly 2–5 hours, depending on Hyperband
pruning and the epoch depth reached by the final arms.

## 02:07 SAST sweep state

- Reused validation-only W&B sweep `yrhn2494`; no duplicate work was
  submitted. HEX A100 job `1150968` remains healthy on `srvrocgpu010` after
  `03:36:41`, using four A100-40GB GPUs. Slurm reports `RUNNING`, exit status
  `0:0`, and no traceback, CUDA OOM, disk-full, or quota error. No L40S work is
  active.
- Progress is `14/24` completed or Hyperband-pruned, `4/24` active, and `6/24`
  not yet started. The active run IDs are `qqublsn7`, `l8jblgds`, `kl5ape6t`,
  and `l5wk3roc`. Trial roots remain isolated under
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260731/ner_xho/`.
- The validation leaders remain `oxlksi6t` at task-native NER F1
  `0.5935708946` (`checkpoint-345`) and `u93p2qj7` at `0.5786350148`
  (`checkpoint-690`). Newly finished `ipg3wnuv` improved to `0.4716732543`
  but did not displace either leader. Active `l8jblgds` has reached
  `0.3137592561`; active `qqublsn7` `0.2964684015`; new `kl5ape6t` and
  `l5wk3roc` are at `0.0643758156` and `0.0754834685`. These remain
  provisional validation metrics and are not held-out-test headlines.
- Sweep artifacts occupy `2.1 GiB`. HEX scratch remains capped at 100 GB and
  has risen to 92 GB used (`92.4%`); the requested 300 GB expansion is still
  not active. No artifact was moved or removed because active run state and
  final-ranking provenance are still required. Held jobs `1117467/1117468`
  were untouched.
- Selection remains validation-only `eval/all_f1`. Complete the 24-run screen,
  rerun the top two recipes across three seeds, select by mean validation F1
  with stability considered, and freeze one checkpoint before the single
  official held-out-test evaluation. The canonical sheet remains unchanged.

Estimated remaining screen time is roughly 2–4 hours, depending on pruning and
the training depth reached by the final six arms.

## 03:08 SAST sweep state

- Reused validation-only W&B sweep `yrhn2494`; no duplicate work was
  submitted. HEX A100 job `1150968` remains healthy on `srvrocgpu010` after
  `04:38:12`, with four A100-40GB GPUs allocated. Slurm reports `RUNNING`,
  exit status `0:0`, and no traceback, CUDA OOM, disk-full, or quota error. No
  L40S work is active.
- Progress is `18/24` completed or Hyperband-pruned, `3/24` active, and `3/24`
  not yet started. The active run IDs are `l5wk3roc`, `tyvese0z`, and
  `89k3jkvv`; one of the four W&B agents has naturally exhausted its six-run
  cap. Trial roots remain isolated under
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260731/ner_xho/`.
- The validation leaders remain `oxlksi6t` at task-native NER F1
  `0.5935708946` (`checkpoint-345`) and `u93p2qj7` at `0.5786350148`
  (`checkpoint-690`). Finished `l8jblgds` reached `0.5720698254`, close enough
  to remain relevant to the final ranking but currently third. Active
  `l5wk3roc` has reached `0.5367922175`; active `tyvese0z` `0.5266604303`;
  new `89k3jkvv` has no retained trainer state yet. These are provisional
  validation metrics, not held-out-test headlines.
- Sweep artifacts occupy `2.4 GiB`. HEX scratch remains capped at 100 GB and
  is 92 GB used (`92.8%`); the requested 300 GB expansion is still not active.
  No artifact was moved or removed because active state and final-ranking
  provenance are still required. Held jobs `1117467/1117468` were untouched.
- Selection remains validation-only `eval/all_f1`. Complete the 24-run screen,
  capture all configs and statuses, rerun the top two recipes across three
  seeds, select by mean validation F1 with stability considered, and freeze
  one checkpoint before the single official held-out-test evaluation. The
  canonical sheet remains unchanged.

Estimated remaining screen time is roughly 1–3 hours, depending on pruning and
the depth reached by the last three arms.

## 04:09 SAST sweep state

- Reused validation-only W&B sweep `yrhn2494`; no duplicate work was
  submitted. HEX A100 job `1150968` remains healthy on `srvrocgpu010` after
  `05:37:42`, with four A100-40GB GPUs allocated. Slurm reports `RUNNING`,
  exit status `0:0`, and no traceback, CUDA OOM, disk-full, or quota error. No
  L40S work is active.
- Progress is `20/24` completed or Hyperband-pruned, `3/24` active, and `1/24`
  not yet started. The active run IDs are `89k3jkvv`, `b888dh4l`, and
  `dvefwbqe`. Trial roots remain isolated under
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260731/ner_xho/`.
- Completed `tyvese0z` is the new provisional validation leader at task-native
  NER F1 `0.6112775810`. Its retained best is `checkpoint-552`, reached around
  epoch 12 of the 15-epoch ceiling. Recipe: effective batch 32, LoRA rank/alpha
  32/64, LR `2.8248532361e-4`, constant-with-warmup, dropout `0.1`, warmup
  `0.05`, and weight decay `0.0505264329`. Previous leaders `oxlksi6t`
  (`0.5935708946`) and `u93p2qj7` (`0.5786350148`) move to second and third.
- Active `b888dh4l` has reached `0.5517895236`; active `89k3jkvv`
  `0.4079331942`; active `dvefwbqe` `0.0136601045`. These are provisional
  validation metrics, not held-out-test headlines. No recipe is frozen yet.
- Sweep artifacts occupy `2.8 GiB`. HEX scratch remains capped at 100 GB and
  refreshed from 93.0% to 93.2% used during the checks; the requested 300 GB
  expansion is still not active. No artifact was moved or removed because
  active state and final-ranking provenance are still required. Held jobs
  `1117467/1117468` were untouched.
- Selection remains validation-only `eval/all_f1`. Complete the final four
  trials, capture all configurations and statuses, rerun the final top two
  recipes across three seeds, select by mean validation F1 with stability
  considered, and freeze one checkpoint before the single official held-out
  test. The canonical sheet remains unchanged.

Estimated remaining screen time is roughly 1–2 hours, depending on pruning and
the depth reached by the last active arms.

## 05:10 SAST screen closeout and stability submission

- HEX A100 screen job `1150968` completed cleanly (`0:0`) in `06:29:09` on
  `srvrocgpu010`; all `24/24` W&B sweep `yrhn2494` trials completed or were
  Hyperband-pruned. No traceback, CUDA OOM, disk-full, or quota error occurred.
  The FLA GatedDeltaNet fast path and per-run artifact isolation remained
  active throughout.
- Final validation top two are `tyvese0z` at task-native NER F1
  `0.6112775810` (`checkpoint-552`) and `b888dh4l` at `0.6086044071`
  (`checkpoint-207`). Third is `r06w9rhf` at `0.5944531659`, followed by
  `oxlksi6t` at `0.5935708946`; the final arm did not displace the top two.
- Finalist `tyvese0z`: effective batch 32, LoRA rank/alpha 32/64, LR
  `2.8248532361e-4`, constant-with-warmup, dropout `0.1`, warmup `0.05`,
  weight decay `0.0505264329`. Finalist `b888dh4l`: effective batch 64, LoRA
  rank/alpha 64/128, LR `2.7901463077e-4`, constant-with-warmup, dropout
  `0.1`, warmup `0.03`, weight decay `0.0263425938`. Both original screen runs
  used seed 42.
- Added and locally parsed fixed grid manifests
  `src/conf/sweeps/gdn_ner_xho_stability_tyvese0z.yaml` and
  `src/conf/sweeps/gdn_ner_xho_stability_b888dh4l.yaml`. They reuse each
  finalist exactly and run only missing seeds 17 and 73. Counting the retained
  seed-42 screen runs yields the required three seeds per recipe without two
  duplicate reruns.
- Synced only the two manifests and submitted A100-only stability jobs
  `1151254` (`tyvese0z`, two runs) and `1151255` (`b888dh4l`, two runs). Both
  request two `gpu:ampere` A100-40GB GPUs, eight CPUs, and are currently
  `PENDING`: `1151254` for resources, `1151255` for priority. No L40S job or
  duplicate stability run exists.
- Stability artifact roots are
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260731/ner_xho_stability/tyvese0z/`
  and
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260731/ner_xho_stability/b888dh4l/`.
  Selection remains validation-only `eval/all_f1`: compare three-seed means
  and variance, then freeze one validation-selected checkpoint before a single
  official held-out test. No test or canonical-sheet write occurred.
- HEX scratch remains 100 GB / 93 GB used (`93.2%`); the 300 GB expansion is
  not active. Screen artifacts remain protected for complete provenance.
  `save_only_model=true` and `save_total_limit=1` bound new stability storage.
  Held jobs `1117467/1117468` were untouched.

Queue-start ETA depends on A100 availability; once running, the four new seed
runs are expected to finish in roughly 1–3 hours based on the screen finalists.

## 06:09 SAST stability state

- Both validation-only stability jobs started together on `srvrocgpu010` and
  exactly fill its four A100-40GB GPUs: `1151254` (`tyvese0z`) is `RUNNING`
  after `00:41:38`, and `1151255` (`b888dh4l`) is `RUNNING` after `00:40:47`.
  Each job has two one-GPU W&B agents. No L40S work, traceback, CUDA OOM,
  disk-full, or quota error is present.
- W&B sweep `6f1rjqya` / job `1151254` active run IDs are `01wv2ha8` and
  `0rngpkpg`; their provisional validation `eval/all_f1` bests are
  `0.5088959298` and `0.5200672915`. W&B sweep `w1r61z4q` / job `1151255`
  active run IDs are `8d1lpkib` and `apa52myz`; provisional bests are
  `0.4302186879` and `0.3684102564`. All four are still training and these
  values are not final rankings or held-out-test results.
- The four runs cover seeds 17 and 73 for their respective fixed recipes;
  exact run-to-seed mapping will be captured from finalized W&B configs at
  closeout. The retained seed-42 screen runs remain `tyvese0z` F1
  `0.6112775810` and `b888dh4l` `0.6086044071`.
- Stability artifacts occupy `464 MiB` under
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260731/ner_xho_stability/`.
  HEX scratch remains capped at 100 GB and is 93 GB used (`93.7%`); the 300 GB
  expansion is not active. No artifact was moved or removed. Held jobs
  `1117467/1117468` were untouched.
- Selection remains validation-only `eval/all_f1`. After all four runs finish,
  compute each recipe's three-seed mean and dispersion, freeze the selected
  validation winner/checkpoint, and only then run one official held-out test.
  The canonical sheet remains unchanged.

Estimated remaining stability time is roughly 1–2 hours based on current
training depth and the original finalist runtimes.

## 07:09 SAST stability state

- A100 stability jobs remain healthy on `srvrocgpu010`: `1151254`
  (`tyvese0z`) is `RUNNING` after `01:41:08`, and `1151255` (`b888dh4l`) is
  `RUNNING` after `01:40:17`. Each job has finished one of its two runs; the
  other continues. No L40S work, traceback, CUDA OOM, disk-full, or quota
  error is present.
- `tyvese0z` seed 17 run `01wv2ha8` completed at validation `eval/all_f1`
  `0.5977623868`, retaining `checkpoint-414`. Seed 73 run `0rngpkpg` remains
  active and has reached `0.6118340290`, retaining `checkpoint-552` so far.
  The retained seed 42 screen value is `0.6112775810`.
- `b888dh4l` seed 73 run `8d1lpkib` completed at validation `eval/all_f1`
  `0.5471969334`, retaining `checkpoint-161`. Seed 17 run `apa52myz` remains
  active and has reached `0.6014634146`, retaining `checkpoint-299` so far.
  The retained seed 42 screen value is `0.6086044071`.
- The run-to-seed mapping was recovered from the ordered W&B agent config and
  launch records in `slurm-1151254.out` and `slurm-1151255.out`. Current
  provisional three-seed means are about `0.6070` for `tyvese0z` and `0.5858`
  for `b888dh4l`, but selection remains open because both active best metrics
  can still improve.
- Stability artifacts occupy `575 MiB` under
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260731/ner_xho_stability/`.
  HEX scratch remains capped at 100 GB and is 93 GB used (`93.8%`); the 300 GB
  expansion is not active. No artifact was moved or removed. Held jobs
  `1117467/1117468` were untouched.
- Selection remains validation-only `eval/all_f1`. Freeze no recipe until both
  active runs complete and the final three-seed means and dispersion are
  audited. No held-out test or canonical-sheet write occurred.

Estimated remaining stability time is under one hour if the active runs finish
at the same depth as their screen counterparts.

## 08:12 SAST stability closeout and frozen test

- Stability jobs completed cleanly on A100 node `srvrocgpu010`: `1151254`
  (`tyvese0z`) in `01:48:59`, and `1151255` (`b888dh4l`) in `01:48:39`, both
  exit `0:0`. All four missing seed runs finished; no traceback, CUDA OOM,
  disk-full, or quota error occurred.
- `tyvese0z` validation `eval/all_f1` by seed is: seed 17 `0.5977623868`
  (`01wv2ha8`, `checkpoint-414`), seed 42 `0.6112775810` (`tyvese0z`,
  `checkpoint-552`), seed 73 `0.6164985777` (`0rngpkpg`, `checkpoint-644`).
  Mean `0.6085128485`, sample SD `0.0096692307`, range
  `0.5977623868-0.6164985777`.
- `b888dh4l` validation `eval/all_f1` by seed is: seed 17 `0.6014634146`
  (`apa52myz`, `checkpoint-299`), seed 42 `0.6086044071` (`b888dh4l`,
  `checkpoint-207`), seed 73 `0.5471969334` (`8d1lpkib`, `checkpoint-161`).
  Mean `0.5857549184`, sample SD `0.0335825416`, range
  `0.5471969334-0.6086044071`.
- Validation-only selection therefore freezes the `tyvese0z` recipe. The
  representative seed-42 checkpoint is closest to the three-seed mean and was
  selected before stability; it is frozen at
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260731/ner_xho/tyvese0z/checkpoint-552`.
  This avoids choosing the highest lucky seed. Directory size is `109,927,977`
  bytes; adapter weight SHA-256 is
  `11c640ea243d14a42e3deea03bddb270676c99eca993cdc2f8ed3b012e57d8af`;
  trainer-state SHA-256 is
  `77288e185c0f69fefaf276ad88dba15778b9fdbf772e5d6df226a158baa1ae73`.
- Synced the existing matched Xhosa NER evaluation launcher/config with
  relative paths and submitted exactly one official frozen held-out test as
  A100 job `1151321`. The job is `RUNNING` on `srvrocgpu010`, correctly loaded
  the GDN base plus the frozen local adapter, and writes to
  `/scratch/lmbanr001/masters/sallm/results/final/gdn_ner_xho_hpo_tyvese0z_seed42_test_20260801/`.
  Initial health is clean. A first flat rsync also left harmless root-level
  copies `launch_evaluation.sh` and `run_mamba_masakhaner_xho.yaml` on HEX;
  they are inert and were not deleted without explicit cleanup authority.
- The official test uses all five fixed MasakhaNER Xhosa prompts. It cannot
  affect checkpoint selection. On completion, report an explicitly labelled
  **best prompt** headline plus mean, range, winner/template, all prompt values,
  artifact hashes, and the descriptive-not-unbiased warning before any sheet
  promotion.
- HEX scratch remains 100 GB / 93 GB used (`93.7%`); the 300 GB expansion is
  not active. Stability artifacts occupy `470 MiB`. No checkpoint was removed;
  held jobs `1117467/1117468` were untouched.

Estimated held-out evaluation time is roughly one hour based on the previous
matched 5,000-generation Xhosa NER run.

## 17:55 SAST Xhosa closeout, storage refresh, and next-wave canaries

- Xhosa NER HPO is finalized. Screen `1150968`, stability jobs `1151254` and
  `1151255`, and official frozen-test job `1151321` all completed `0:0`.
  Validation-only `eval/all_f1` selected recipe `tyvese0z`: three-seed mean
  `0.6085128485`, sample SD `0.0096692307`. The representative seed-42 winner
  remains
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260731/ner_xho/tyvese0z/checkpoint-552`.
  Adapter SHA-256 is
  `11c640ea243d14a42e3deea03bddb270676c99eca993cdc2f8ed3b012e57d8af`;
  trainer-state SHA-256 is
  `77288e185c0f69fefaf276ad88dba15778b9fdbf772e5d6df226a158baa1ae73`.
- Official Xhosa test provenance is under
  `/scratch/lmbanr001/masters/sallm/results/final/gdn_ner_xho_hpo_tyvese0z_seed42_test_20260801/`.
  The explicitly labelled **best prompt** is P5 F1 `0.6286711253`; mean
  `0.6028539149`; range `0.5192250373-0.6286711253`. This best-of-five
  test-prompt statistic is descriptive, not an unbiased estimate. Verified
  promotion is `GatedDeltaNet Results!E4` and `Comparison Data!I44=0.6287`.
- Archived 23 losing Xhosa screen directories plus the stability tree to
  Kombuys at
  `/scratch/alombard/masters/sallm/archive/hex_20260801/checkpoints/gdn_hpo_20260731/`.
  The 488-file checksum dry-run reported zero differences before the exact
  cold HEX copies were removed. The frozen winner and canonical test artifact
  remain on HEX. Refreshed HEX quota is `90/100 GB` (`90.5%`); the requested
  300 GB expansion is still inactive.
- Slurm now reports former held jobs `1117467/1117468` as `CANCELLED by 0`,
  both exit `0:0`. This orchestrator did not cancel or release them; the state
  is recorded as external and they will not be recreated without a concrete
  need.
- Added and locally validated matched 24-trial Bayesian/Hyperband manifests
  `src/conf/sweeps/gdn_ner_zul.yaml` and
  `src/conf/sweeps/gdn_ner_tsn.yaml`. Both search ranks `16/32/64`, effective
  batches `32/64`, use the FLA-compatible GDN LoRA targets, isolate outputs,
  cap retained checkpoints with `save_total_limit=1`, and select task-native
  validation `eval/all_f1`. Local checks were `SWEEPS_OK`, HPO tests `2 passed`,
  and launcher syntax success.
- Synced only those two manifests to Kombuys and verified SHA-256 values
  `6a3d76139e01b25fcbd12c9a132b5aa081de430180606a645d4598dc5742a07e`
  (Zulu) and
  `1739052dd910760cea9256b07985150c4493a34bfb9b40d3ed7027310403d03e`
  (Tswana). The FLA import/fast-path guard passed.
- Kombuys tmux task `gdn-ner-zul-tsn-canaries` is running sequential compact
  one-step validation-only Zulu then Tswana canaries on GPU 0 (RTX 5090).
  Each canary asserts the exact target language, `train`/`validation` splits,
  absence of a test split, GDN checkpoint/architecture, and
  `eval_all_f1` selection before training. Isolated roots are
  `/scratch/alombard/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul_canary`
  and
  `/scratch/alombard/masters/sallm/checkpoints/gdn_hpo_20260801/ner_tsn_canary`;
  logs are
  `/scratch/alombard/masters/sallm/logs/gdn_hpo_20260801_ner_zul_canary.log`
  and
  `/scratch/alombard/masters/sallm/logs/gdn_hpo_20260801_ner_tsn_canary.log`.
  Initial health is normal model initialization with no traceback or OOM.
- No SALLM HEX GPU work is active. A100 is the only occupied HEX partition
  family: other users hold three A100-40GB GPUs on `srvrocgpu010` and two
  A100-80GB GPUs on `srvrocgpu011`. Broad Zulu submission remains pending the
  canary result and an exact duplicate/output audit; no L40S work will be
  launched concurrently.

## 18:02 SAST Zulu canary gate and production sweep launch

- Kombuys tmux task `gdn-ner-zul-tsn-canaries` completed both sequential
  validation-only checks without traceback, CUDA OOM, or disk failure. Zulu
  completed in `134.7373s` at
  `/scratch/alombard/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul_canary`
  with log
  `/scratch/alombard/masters/sallm/logs/gdn_hpo_20260801_ner_zul_canary.log`;
  Tswana completed in `90.7803s` at
  `/scratch/alombard/masters/sallm/checkpoints/gdn_hpo_20260801/ner_tsn_canary`
  with log
  `/scratch/alombard/masters/sallm/logs/gdn_hpo_20260801_ner_tsn_canary.log`.
  Both passed the language/split/checkpoint/metric assertions and compiled the
  FLA TileLang fast path. These one-step runs are infrastructure checks, not
  scientific results.
- Re-ran local HPO tests and launcher syntax successfully. Exact duplicate
  audit found no prior or active Zulu HPO job, no production output root, and
  no production log. Synced only `gdn_ner_zul.yaml` and `gdn_ner_tsn.yaml` to
  HEX; remote hashes match the verified local hashes. The already-deployed
  launcher and trial runner also matched local SHA-256 exactly.
- Submitted one production Zulu screen as HEX job `1154777`, W&B sweep
  `ctbysve8`. The initial four-`ampere` request was pending behind the busy
  40-GB node; retargeting the same pending job to four `amperemk` GPUs exposed
  an association GRES limit. Without cancelling or duplicating the job, it was
  retargeted to the two available A100-80GB GPUs under `nlpgroup80` and entered
  `RUNNING` on `srvrocgpu011` at `17:59:55` SAST.
- Job `1154777` is healthy. The compute-node FLA guard passed, W&B launched two
  agents with 12-run caps each, and initial isolated run IDs are `a8zz1dgr`
  and `d7r8mpjk` under
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul/`.
  Both loaded the intended GDN checkpoint and rank-64 FLA-compatible LoRA
  targets; Zulu data counts are train `1,441`, validation `836`. Initial GPU
  memory is about `8.7 GiB` per A100-80GB. No L40S or other SALLM GPU job is
  active.
- Selection remains task-native validation `eval/all_f1` only. The 24-trial
  Bayesian/Hyperband screen will promote the top two recipes to three-seed
  stability; one representative validation-selected checkpoint will then be
  frozen before any official held-out test. No sheet read or write occurred.
- HEX remains `90/100 GB` (`90.5%`); the 300 GB expansion is inactive. The
  bounded one-checkpoint-per-trial design previously used about `2.4 GiB` for
  24 Xhosa arms, so the single Zulu wave has adequate verified headroom, but
  Tswana will not overlap it. Estimated Zulu screen completion is roughly
  `10-16h` on two GPUs, subject to Hyperband pruning.

## 19:00 SAST Zulu screen state

- Reused production job `1154777` / W&B sweep `ctbysve8`; no duplicate work
  was submitted. It remains `RUNNING` on `srvrocgpu011` after `01:00:01` with
  two A100-80GB GPUs under `nlpgroup80`.
- Progress is `2/24` completed or Hyperband-pruned, `2/24` active, and `20/24`
  not yet started. Finished run IDs are `d7r8mpjk` and `w7jhm2s2`; active run
  IDs are the continuing `a8zz1dgr` and new `ya72osah`. All roots are isolated
  under
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul/`.
- Provisional validation `eval/all_f1` bests from retained trainer states are
  `a8zz1dgr` `0.3213313162` (`checkpoint-161`), `d7r8mpjk`
  `0.0610000000` (`checkpoint-69`), `w7jhm2s2` `0.0495790458`
  (`checkpoint-69`), and early active `ya72osah` `0.0035756853`
  (`checkpoint-46`). These are validation-only screen values, not final
  rankings or held-out-test headlines.
- Health is normal: both agents continue training, batch peak RSS is about
  `6.1 GiB`, no traceback, CUDA OOM, disk-full, quota, or nonzero-run-exit
  appears. Repeated W&B internal logger messages about the existing
  `/home/lmbanr001/.cache` path are nonfatal; each affected run subsequently
  creates a stream and trains normally, so no intervention is warranted.
- Sweep artifacts occupy `463 MiB`. Refreshed HEX quota is `90/100 GB`
  (`91.0%` after rounding); the 300 GB expansion remains inactive. No L40S or
  other SALLM GPU job is active, and Tswana remains gated behind Zulu.
- Metric selection remains task-native validation `eval/all_f1`; after the
  24-run screen, repeat the top two across three seeds, select by mean and
  stability, and freeze one representative checkpoint before the single
  official held-out test. Estimated remaining screen time is roughly `8-14h`,
  depending on pruning and the depth reached by the current leader.

## 20:01 SAST Zulu screen state

- Reused job `1154777` / sweep `ctbysve8`; it remains healthy on two
  A100-80GB GPUs after `02:01:09`. No duplicate or second-partition work was
  launched.
- Progress is `3/24` completed or Hyperband-pruned, `2/24` active, and `19/24`
  not yet started. Finished IDs are `d7r8mpjk`, `w7jhm2s2`, and
  `a8zz1dgr`; active IDs are `ya72osah` and newly started `urxj18hh`.
- The finalized single-seed screen leader is now `a8zz1dgr` at validation
  `eval/all_f1=0.4707317073`, retaining `checkpoint-345`. Active
  `ya72osah` has reached `0.3819699499` at `checkpoint-414`; earlier pruned
  runs `d7r8mpjk` and `w7jhm2s2` remain at `0.0610000000` and
  `0.0495790458`. `urxj18hh` has not yet produced a retained trainer state.
  These are provisional validation values, not stable selections or test
  headlines.
- Slurm and targeted logs show no traceback, CUDA OOM, disk-full, quota, kill,
  or failed-run exit. Batch peak RSS is about `6.4 GiB`. Artifacts remain
  isolated under
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul/`
  and occupy `465 MiB`.
- HEX quota remains `90/100 GB` (`91.0%` after rounding); the 300 GB expansion
  is inactive. Metric selection remains validation-only `eval/all_f1`, with
  top-two three-seed stability and representative-checkpoint freezing required
  before any held-out test. Estimated remaining screen time remains `8-14h`.

## 21:18 SAST Zulu screen state

- Reused job `1154777` / sweep `ctbysve8`; it is `RUNNING` after `03:18:17`
  on the same two A100-80GB GPUs. No duplicate, L40S, or overlapping Tswana
  work was launched.
- Progress is `7/24` completed or Hyperband-pruned, `2/24` active, and `15/24`
  not yet started. Completed IDs are `d7r8mpjk`, `w7jhm2s2`, `a8zz1dgr`,
  `urxj18hh`, `w4emz278`, `ya72osah`, and `zkzjjdzn`; active IDs are
  `m3wecmma` and newly started `uqk7kla6`.
- Validation-only leaders are close: completed `a8zz1dgr`
  `eval/all_f1=0.4707317073` at `checkpoint-345`, followed by completed
  `ya72osah` `0.4643150123` at `checkpoint-690`. Active `m3wecmma` has reached
  `0.2175632911` at `checkpoint-184`; all other observed arms are at or below
  `0.0610`. This proximity reinforces the requirement for three-seed finalist
  stability rather than single-seed selection. None is a held-out-test value.
- Health remains clean: no traceback, CUDA OOM, disk-full, quota, kill, or
  failed-run exit. Batch peak RSS is about `7.0 GiB`.
- The sweep now occupies `1.2 GiB`. Seven roots retain one checkpoint; two
  pruned roots (`urxj18hh`, `w4emz278`) retain both best and last checkpoints,
  the expected Transformers exception when `load_best_model_at_end=true` and
  the final checkpoint differs from the best even with `save_total_limit=1`.
  Per-root size is `96-211 MiB`; projected full-screen storage remains within
  current headroom, so active artifacts were not touched.
- HEX quota is `91/100 GB` (`91.7%`); 300 GB remains inactive. Selection stays
  validation-only `eval/all_f1`, followed by top-two three-seed stability and
  representative-checkpoint freezing before test. Estimated remaining screen
  time is `6-10h`, depending on pruning depth.

## 22:11 SAST Zulu screen state

- Reused job `1154777` / sweep `ctbysve8`; it remains `RUNNING` after
  `04:11:11` on two A100-80GB GPUs. No duplicate or cross-partition work was
  launched.
- Progress is `8/24` completed or Hyperband-pruned, `2/24` active, and `14/24`
  not yet started. `uqk7kla6` completed since the previous pass; active IDs
  are continuing `m3wecmma` and new `rhups1t5`.
- Active `m3wecmma` is the new provisional validation leader at
  `eval/all_f1=0.4889260343`, with the best first observed at
  `checkpoint-460` and retained again at `checkpoint-506`. Completed
  `a8zz1dgr` (`0.4707317073`) and `ya72osah` (`0.4643150123`) remain close;
  active `rhups1t5` has reached `0.1531258606` at `checkpoint-184`.
  These remain validation-only screen values, not stable selections or test
  headlines.
- Health is clean: no traceback, CUDA OOM, disk-full, quota, kill, or
  failed-run exit. Batch peak RSS is about `7.1 GiB`.
- Artifacts occupy `1.4 GiB` under
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul/`.
  HEX is `91/100 GB` (`91.9%`); current growth remains within headroom, so no
  active or ranking-relevant artifact was moved or removed. The 300 GB
  expansion remains inactive.
- Selection remains task-native validation `eval/all_f1`, followed by top-two
  three-seed stability and representative checkpoint freezing before one
  official held-out test. Estimated remaining screen time is `6-10h`.

## 23:12 SAST Zulu screen state

- Reused job `1154777` / sweep `ctbysve8`; it remains `RUNNING` on two
  A100-80GB GPUs after `05:11:52`. No duplicate or cross-partition work was
  launched.
- Progress is `10/24` completed or Hyperband-pruned, `2/24` active, and
  `12/24` not yet started. `m3wecmma` and `otkzwmgd` completed since the last
  pass. Active IDs are `rhups1t5` and newly started `1acqnc84`.
- Completed `m3wecmma` remains the provisional validation leader at
  `eval/all_f1=0.4889260343`, retaining `checkpoint-460`. Completed
  `a8zz1dgr` (`0.4707317073`) and `ya72osah` (`0.4643150123`) remain second
  and third. Active `rhups1t5` has reached `0.3795688847` at
  `checkpoint-414`, and active `1acqnc84` `0.2699386503` at
  `checkpoint-69`. All are validation-only screen values, not stable
  selections or held-out-test headlines.
- Health is clean: no traceback, CUDA OOM, disk-full, quota, kill, or
  failed-run exit. Batch peak RSS is about `7.1 GiB`.
- Artifacts occupy `1.5 GiB` under
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul/`.
  HEX is `92/100 GB` (`92.0%`); projected screen growth still fits the current
  headroom, so no active or provenance-relevant artifact was touched. The
  300 GB expansion remains inactive.
- Selection remains task-native validation `eval/all_f1`, followed by top-two
  three-seed stability and representative-checkpoint freezing before the
  official held-out test. Estimated remaining screen time is `5-8h`.
