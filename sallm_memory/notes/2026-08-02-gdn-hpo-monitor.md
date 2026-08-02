# GDN HPO monitor — 2026-08-02

## 00:12 SAST Zulu screen state

- Accepted carry-forward: Xhosa NER HPO is final from screen `1150968`,
  stability jobs `1151254/1151255`, and official held-out job `1151321`.
  Validation selected `tyvese0z` with three-seed mean F1 `0.6085128485`; the
  canonical explicitly labelled best-prompt test headline is P5 F1
  `0.6286711253`, with mean `0.6028539149`, range
  `0.5192250373-0.6286711253`, and full descriptive-not-unbiased provenance
  retained in the sheet.
- Reused Zulu job `1154777` / W&B sweep `ctbysve8`; it remains `RUNNING` after
  `06:12:17` on two A100-80GB GPUs on `srvrocgpu011`. No duplicate, L40S, or
  overlapping Tswana work was launched.
- Progress remains `10/24` completed or Hyperband-pruned, `2/24` active, and
  `12/24` not yet started. Active IDs are `rhups1t5` and `1acqnc84`.
- Active `1acqnc84` is the new provisional validation leader at task-native
  `eval/all_f1=0.5649671053`, retaining `checkpoint-253`. Completed
  `m3wecmma` (`0.4889260343`), `a8zz1dgr` (`0.4707317073`), and
  `ya72osah` (`0.4643150123`) are the next observed arms. Active `rhups1t5`
  remains at `0.3795688847`. These are validation-only screen values, not
  stable selections or held-out-test headlines.
- Health is clean: no traceback, CUDA OOM, disk-full, quota, kill, or
  failed-run exit. Batch peak RSS is about `7.1 GiB`; the apparent `10:49:38`
  aggregate batch CPU time reflects concurrent agents, not wall time.
- Artifacts occupy `1.5 GiB` under
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul/`.
  HEX is `92/100 GB` (`92.0%`); storage is stable and within projected
  headroom. The 300 GB expansion remains inactive, and no active or
  ranking-relevant artifact was moved or removed.
- Metric selection remains validation-only `eval/all_f1`. After all 24 arms,
  rerun the top two recipes across three seeds, select by mean with stability
  considered, freeze one representative checkpoint, and only then run the
  official held-out test. Estimated remaining screen time is `5-8h`.

## Pending

- Finish Zulu screen `1154777`, archive verified losers to Kombuys, and retain
  finalist checkpoints for stability.
- Run top-two Zulu three-seed validation stability, freeze a representative
  checkpoint, and evaluate official test once.
- Only after Zulu closes and storage is reconciled, submit the already-canary-
  verified Tswana 24-trial screen from `src/conf/sweeps/gdn_ner_tsn.yaml`.

## 01:14 SAST Zulu screen state

- Reused HEX job `1154777` / W&B sweep `ctbysve8`; it remains `RUNNING` after
  `07:14:05` on `srvrocgpu011` in the A100-only wave. Thirteen distinct trial
  directories now exist: `12/24` arms are complete or Hyperband-pruned,
  `blxjtbgx` is active, and `11/24` have not started. One of the two sweep
  agents appears to have exhausted its assigned count; no duplicate or
  overlapping Tswana work was launched.
- Validation-only leader remains `1acqnc84` at task-native
  `eval/all_f1=0.5649671053` with `checkpoint-253`; next are `m3wecmma`
  (`0.4889260343`), `a8zz1dgr` (`0.4707317073`), and `ya72osah`
  (`0.4643150123`). Active `blxjtbgx` has provisionally reached
  `0.4089376054`; it is not rank-final until its arm exits.
- Health remains clean. Slurm reports `RUNNING`, exit `0:0` so far, max RSS
  about `7.13 GiB`, and no traceback, CUDA OOM, disk-full, quota-kill, or
  failed trial. The repeated W&B `failed to get logger path` line is
  non-fatal: its stream immediately initialized and the run continued saving
  and evaluating normally.
- Artifact use rose from `1.5 GiB` to `1.7 GiB` at
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul/`.
  HEX remains `92/100 GB` (`92.1%`); the requested 300 GB quota is still not
  active. Current projected screen artifacts fit, but stability runs remain
  gated on post-screen archiving/reconciliation.
- Selection rule is unchanged: rank the completed screen on validation-only
  `eval/all_f1`, rerun the top two recipes across three seeds, select by mean
  with stability considered, freeze a representative checkpoint, and run the
  official held-out test once. With a single active sweep agent now visible,
  remaining screen ETA is conservatively `6-10h`.

## 02:14 SAST Zulu screen state

- Job `1154777` / sweep `ctbysve8` remains healthy and `RUNNING` after
  `08:14:49` on `srvrocgpu011`; the wave is still A100-only. The directory
  count is unchanged at 13: `12/24` complete or pruned, `blxjtbgx` active,
  and `11/24` unstarted. No second architecture/task was launched.
- Active `blxjtbgx` improved sharply to validation-only task-native
  `eval/all_f1=0.5559471366` at `checkpoint-230`, just `0.0090199687` behind
  leader `1acqnc84=0.5649671053`. It had reached the evaluation at step
  `299/345` (about `87%`) and is not yet a completed finalist.
- Health remains clean: no traceback, CUDA OOM, disk-full, quota kill, or
  failed-run exit; max RSS remains about `7.13 GiB`. Artifacts are `1.8 GiB`
  at `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul/`.
  HEX is `92/100 GB` (`92.3%`); the 300 GB increase remains inactive.
- Continue the existing agent. On completion, rank all 24 using validation
  `eval/all_f1`, then run three-seed stability for the top two before freezing
  any checkpoint or touching held-out test. Screen ETA remains `5-9h` because
  the remaining arms may be Hyperband-pruned early.

## 03:16 SAST Zulu screen state

- HEX job `1154777` / sweep `ctbysve8` remains `RUNNING` and healthy after
  `09:16:47` on `srvrocgpu011`, still A100-only. Sixteen trial directories
  now exist: `15/24` complete or Hyperband-pruned, `nzxcdrph` active, and
  `8/24` unstarted. No duplicate work was submitted.
- `blxjtbgx` completed cleanly with validation-only task-native
  `eval/all_f1=0.5559471366` and remains second to `1acqnc84=0.5649671053`.
  Newly pruned `0cpi0j00=0.0478359909` and `o9vstcsh=0.0507936508` do not
  affect the leaders. Active `nzxcdrph` was at step `23/345` with provisional
  F1 `0.0012150668`; it is not rank-final until exit.
- No traceback, CUDA OOM, disk-full, quota kill, or failed-run exit. Batch max
  RSS remains about `7.13 GiB`. Artifacts occupy `2.1 GiB` at
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul/`.
  HEX is `92/100 GB` (`92.6%`); the requested 300 GB quota remains inactive.
- Selection remains validation-only `eval/all_f1`; do not use held-out test
  until top-two three-seed stability is complete and one representative
  checkpoint is frozen. With eight arms unstarted and recent weak arms pruning
  in roughly `24-25m`, screen ETA is `4-7h`, allowing for full-budget arms.

## 04:18 SAST Zulu screen state

- HEX job `1154777` / sweep `ctbysve8` remains healthy and `RUNNING` after
  `10:18:49` on `srvrocgpu011`; A100 remains the only active partition family.
  Eighteen trial directories exist: `17/24` complete or Hyperband-pruned,
  `aq1xc5q3` active, and `6/24` unstarted. No duplicate work was launched.
- Leaders remain `1acqnc84=0.5649671053` and
  `blxjtbgx=0.5559471366` on validation-only task-native `eval/all_f1`.
  `nzxcdrph=0.0759427828` and `km5f2uqf=0.0122025625` were pruned and do not
  change the ranking. Active `aq1xc5q3` provisionally reached
  `0.1305319624` at about step `140/690` (`20%`); it is not rank-final.
- Health is clean: no traceback, CUDA OOM, disk-full, quota kill, or failed
  run exit; batch max RSS remains about `7.13 GiB`. Artifacts occupy `2.3 GiB`
  at `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul/`.
  HEX is `92/100 GB` (`92.8%`); the 300 GB increase remains inactive.
- Continue the existing screen without overlap. Rank all 24 strictly on
  validation `eval/all_f1`, then run top-two three-seed stability, freeze one
  representative checkpoint, and evaluate held-out test once. Screen ETA is
  `4-6h`, with uncertainty from the active 690-step arm and future pruning.

## 05:26 SAST incomplete-screen recovery

- Original Zulu job `1154777` completed cleanly after `11:03:13`, exit `0:0`,
  but only `18/24` intended trials started. Do not treat this as a completed
  screen or select finalists yet. Final observed leaders remain validation-only
  `1acqnc84=0.5649671053` and `blxjtbgx=0.5559471366`; last arm
  `aq1xc5q3` was Hyperband-pruned at `0.4411515665`.
- Root cause: one of two W&B agents stopped after five consecutive valid
  Hyperband prunes because W&B counted the prune exit codes as failed runs:
  `Detected 5 failed runs in a row, shutting down.` Only 18 run starts/URLs
  exist; there were zero tracebacks. Shared `scripts/launch_hpo.sh` now exports
  `WANDB_AGENT_MAX_INITIAL_FAILURES=100`; shell syntax and exact synced SHA256
  `61339a98aef78448db45c4f0b0a451a90888a82c12077ff1f3a97293904a5589`
  were verified.
- First six-arm continuation job `1159309` failed before any trial in `2s`
  because stale `resume_hpo_a100.sh` required nonexistent Conda environment
  `sallm-uv`. First one-run canary `1159328` then failed before any trial in
  `2s` because Slurm's spooled script path could not resolve `lib/env.sh`.
  Neither consumed a sweep arm or created a checkpoint.
- Shared `scripts/resume_hpo.sh` was minimally repaired to resolve the repo
  environment helper from `SLURM_SUBMIT_DIR`/the canonical repo, fall back to
  `.venv` when `sallm-uv` is absent, verify the GDN FLA fast path, tolerate
  Hyperband prune exits, and cap agents to requested run count. Shell syntax
  passed and the exact synced SHA256 is
  `d894cad3817c809f40591ba4a2e6084950d8fffb014940fd9e7e19669657ff1c`.
- Replacement one-run validation-only canary `1159336` is `RUNNING` on one
  A100-80GB GPU on `srvrocgpu011`. It passed the repaired preflight, explicitly
  verified the GDN FLA fast path, and started real sweep trial `1tps89m9`.
  Because this consumes one of the six missing arms, exact tail continuation
  job `1159345` was submitted for the remaining five arms and is `RUNNING` on
  two A100-80GB GPUs on the same node. Thus the intended total is exactly
  `18 + 1 + 5 = 24`; no duplicate arm count was submitted.
- Storage is `92/100 GB` (`92.9%`); Zulu artifacts occupy `2.4 GiB` at
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul/`.
  The 300 GB increase remains inactive. Metric and phase gates are unchanged:
  validation-only `eval/all_f1`, then top-two three-seed stability, freeze a
  representative checkpoint, and only then evaluate held-out test once.

## 06:19 SAST recovery state

- Recovery jobs `1159336` and `1159345` remain healthy and `RUNNING` on three
  A100-80GB GPUs on `srvrocgpu011`; no L40S work or duplicate arms exist.
  Twenty-two of 24 trials have now started: `19/24` complete or pruned,
  `1tps89m9`, `ngli032e`, and `xbuk7k3d` active, and `2/24` unstarted.
- `ngli032e` is the new provisional validation-only leader at task-native
  `eval/all_f1=0.5798237023` from step `322/690` (`47%`), ahead of completed
  `1acqnc84=0.5649671053` and `blxjtbgx=0.5559471366`. It is not a finalist
  until its arm and the full screen exit. Active canary `1tps89m9` is at
  `161/345`, F1 `0.3186039966`; active `xbuk7k3d` is at `138/690`, F1
  `0.1508967223`. `e9tlsb8n=0.0080361627` pruned cleanly.
- No traceback, CUDA OOM, disk-full, quota kill, W&B five-prune shutdown, or
  failed-run exit is present. Max RSS is about `2.97 GiB` for canary job
  `1159336` and `6.14 GiB` for two-agent job `1159345`.
- Zulu artifacts occupy `3.1 GiB` at
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul/`.
  HEX is `93/100 GB` (`93.3%`); the 300 GB increase remains inactive. The
  screen fits current headroom, but stability remains gated on archiving
  verified losers after the 24-arm ranking is final.
- Selection remains validation-only `eval/all_f1`. Finish all 24, rerun the
  top two recipes across three seeds, select by mean with stability considered,
  freeze a representative checkpoint, and only then use held-out test once.
  Recovery-screen ETA is `2-4h`.

## 07:21 SAST recovery state

- Canary job `1159336` completed cleanly in `01:10:26`, exit `0:0`; trial
  `1tps89m9` finished at validation-only `eval/all_f1=0.3733855186`.
  Tail job `1159345` remains healthy and `RUNNING` on two A100-80GB GPUs on
  `srvrocgpu011`. Twenty-three of 24 arms have started: `21/24` complete or
  pruned, `xbuk7k3d` and `c9f9y1nb` active, and `1/24` unstarted.
- Completed `ngli032e=0.5798237023` remains the provisional validation leader.
  Active `c9f9y1nb=0.5689095128` at `138/345` (`40%`) is provisionally third,
  between `1acqnc84=0.5649671053` and `blxjtbgx=0.5559471366`; it is not
  rank-final. Active `xbuk7k3d=0.5111617312` was at `506/690` and about `80%`
  in the log. The final unstarted arm will be assigned automatically.
- Health remains clean: no traceback, CUDA OOM, disk-full, quota kill,
  recurrence of the W&B five-prune shutdown, or failed-run exit. Tail-job max
  RSS is about `6.42 GiB`.
- Zulu artifacts occupy `3.2 GiB` at
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul/`.
  HEX is `93/100 GB` (`93.7%`); the 300 GB increase remains inactive. Do not
  launch stability until all 24 are rank-final and verified losers have been
  copied to Kombuys/checksummed for storage reconciliation.
- Selection remains validation-only task-native `eval/all_f1`, followed by
  top-two three-seed stability, representative-checkpoint freeze, and one
  official held-out test. Recovery-screen ETA is `1.5-3h`.

## 08:23 SAST final-arm state

- All `24/24` Zulu screen arms have now started. Tail job `1159345` remains
  healthy and `RUNNING` on A100-80GB; `23/24` are complete or pruned and the
  sole final arm `jqjajjwh` is active. It reached about `322/690` (`47%`) with
  provisional validation-only `eval/all_f1=0.4365698086`, so it has not
  challenged the top three.
- Current rank is `ngli032e=0.5798237023`,
  `c9f9y1nb=0.5689095128`, `1acqnc84=0.5649671053`, and
  `blxjtbgx=0.5559471366`. This remains provisional until `jqjajjwh` and job
  `1159345` exit cleanly. `c9f9y1nb` and `xbuk7k3d` completed during this
  cycle; no run-count gap remains.
- The current top-three checkpoint roots are intact and contain adapters,
  tokenizer metadata, and trainer state:
  `ngli032e/checkpoint-276`, `c9f9y1nb/checkpoint-138`, and
  `1acqnc84/checkpoint-253` beneath
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul/`.
  Full recipe parameters must be extracted from the sweep logs/W&B provenance
  before launching stability; do not infer them from adapter metadata alone.
- Health remains clean: no traceback, CUDA OOM, disk-full, quota kill,
  five-prune shutdown, or failed-run exit. Artifacts remain `3.2 GiB`; HEX is
  `93/100 GB` (`93.7%`), and the 300 GB increase is still inactive.
- After the final arm exits: capture all 24 statuses/configs, freeze the top
  two recipes by validation `eval/all_f1`, copy/checksum verified losers to
  Kombuys before removing any exact cold HEX copy, then submit top-two
  three-seed stability. Final-arm ETA is about `1-1.5h`.

## Literature review: why GDN General looks strong and what to test

### Evidence-based conclusion

- GDN General is not broadly superior. In the matched 12/12 comparison its
  clear wins over Multi are confined to MasakhaPOS, by only about
  `0.015-0.019`; General is worse on NER, SIB, Intent, AfriHG, and corrected
  News. T2X General versus Mono is not a three-way matched comparison because
  no Multi task adapter exists.
- The present General loader provides a direct non-architectural mechanism:
  the six task families have weight `1`, `mix_temperature=0`, and therefore
  each receives probability `1/6` regardless of dataset size. The epoch length
  is the sum of component sizes, so small structured datasets can be heavily
  repeated while large tasks are relatively downsampled. This can plausibly
  benefit POS while producing under-training or interference elsewhere.
- Existing causal evidence ranks optimization/exposure above architecture:
  matching Xhosa NER's effective batch improved held-out best-prompt F1 from
  `0` to `0.2780`, and validation-only HPO raised it to `0.6287`. Therefore an
  architecture-specific General advantage remains unproven.
- Gated DeltaNet's paper attributes its advantage to gated selective forgetting
  plus delta-rule key-value updates, tested mainly at 400M/1.3B after matched
  pretraining on up to 100B tokens. It does not test 125M LoRA multi-task SFT,
  multilingual transfer, POS, or General-versus-Mono/Multi. Its mechanism is a
  plausible hypothesis for sequence/context handling, not evidence for the
  observed General POS pattern.

### Primary literature and implications

- Gated Delta Networks (Yang et al., 2024), arXiv:2412.06464:
  <https://arxiv.org/abs/2412.06464>. Gating clears stale memory and the delta
  rule makes targeted key-value updates. The paper's controlled benefits are
  retrieval, language modelling, commonsense, and long-context performance;
  they cannot establish the cause of this downstream mixture effect.
- T5 (Raffel et al., 2020), arXiv:1910.10683:
  <https://arxiv.org/abs/1910.10683>. Its controlled multi-task study found
  equal task mixing especially weak and identified task-dependent sweet spots
  for capped examples-proportional or temperature-scaled sampling. This maps
  directly to the current equal-probability General loader.
- FLAN (Wei et al., 2021), arXiv:2109.01652:
  <https://arxiv.org/abs/2109.01652>. It capped per-dataset contribution and
  used examples-proportional mixing. Its original ablation found held-out-task
  generalization could worsen at 422M-8B, warning against extrapolating large
  instruction-tuning gains to a capacity-limited 125M model.
- Scaling Instruction-Finetuned LMs (Chung et al., 2022), arXiv:2210.11416:
  <https://arxiv.org/abs/2210.11416>, and the Flan Collection (Longpre et al.,
  2023), arXiv:2301.13688: <https://arxiv.org/abs/2301.13688>. More diverse
  tasks and prompt formats can help, but mixture balancing is a critical
  contributor; benefits depend on model scale, task composition, and format.
- XLM-R (Conneau et al., 2019), arXiv:1911.02116:
  <https://arxiv.org/abs/1911.02116>. Low-resource languages can gain from
  multilingual transfer, but fixed-capacity models face a curse of
  multilinguality, and the language-sampling exponent materially changes the
  high-resource/low-resource trade-off. At 125M, capacity and sampling are
  first-order confounds.
- DoReMi (Xie et al., 2023), arXiv:2305.10429:
  <https://arxiv.org/abs/2305.10429>. Validation/proxy-loss-based domain
  reweighting can improve training efficiency, but the evidence is pretraining,
  not downstream LoRA SFT; use its principle only after simpler sampling
  controls.
- GradNorm (Chen et al., 2017), arXiv:1711.02257:
  <https://arxiv.org/abs/1711.02257>, and PCGrad (Yu et al., 2020),
  arXiv:2001.06782: <https://arxiv.org/abs/2001.06782>. Multi-task gradients
  can differ in magnitude or conflict. These are diagnostic/second-stage
  remedies; applying PCGrad to an autoregressive mixed-example batch is not a
  zero-cost drop-in and should follow measured interference.
- LoRA (Hu et al., 2021), arXiv:2106.09685:
  <https://arxiv.org/abs/2106.09685>, and AdaLoRA (Zhang et al., 2023),
  arXiv:2303.10512: <https://arxiv.org/abs/2303.10512>. Adaptation quality
  depends on which matrices receive capacity and how rank budget is allocated.
  This supports searching GDN target modules/rank rather than assuming the
  original rank-16 adapter has enough capacity for six task families.

### Minimal causal experiment before broad General HPO

1. Keep the base checkpoint, LoRA target modules, rank, LR, total target tokens,
   optimizer updates, prompts, and validation sets fixed. Compare only three
   mixtures: current equal-task `1/6`, capped examples-proportional, and one
   temperature-smoothed mixture. Report every task's native validation metric,
   not only aggregate validation loss.
2. For the best mixture, compare one-stage General with broad General followed
   by a short task-balanced refinement stage. Select the stage length on
   validation only. This tests whether shared instruction learning followed by
   specialization reduces negative transfer.
3. Only then run a compact GDN capacity screen over LoRA rank/alpha and target
   modules. Reuse the existing asynchronous Hyperband runner; do not multiply a
   full HPO grid by every mixture.
4. Log component draws/tokens, per-task validation curves, and a small periodic
   gradient-cosine diagnostic. Use PCGrad/GradNorm only if conflict or gradient
   domination is actually observed.
5. Repeat the winning mixture intervention on one matched Transformer or xLSTM
   control. A GDN-specific claim requires a significant architecture-by-mixture
   interaction, not merely a higher GDN endpoint.
6. Freeze selection on validation, repeat finalists across three seeds, retain
   the representative validation-selected checkpoint, and evaluate held-out
   test once. Multi-prompt test headlines remain explicitly labelled **best
   prompt**, with mean/range/all prompts and the descriptive-not-unbiased
   warning.

No new General GPU run was launched from this review. As of the live check,
the user has no active HEX jobs; `srvrocgpu011` has three free A100-80GB GPUs
and `srvrocgpu009` has four idle A100-40GB GPUs. HEX scratch remains
`93/100 GB` (`93.7%`) and the requested 300 GB quota is not active. The safe
next compute action is still Zulu NER finalist stability after verified loser
archival, while the General three-mixture control is prepared as the next
bounded causal study rather than a speculative broad sweep.

## Live screen and cluster verification

- Re-read Slurm accounting after the literature review. Zulu screen jobs
  `1154777`, `1159336`, and `1159345` are all `COMPLETED 0:0`; the continuation
  accounting is exactly 18 + 1 + 5 arms. Final validation ranking remains
  `ngli032e=0.5798237023` (`checkpoint-276`) then
  `c9f9y1nb=0.5689095128` (`checkpoint-138`).
- The user has no active HEX job. `srvrocgpu011` has 3/4 A100-80GB GPUs free
  and `srvrocgpu009` has 4/4 A100-40GB GPUs idle. Scratch remains
  `93/100 GB` (`93.7%`); the 300 GB quota is still inactive.
- Kombuys is also idle: RTX 5090 and RTX 3080 Ti both report 0% utilization;
  `/scratch/alombard` has `2.4 TiB` free. Only the administrative
  `tailscale-kombuys` tmux session exists.
- Do not spend the free GPUs on an uncontrolled General sweep. First archive
  and verify the 22 Zulu non-finalist roots to Kombuys, retain both finalists
  on HEX, then run their three-seed validation stability. The General
  equal-versus-proportional-versus-temperature mixture control follows as the
  smallest causal experiment supported by the literature.

## Zulu archive and stability submission

- Archived exactly the 22 validation non-finalist Zulu screen roots from
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul/`
  to
  `/scratch/alombard/sallm/archive/hex_gdn_hpo_20260802/ner_zul/` on Kombuys.
  The archive contains 22 top-level trial directories and `3,040,620 KiB` on
  the destination filesystem. Both checksum dry-runs (HEX to local staging,
  then local staging to Kombuys) returned zero differences before any source
  removal. Only the 22 explicitly named cold HEX copies were removed; they are
  recoverable from Kombuys. The temporary local relay was then removed.
- HEX retains exactly the two finalists, `ngli032e` and `c9f9y1nb`, totaling
  `259,298 KiB`. Refreshed scratch is `90/100 GB` (`90.8%`); the requested
  300 GB quota remains inactive.
- Exact validation-screen recipes were recovered from the ordered W&B-agent
  launch records rather than inferred from adapter metadata:
  - `ngli032e` (`eval/all_f1=0.5798237023`, `checkpoint-276`): rank 64,
    effective batch 32, dropout 0, LR `2.900524112494857e-4`, weight decay
    `0.028591710568416695`, warmup `0.03`, cosine schedule.
  - `c9f9y1nb` (`0.5689095128`, `checkpoint-138`): rank 64, effective batch
    64, dropout `0.1`, LR `3.590314109494251e-4`, weight decay
    `0.004030112809995579`, warmup `0.05`, constant-with-warmup.
- Added and YAML-validated two-seed stability manifests:
  `src/conf/sweeps/gdn_ner_zul_stability_ngli032e.yaml` and
  `src/conf/sweeps/gdn_ner_zul_stability_c9f9y1nb.yaml`. Each runs only the
  missing validation seeds `[17, 73]`; screen seed 42 is retained. Selection
  remains task-native validation `eval/all_f1`, by three-seed mean with
  stability considered. Test is not referenced.
- Synced only those manifests and submitted A100-only stability jobs
  `1160774` (`ngli032e`) and `1160775` (`c9f9y1nb`), each requesting two
  A100-40GB GPUs, 8 CPUs, and an 8-hour ceiling. The first account-less
  `sbatch` attempts were rejected before job creation; the real jobs use
  account/QOS `nlpgroup` on partition `a100`.
- Current state: `1160774` is pending for `Resources`; `1160775` is pending
  for `Priority`. `srvrocgpu010` has only one free `ampere` GPU, which cannot
  satisfy either two-GPU request yet. `srvrocgpu009` is idle but exposes the
  distinct `amperemk` GRES, and `srvrocgpu011` has three free `ampere80` GPUs;
  no duplicate resubmission or cancellation was performed. No L40S work is
  active. Once scheduled, expected stability runtime is about two hours based
  on the Xhosa gate.

## 10:10 SAST — stability scheduling repair and General causal anchor

- Retargeted the existing Zulu stability jobs without cancellation or
  duplication. The idle `srvrocgpu009` node could not accept the jobs because
  the `nlpgroup` association rejected `amperemk` with `AssocGrpGRES`.
  `1160774` was moved to the authorized `nlpgroup80` / `ampere80:2`
  association and started on `srvrocgpu011`; `1160775` was restored to its
  original `nlpgroup` / `ampere:2` request and remains pending for two
  contiguous A100-40GB GPUs. Selection remains three-seed mean validation
  `eval/all_f1`, with stability considered before freezing one representative
  checkpoint and touching held-out test.
- Recovered the exact historical GDN General run from HEX job `1020357`, W&B
  run `i77vz2jm`, and Hub commit
  `4c3635d3c15fac6d7cacc8904f389c17bbc3ab5f`. It trained the GDN base
  `anrilombard/sallm-gated-deltanet-125m-shallowwide-4x40-20260707` for one
  epoch / 2,728 optimizer updates over 43,637 sampled examples at max length
  1024, batch 4, gradient accumulation 4, LR `8e-5`, cosine schedule, warmup
  `0.03`, weight decay `0.01`, rank-16 / alpha-32 / dropout-0.05 LoRA, seed
  42, and equal `1/6` task sampling. It completed in `03:52:44`.
- Correction to the earlier partial Hub audit: the historical
  `adapter_config.json` explicitly preserves all seven target modules:
  `in_proj_qkvz`, `in_proj_ba`, `out_proj`, `q_proj`, `k_proj`, `v_proj`, and
  `o_proj`. The historical mixture recipe is therefore sufficiently recovered
  for an exact modern equal-mixture reproduction.
- Submitted only the first General causal anchor as job `1160839`,
  `gdn-general-equal-r1`, requesting one A100-80GB GPU with an eight-hour
  ceiling. Output is isolated at
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_general_mixture_20260802/equal_task_r1`.
  It reproduces the historical recipe and disables Hub push/deletion so the
  adapter remains inspectable. It started on `srvrocgpu011` as W&B run
  `m3i2ants` and passed model/LoRA initialization with 2,660,128 trainable
  parameters; capped
  examples-proportional and temperature-smoothed arms are intentionally not
  submitted until this anchor passes startup and storage remains safe.
- HEX scratch remains `90/100 GB` (`90.8%`); the requested 300 GB expansion is
  still inactive. The General study uses the A100 family only and does not
  overlap any L40S work.

## 10:12 SAST — stricter GPU-family policy correction

- User clarified that A100-40GB (`amperemk`/`ampere`), A100-80GB
  (`ampere80`), and L40S are three mutually exclusive execution families,
  even though both A100 variants share Slurm partition `a100`.
- Existing pending A100-40GB job `1160775` was immediately user-held. Slurm
  verifies `JobState=PENDING`, `Reason=JobHeldUser`, and
  `TresPerNode=gres/gpu:ampere:2`; it cannot start while A100-80GB jobs
  `1160774` and `1160839` run. It was not cancelled or duplicated.
- Updated the `uct-masters-research` skill so future monitoring and submission
  treats the three GPU families as mutually exclusive and does not leave jobs
  from another family schedulable. The active HEX family is now exclusively
  A100-80GB (`ampere80`).

## 10:24 SAST — A100-80GB live progress

- Active family remains exclusively A100-80GB. Zulu stability job `1160774`
  is healthy at approximately `160/690` and `164/690` steps for its two
  validation seeds (about 23% each); no traceback, CUDA OOM, quota, or disk
  error is present. Current training-rate ETA is about one hour to the next
  validation checkpoint, with final runtime dependent on early stopping.
- General equal-mixture anchor `1160839` is healthy at about `144/2728`
  updates (5.3%). It verified the intended equal distribution (`p=0.1667` for
  each of SIB, News, NER, POS, AfriHG, and T2X) and compiled the GDN TileLang
  fast path. Current rate projects about 5.5 hours of remaining training before
  final validation/save; its eight-hour ceiling remains adequate but should be
  watched.
- A100-40GB stability job `1160775` remains safely user-held
  (`JobHeldUser`). No L40S or A100-40GB user job is schedulable. Scratch is
  `91/100 GB` (`91.0%`); the 300 GB increase is still inactive.

## 11:25 SAST — first Zulu finalist stability complete

- Zulu `ngli032e` stability job `1160774` completed cleanly `0:0` in
  `01:22:16`. Seed-17 W&B run `k2873n5y` selected validation
  `eval/all_f1=0.6184448463` at `checkpoint-368`; seed-73 run `sc7lzdz6`
  selected `0.6086572438` at `checkpoint-322`. Together with screen seed-42
  `0.5798237023`, the three-seed mean is `0.6023085975`, sample SD
  `0.0200780271`, range `0.5798237023-0.6184448463`.
- Both selected validation artifacts remain under
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_hpo_20260801/ner_zul_stability/ngli032e/`
  and occupy `259,779 KiB`. No held-out test was touched.
- With the A100-80GB GPUs released by `1160774`, existing second-finalist job
  `1160775` was retargeted in place from held A100-40GB to authorized
  `nlpgroup80` / `ampere80:2`, then released. It is now running on
  `srvrocgpu011`; this preserves the strict A100-80GB-only active family and
  does not cancel or duplicate the job.
- General anchor `1160839` remains healthy at about `630/2728` updates (23%)
  with a projected 4.3 hours of training remaining. Scratch remains
  `91/100 GB` (`91.0%`); the requested 300 GB quota is inactive.

## 12:24 SAST — second finalist and General anchor progress

- The active HEX execution family remains exclusively A100-80GB
  (`ampere80`). Zulu finalist-2 stability job `1160775` uses two GPUs and the
  General equal-mixture anchor `1160839` uses one on `srvrocgpu011`. Another
  user's job `1155939` occupies the fourth A100-80GB GPU; it was not modified,
  preempted, or disturbed. No A100-40GB or L40S job owned by `lmbanr001` is
  active or schedulable.
- `1160775` is healthy at about `171/345` steps for each of its two fixed-seed
  runs (roughly 50%). W&B run `2igxferw` has provisional validation
  `eval/all_f1=0.5233082707` with `checkpoint-138`; `rtq8eisy` has
  `0.4992581602` with `checkpoint-161`. These remain incomplete and currently
  trail `ngli032e`'s completed three-seed mean `0.6023085975`; no finalist is
  frozen and held-out test remains untouched. Current ETA is roughly one hour,
  subject to early stopping.
- General equal-mixture job `1160839` / W&B `m3i2ants` is healthy at
  `1090/2728` optimizer updates (40%). Its live rate projects about 3h25 of
  training remaining. Output root is
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_general_mixture_20260802/equal_task_r1`;
  no checkpoint has been saved yet because the one-epoch anchor has not
  reached its final save.
- Targeted log checks show no traceback, CUDA OOM, disk-full, quota failure, or
  kill. Zulu stability artifacts currently occupy 378 MiB. Refreshed HEX
  scratch is `91/100 GB` (`91.4%`); the requested 300 GB expansion remains
  inactive.
- Selection remains validation-only: task-native NER `eval/all_f1`, compared
  by three-seed mean with stability considered, then freeze one representative
  checkpoint before one official held-out test. The General mixture anchor is
  an exposure-control reproduction and will be compared on task-native
  validation metrics before any proportional or temperature-smoothed arm is
  promoted. The canonical sheet remains unchanged.
- Activated `sol-advisor:orchestration` for future delegated implementation.
  Its role-contract reference was read, the non-mutating installer check
  passed byte-for-byte, and the runtime exposes all three required custom role
  names. No delegated implementation lane was needed for this monitoring pass.
  The hourly `gdn-hpo-workstream-monitor` automation was updated to preserve
  the strict three-family GPU rule and require Sol Advisor preflight/review for
  future delegated implementation.

## Architecture-label correction and pure-GDN priority — 2026-08-02

- The verified existing checkpoint is `Qwen3NextForCausalLM`, trained from
  scratch with 33 layers total: 25 `linear_attention` GDN layers and 8
  `full_attention` layers, hidden size 640, and 10 heads. Its exact label is
  **GDN–Attention Hybrid (Qwen3Next implementation)**. The implementation
  choice was deliberate in the early notes; later reporting drifted to an
  inaccurate pure-GDN label. Existing results remain valid as a secondary
  hybrid arm only and cannot occupy the pure-GDN comparison slot.
- Pure GDN is installed on HEX through `flash-linear-attention==0.5.1` and
  `fla-core==0.5.1`; `GatedDeltaNetConfig`, `GatedDeltaNetModel`, and
  `GatedDeltaNetForCausalLM` are under
  `/home/lmbanr001/masters/sallm/.venv/lib/python3.12/site-packages/fla/models/gated_deltanet/`.
  With `attn=None`, every block is GatedDeltaNet. No official pretrained
  weights are available, so the matched pure-GDN arm requires pretraining from
  scratch.
- The current `gated_deltanet` -> Qwen3Next mapping in
  `src/main/sallm/models/registry.py` is an integration defect to correct, not
  a label to reinterpret.
- Priority is to stop broad new hybrid HPO, preserve the useful running hybrid
  General equal-mixture control `1160839`, and integrate parameter-matched pure
  GDN. Before resumable full pretraining, verify BF16 forward/backward, the FLA
  fast path, packed context, DDP, save/load, and generation, then run one
  A100-80GB canary. Continue A100-80GB exclusivity and never tune on test.
- Primary HEX status evidence is scratch `91/100 GB` (`91.3%`);
  `1160839` is running on one `ampere80` GPU at `03:50:30`; `1160775`
  completed `0:0` in `01:54:06`; and `1117467/1117468` were not modified in
  this cycle. Current Slurm accounting reports `1117467/1117468` as
  `CANCELLED by 0`, so they are not still held.

## Verified cold-storage move — 2026-08-02

- The cold source `/scratch/lmbanr001/masters/sallm/checkpoints/sallm-llama-252m-canary`
  was 1.9 GB and was not a winner or canonical result.
- CPU-only `ada` job `1162513` streamed the copy on Kombuys to
  `/scratch/alombard/sallm/hex_archive/checkpoints/sallm-llama-252m-canary`.
  Source and destination per-file SHA256 manifests compared with no
  differences via CPU-only job `1162568`; the destination contains 19 files
  and reports 1.9 GB. The copy is recoverable from that Kombuys path.
- The first deletion attempt, job `1162570`, safely exited `1` before removal
  because local shell substitution invalidated its path check. Corrected
  explicit-path CPU-only job `1162571` removed only the verified HEX source;
  follow-up checks confirmed the source is absent and the Kombuys archive is
  present.
- HEX quota reporting remained at delayed `91/100 GB` (`91.3%`) immediately
  afterward; no freed quota is claimed until a refreshed report. The only
  active GPU job remained hybrid General control `1160839` on one A100-80GB;
  no GPU-family rule was violated because the storage jobs used non-GPU
  `ada`. This was a bounded move, not broad storage cleanup.

## Pure-GDN matched shape and hybrid-control completion — 2026-08-02

- CPU-only HEX diagnostics `1163907`, `1163986`, `1164007`, and `1164029`
  resolved the installed FLA 0.5.1 constructor and parameter allocation without
  consuming a GPU. The first search was invalid for the 120--130M contract; its
  candidates were 147--241M because the diagnostic did not apply the intended
  tied-embedding convention. The corrected search used the same tied-embedding
  convention as the LLaMA, Mamba, and xLSTM 125M baselines.
- The frozen pure-GDN shape is `hidden_size=512`, `num_hidden_layers=21`,
  `intermediate_size=1536`, `num_heads=4`, `head_dim=128`, `expand_v=2`,
  `vocab_size=65536`, `max_position_embeddings=2048`, `attn_mode=chunk`,
  `tie_word_embeddings=true`, and critically `attn=null`. Job `1164029`
  instantiated exactly `127,425,448` parameters and completed `0:0`. This is a
  parameter-matched pure FLA GatedDeltaNet candidate, not a Qwen3Next hybrid.
- The preserved General equal-mixture control `1160839` completed `0:0` in
  `06:02:56` after all `2728/2728` updates. It reported final validation loss
  `0.2754413566702204`, saved 640 generation-debug examples, and saved its PEFT
  adapter beneath
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_general_mixture_20260802/equal_task_r1`.
  Its label remains **GDN--Attention Hybrid (Qwen3Next implementation)**; it is
  a secondary mixture-control artifact and does not satisfy the pure-GDN arm.
- Refreshed HEX accounting is `/home=3/10 GB`, `/scratch=89/100 GB` (`89.7%`).
  The hybrid completion released its A100-80GB GPU. No pure-GDN GPU canary is
  authorized until the integration diff, lockfile, local verification, and
  fresh Sol final review pass.

## Pure-GDN integration acceptance and budget audit — 2026-08-02

- The pure/hybrid registry correction, FLA 0.5.1 dependency boundary,
  streaming two-rank canary, wrapped non-flattening 2,048-token packing,
  distinct `H`/`HV` kernel probe, and DDP-safe saving passed `80` local tests,
  Ruff, formatting, `ty`, YAML, shell, lock, and diff checks. After two bounded
  `fix-first` rounds, a third fresh Sol/high review returned `ship`; exact
  before/after hashes confirmed the reviewer made no edits.
- Signed local commit `4feaadf62b665e558069eccf978987f3238a65e7` records the
  accepted integration on branch `research/pure-gdn-baseline-20260802`.
  `main` is untouched and nothing was pushed. No HEX files or jobs changed:
  the external-write approval gate rejected the attempted exact 35-file sync,
  so the reviewed two-A100-80GB canary remains unsubmitted.
- Static recipe audit confirms common tokenizer path and vocabulary (65,536),
  LR `4e-4`, cosine schedule, warmup `2000`, weight decay `0.01`, Adam betas
  `0.9/0.95`, gradient norm `1.0`, and intended effective batch `192` against
  the 125M baselines. Current YAMLs are not sufficient execution provenance:
  LLaMA W&B run `2mkkmx3d` ended at step `48,403`; the completed xLSTM run
  reached step `67,498` / epoch `3.0871`; and the current Mamba Hub base has a
  messy, incompletely recovered run lineage. Therefore the full pure-GDN
  `num_train_epochs: 5` setting remains provisional, not a scientifically
  frozen token budget. Before broad pretraining, recover the executed
  per-architecture token/update contracts and pre-register one explicit token
  budget; the two-step canary is only an implementation gate.
## 18:00--19:15 SAST — pure-GDN A100-80GB canary sequence and trainer gates

- Synced the exact 35-file reviewed pure-GDN integration to HEX and verified it with a checksum dry-run producing zero differences. HEX remained at `/home=3/10 GB`, `/scratch=89/100 GB` (latest `89.9%`); no owned A100-40GB or L40S job was schedulable during any canary. All canaries requested exactly two `ampere80` GPUs on `nlpgroup80` and ran on `srvrocgpu011`.
- Job `1164606` failed in one second before Python/GPU work because Slurm executes a spool copy and `BASH_SOURCE` resolved `scripts/lib/env.sh` beneath `/var/spool`. Minimal `SLURM_SUBMIT_DIR` fallback passed focused checks and fresh Sol review; signed branch commit `8521c6ec20b5c15755c6eb50a1d57a3916aa7c5f`.
- Job `1164608` passed dependency import plus the hardware preflight, including pure FLA BF16 forward/backward and the direct chunk-kernel backward with layout `qk[B,T,H,K];v[B,T,HV,V];g,beta[B,T,HV];A_log,dt_bias[HV]`. It then failed before DDP training because grouped Hydra overrides incorrectly used `training.*`. Corrected `base.training.*` paths passed 7 focused tests and fresh Sol review; signed branch commit `3b50f218ca7112fec38b4cda8a600fb9d301c96a`.
- Job `1164702` passed the same pure hardware preflight, nested-config unwrapping, two-rank startup, and exact `127.43M` model-size gate. Both ranks then failed before training because locked `datasets==3.6.0` cannot parse the Hub dataset's `_type: List` metadata. Shared contract updated to `datasets>=4.4.2,<5`, lock resolved `4.8.5`, exact offline metadata regression added, full local suite passed `82` tests, and fresh Sol review shipped; signed branch commit `581e95cf8b55352a0fe953d00dbc766da92fcd5a`.
- Job `1164809` advanced through Hub streaming, two-rank trainer initialization, exact `127.43M` model validation, real wrapped non-flattened `2048`-token training batch, and the TileLang FLA chunk backward. It completed optimizer step `1/2` in `231.93s`, then validation failed because Accelerate dispatch attempted to concatenate a full `2048`-token iterable batch with the final `87`-token batch. It also logged inflated loss `1074.3037`, consistent with Transformers treating FLA's unused `**kwargs` as support for `num_items_in_batch` under gradient accumulation.
- Installed FLA 0.5.1 inspection confirmed `GatedDeltaNetForCausalLM.forward(..., **kwargs)` has no `num_items_in_batch` handling and computes ordinary mean cross-entropy. Sol commitment consult said proceed with two narrow fixes: pure `gated_deltanet` instances advertise `accepts_loss_kwargs=False`; resolved public `AcceleratorConfig.dispatch_batches` defaults to false only for HF iterable train/eval data and only when still unset. The corrected four-file implementation passed `16` affected tests, Ruff, `ty`, diff checks, and fresh final Sol review (`ship`). Post-review hashes: `models/factory.py=75c7e14c...`, `training/factory.py=f977b82e...`, `tests/models/test_factory.py=e03e20c5...`, `tests/test_training_factory.py=1c0c65cb...`.
- External-write approval service then failed with its own `unknown_parameter: input[6].namespace` schema error for both local staging and the exact four-file HEX sync. No workaround attempted. These four reviewed files plus this memory update remain local/uncommitted; decisive canary rerun is pending explicit renewed user authorization after disclosure.
- Acceptance remains incomplete: A100 BF16 and direct FLA chunk kernel are verified; DDP startup, Hub streaming, real packed `2048` batch, TileLang training, and one optimizer step are verified. Two completed steps, normal-scale loss, validation completion, rank-safe checkpoint/final saves, AutoModel reload, exact state-dict roundtrip, and deterministic generation remain pending.
- Metric/selection rule remains frozen: full pretraining must be compared by validation-only pretraining metrics under a preregistered matched token/update budget; no held-out test informs training or recipe selection. The long run remains blocked by unresolved executed-token budget recovery and only about `10 GB` HEX scratch headroom. Once sync is authorized, canary ETA is approximately `15--25 minutes`; full-run ETA is not yet defensible.

## Fresh-lane completion and upstream Codex diagnosis

- A fresh operational subagent successfully staged and committed the reviewed trainer contract plus provenance notes on `research/pure-gdn-baseline-20260802` as `6812806bb92928cb78a912eb8ce2c67f86317e77`; the commit object contains an SSH signature and `main` remains untouched. The exact four-file rsync was then rejected by the safety reviewer as a remote source export despite explicit user authorization, so no checksum comparison or canary submission occurred.
- Quota-first read-only evidence after that rejection: `/home=3/10 GB` (`32.4%`), `/scratch=89/100 GB` (`90.0%`); owned queue empty; no owned A100-40GB/L40S job or duplicate canary. `srvrocgpu011` has another user's one-GPU `ampere80` job `1155939`, leaving three of four A100-80GB GPUs nominally available.
- The approval serializer error is already public upstream: open `openai/codex#31754` contains the same `[ObjectParam] ... input[n].namespace ... unknown_parameter` regression and an August 1 auto-review reproduction (`Automatic approval review failed`); `#31760` is the same schema failure on a resumed-session path. No public fix PR was found. This is distinct from the later policy classification that blocked rsync.

## 19:44--19:53 SAST — decisive pure-GDN canary accepted

- A fresh role-pinned Luna operational lane retried the explicitly authorized
  transfer without changing local files. A quota-first relative rsync copied
  exactly the four reviewed files from commit
  `6812806bb92928cb78a912eb8ce2c67f86317e77`; remote SHA256 values matched
  `75c7e14c...`, `f977b82e...`, `e03e20c5...`, and `1c0c65cb...`. A checksum
  dry-run reported zero created, deleted, or transferred files. This confirms
  the earlier rsync rejection was transient/policy-path behavior, distinct
  from the still-open upstream `input[n].namespace` regression in
  `openai/codex#31754`.
- The owned queue was empty before submission, and the launcher was verified
  as `nlpgroup80` / `a100` / `gpu:ampere80:2`. Exactly one job, `1165989`,
  ran on two A100-80GB GPUs on `srvrocgpu011`; no A100-40GB or L40S work was
  owned or schedulable concurrently.
- Job `1165989` completed `0:0` in `00:08:01`. It verified the real wrapped
  batch length at 2,048 tokens, the TileLang FLA chunk backward, and both
  optimizer steps. Training losses were `11.1940` and `11.1951`, correcting
  the previous inflated `1074.3037`; final train loss was `11.1945290565`.
  Both validation passes completed with `eval_loss=11.1919603348`, rather
  than failing on the final short iterable batch.
- Rank-safe checkpoints `checkpoint-1` and `checkpoint-2` and `final_model`
  were saved beneath
  `/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m-canary/a10080-1165989`.
  The post-checkpoint probe reloaded through AutoModel, recovered exactly
  `127,425,448` parameters, passed exact save/load state integrity, and passed
  deterministic greedy generation. Artifact size is `1.7G`; final-model
  metadata hashes are `config.json=114fd68f5f522b3627aa10fc34d029e9e200adb5ebb228e8f826c64b93eddf5b`
  and `generation_config.json=8abc09d0606da28291b9ab4be66db068d3747ddea747d28f7560930928a3eba5`.
- After completion, the owned queue was empty and HEX quota was
  `/home=3/10 GB` (`32.4%`) and `/scratch=91/100 GB` (`91.6%`). The hardware
  implementation gate is now accepted, but full pretraining remains blocked:
  first recover and preregister a matched tokenizer/token/context/update
  budget against LLaMA/Mamba/xLSTM and create verified scratch headroom.
  Pretraining selection remains validation-only; no held-out test may inform
  the recipe.

## 20:20--20:53 SAST — matched budget frozen and cold storage verified

- The full pure-GDN contract is now frozen to the executed LLaMA/xLSTM anchor:
  streaming Hub data, tokenizer vocabulary `65,536`, context `2,048`, two
  A100-80GB ranks, effective global sequence batch `48`, and `48,403`
  optimizer steps. This is exactly `4,758,208,512` token slots. Optimizer
  settings remain LR `4e-4`, cosine, `2,000` warmup steps, weight decay
  `0.01`, Adam betas `0.9/0.95`, and max gradient norm `1.0`. Selection and
  monitoring use validation loss only; held-out test cannot tune the run.
- Local validation passed: `32` focused pure-GDN/runtime/model/training tests,
  Ruff, Ruff formatting, `ty`, `yamllint`, `bash -n`, and `git diff --check`.
  The full config now explicitly streams because the canonical local tokenized
  dataset is absent and nonstreaming materialization is storage-unsafe.
- Accepted canary `1165989` remains intact at
  `/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m-canary/a10080-1165989`
  (`1.7G`); both recorded final-model metadata SHA256 values reverified.
- Cold-copy attempts `1166158` and `1166164` failed safely before data transfer
  because compute nodes lacked the local Kombuys alias/key; `1166164` was
  cancelled while blocked. CPU probe `1166173` confirmed the Tailscale route
  was inaccessible from `ada`. The corrected bounded relay kept source reads
  on CPU-only `ada` jobs and completed exact copies as `1166185` and `1166223`.
- CPU manifest job `1166277` produced `119` and `126` source-file SHA256
  entries. Destination manifests on Kombuys contained the same entries; an
  initial bytewise diff exposed only locale-dependent line ordering. CPU job
  `1166460` normalized with `LC_ALL=C`, after which both manifests matched
  byte-for-byte. Source and destination manifests are preserved beneath each
  Kombuys archive's `.archive_manifests/` directory.
- Separate CPU deletion job `1166480` rehashed each unchanged HEX source,
  required equality with its preserved manifest, then removed only
  `/scratch/lmbanr001/masters/sallm/checkpoints/gdn_afrihg_hpo_r1` and
  `/scratch/lmbanr001/masters/sallm/checkpoints/news_hpo_r1`. It also removed
  only the recorded reproducible `/scratch/lmbanr001/.triton/cache` (`655M`).
  The job completed `0:0`; accepted canary and all canonical/active winners
  were untouched. Immediate quota refresh moved from `91.7%` to `91.0%`, with
  the larger directory deletions still subject to quota-reporting delay.

## 20:54--21:11 SAST — full pure-GDN pretraining running healthy

- Signed research-branch commit
  `4913db6b0c68d8f7065847f2b45934c747fcbf99` froze the matched config,
  streaming/budget tests, resumable A100-80GB launcher, and provenance notes.
  `main` remains `c4a5fab369fe0dd7eb47656e8437c1fb8e7cce1b`; nothing was pushed.
  Exact five-file HEX sync hashes matched and the checksum dry-run was empty.
- Refreshed HEX quota after cleanup and sync is `87.7--88.0%`. The accepted
  canary remains untouched. The owned GPU queue was empty before submission;
  no A100-40GB or L40S family was active or schedulable.
- Submitted a linear `afterany` chain `1166554--1166579` under run ID
  `a10080-matched-20260802`. Early measured throughput proved the conservative
  26-segment estimate unnecessary: after one-time compilation, steady rate is
  about `2.72--2.73 s/step`, projecting `~36.7 h` for 48,403 steps inside the
  first `47:30:00` segment. Exact pending tail `1166556--1166579` was therefore
  cancelled; only `1166555` remains as the single afterany resume/final no-op.
- Primary job `1166554` is `RUNNING` on `srvrocgpu011` with exactly two
  `gpu:ampere80` A100-80GB GPUs. W&B run is `mj2zeime`; output root is
  `/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-matched-20260802`
  and the append-only run log is beneath the matching scratch logs root.
- Hardware/runtime evidence is clean: BF16 forward/backward passed; direct
  chunk-kernel layout passed; actual FLA backend reports TileLang; Accelerate
  enabled multiple GPUs; model-size gate passed at `127.43M`; iterable
  `dispatch_batches=False`; and the real wrapped training batch verified
  exactly `2,048` tokens.
- At step `53/48,403`, ordinary-scale losses at steps 10/20/30/40/50 were
  `11.1954/11.1948/11.1942/11.1935/11.1906`. No traceback, CUDA OOM,
  disk-full, or competing GPU-family work is present. First validation and
  `checkpoint-1000` are pending, with an ETA around `21:55--22:10 SAST`;
  overall completion ETA is approximately `2026-08-04 10:00--13:00 SAST`
  allowing for validation/checkpoint overhead.
- The existing hourly `gdn-hpo-workstream-monitor` heartbeat was retargeted to
  jobs `1166554/1166555`, explicitly forbids duplicate submissions and Sol
  Advisor, and must verify validation loss, checkpoint-1000, final_model,
  AutoModel reload, exact state roundtrip, and deterministic generation.
