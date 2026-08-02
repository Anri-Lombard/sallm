# Mamba Root-Cause Rescue Experiment Checklist

Created: 2026-05-21 20:30 SAST.

Purpose: track the decoder-only experiments we will run one by one to determine
which Mamba weaknesses are fixable through formulation, decoding, scoring,
checkpoint selection, or base-model recipe improvements.

Status legend:

- `[ ]` not started
- `[~]` active/running
- `[x]` complete
- `[!]` blocked or failed
- `[?]` needs user/supervisor decision

Important rule: all experiments in this checklist must remain decoder-only.
Allowed interventions include prompt/formulation changes, constrained decoding,
teacher-forced decoder scoring, checkpoint selection, and base-recipe changes.
Do not switch to encoder models, encoder-decoder models, CRFs, classifier heads,
or external taggers for these rescue gates.

## Defensibility Requirements

Before any result is used to argue that "Mamba cannot do this task" or that
"the gap is architectural", it must pass these checks.

### Matched LLaMA Control

Run a LLaMA-style transformer control whenever the formulation, decoding, or
scoring method changes.

Every experiment needs two separate comparisons:

- **Mamba progress check:** compare against the current best Mamba baseline for
  the same task. A "rescue" only counts as a Mamba improvement if it beats the
  relevant previous Mamba recipe under the same official or diagnostic
  evaluation conditions.
- **Architecture fairness check:** compare against the matched LLaMA-style
  transformer control where feasible, so we can separate "Mamba improved" from
  "Mamba is competitive with the transformer baseline".
- **Shared-method check:** if the same decoder-only change improves both
  Mamba and LLaMA, record it as a general protocol or recipe improvement and
  carry it forward for the later optimized LLaMA rerun where compatible.

Interpretation rules:

- If LLaMA works and Mamba fails under the same decoder-only formulation, the
  issue is specific to Mamba's implementation, configuration, base quality,
  fine-tuning recipe, or architecture.
- If both LLaMA and Mamba fail, the formulation/evaluation method is likely
  flawed or too hard as posed.
- If both improve, the method is a general decoder-only rescue and must be
  reported as a matched-method comparison rather than a Mamba-only fix.
- If Mamba improves more than LLaMA, the method may be a Mamba-specific rescue,
  but still needs official test confirmation before final claims.
- If the method improves over the old Mamba result but LLaMA improves too, the
  final story should report both facts: Mamba was under-optimized, and the
  protocol itself may also improve the transformer baseline.

### Cross-Architecture Carry-Forward Rule

When an experiment produces a general improvement across both Mamba and LLaMA,
record it explicitly rather than discarding it as "not Mamba-specific".
This applies in both directions: Mamba-focused debugging may reveal a cleaner
decoder-only protocol that should later be applied to LLaMA, and LLaMA controls
may reveal methods that are useful even if they do not rescue Mamba.

For every improved method, note:

- whether the gain appears on Mamba only, LLaMA only, or both;
- whether it changes the task formulation, decoding, checkpoint selection,
  scoring, metric hygiene, or training recipe;
- whether it should be carried into the final optimized Mamba run, the final
  optimized LLaMA rerun, or both;
- whether the comparison remains fair after carrying it forward.
- if it is `Shared`, add it to the future LLaMA carry-forward backlog so it is
  not lost while the immediate work is still focused on rescuing Mamba.
- do this even when the current decision is Mamba-focused; a shared
  improvement is future LLaMA work, not Mamba-only evidence.

Every experiment decision block should therefore include a short
carry-forward tag:

- `Mamba-only`: candidate Mamba rescue; requires clear labeling in final
  comparison.
- `LLaMA-only`: useful negative control; may indicate a method that transformers
  can exploit but Mamba cannot under the current base/recipe.
- `Shared`: general decoder-only protocol improvement; rerun or apply to both
  final Mamba and final LLaMA recipes where compatible.
- `Negative`: no carry-forward unless it exposes a root cause or hygiene issue.

Interpretation rule:

- Mamba-specific gains help answer whether Mamba was under-optimized.
- Shared Mamba/LLaMA gains improve the final experimental protocol and should
  be applied to LLaMA later if they are legitimate, decoder-only, and
  comparable.
- LLaMA-only gains are still important because they may show that a method is
  transformer-compatible but not sufficient for Mamba, which strengthens the
  root-cause story.

### Shared Improvement Backlog

Use this section as the parking lot for changes discovered during Mamba rescue
work that may also improve LLaMA. Do not lose these just because the current
goal is to make Mamba defensible.

Current confirmed entries:

- None yet. C1/C1b currently labels AfriHG beam5/length-penalty as
  `Mamba-only`, because it improved Mamba but degraded/overgenerated LLaMA
  under the matched control.
- B2 placeholder/reinsertion is currently `LLaMA-only`/`Negative` for Mamba,
  because it improved the LLaMA diagnostic but collapsed Mamba placeholder
  production.
- E2 cache/batch-size handling is Mamba evaluation hygiene, not a shared
  decoder protocol improvement, unless a later matched LLaMA check shows the
  same sensitivity.

Add future entries here with:

- method or recipe change;
- tasks tested;
- Mamba effect;
- LLaMA effect;
- carry-forward tag: `Mamba-only`, `LLaMA-only`, `Shared`, or `Negative`;
- whether it must be applied to the final optimized LLaMA rerun.

### Final Downstream Confirmation Gate

After this checklist produces a defensible Mamba recipe, do not treat the
diagnostic/rescue results as the final architecture comparison by themselves.
Run or consolidate the downstream suite so the final comparison is:

- best defensible Mamba recipe versus best defensible LLaMA recipe;
- matched train/validation/test splits;
- matched prompt/task formulations where possible;
- matched metric scripts and reporting hygiene;
- matched decoding/result-selection rules unless a model-specific exception is
  explicitly justified;
- shared decoder-only improvements applied to both architectures where
  compatible.

Do not compare a rescued/optimized Mamba recipe against a stale LLaMA row that
does not include shared improvements discovered during Mamba debugging. If a
change helps both architectures, rerun or explicitly align the LLaMA recipe
before using the result for final downstream claims.

For every final downstream result, record three comparisons:

- new Mamba versus previous best Mamba;
- new Mamba versus matched LLaMA;
- whether any observed gain came from a Mamba-specific fix or from a shared
  protocol improvement that should also be applied to LLaMA.

### Configuration Parity

For Mamba-vs-LLaMA comparisons, record and match where possible:

- tokenizer and special tokens;
- prompt/template;
- train/validation/test split;
- train sample count and target-token count;
- max input length and max generated tokens;
- decoding settings;
- fine-tuning method, LR, warmup, batch size, gradient accumulation, epochs or
  steps, label smoothing, weight decay, and checkpoint-selection metric;
- eval harness version and metric script;
- dtype, cache setting, batch-size fallback, and any Mamba-specific eval
  fallback.

### Attribution Ladder

Only call something an architectural issue after checking, in this order:

1. **Metric/harness hygiene:** same split, same task, same scorer, raw outputs
   inspectable.
2. **Template/formulation parity:** same prompt family and output contract.
3. **Decoding parity:** greedy/beam/length/repetition settings checked.
4. **Implementation parity:** HF Mamba path is not silently changing logits or
   generation behavior.
5. **Fine-tuning recipe parity:** reasonable LR/checkpoint-selection/HPO tried.
6. **Base-quality parity:** base PPL/conditional loss compared under same
   tokenizer and target burden.
7. **Architecture:** only after the above gates fail to explain the gap.

### Minimum Evidence Per Final Claim

For each task family, a defensible final statement needs:

- best Mamba metric and best LLaMA metric;
- matched-control diagnostic where formulation changed;
- output examples showing success/failure shape;
- at least one metric beyond the headline score that targets the suspected
  root cause;
- note of whether the task is final, diagnostic-only, or still blocked by base
  training.

### Final Fair Downstream Evaluation Rule

After the root-cause checklist produces a defensible Mamba recipe, rerun the
selected downstream evaluations with a fair final comparison design:

- use the best defensible Mamba recipe and the best defensible LLaMA recipe;
- apply any general cross-architecture improvements to both models where
  compatible;
- keep model-specific fixes clearly labeled as model-specific;
- compare on the same task splits, prompts, metric scripts, decoding settings,
  and result-selection rules;
- compare each final Mamba result both to the previous best Mamba result and to
  the matched LLaMA result, so we can separate "Mamba improved" from "Mamba is
  competitive with the transformer baseline";
- if a rescue-stage improvement is shared but has only been run on one
  architecture so far, mark the final comparison as pending until the matched
  rerun is complete or explicitly justified;
- record final optimized results in the Google Sheet and keep the vault notes
  as the trace of why each recipe choice was made.

The final downstream suite should therefore be treated as a separate
confirmation phase, not as a loose continuation of diagnostics. Diagnostic
wins decide which recipe deserves final evaluation; final claims require the
best defensible Mamba and LLaMA recipes to be compared under matched conditions.

### Architecture Follow-Up Gate

#### Open follow-up TODOs from xLSTM full-base run

Status: `[ ]` pending after current xLSTM full-base artifacts land.

- `[ ]` xLSTM epoch-fairness check: after
  `xlstm_h736_ctx2048_native_4gpu_ddp_llama_budget_20260524` finishes, record
  the final reported `epoch`, token-slot exposure, and audit metrics. If the
  run finishes around `1.7`-`1.8` reported epochs, decide whether to launch a
  continuation or clean rerun that reaches the LLaMA base's `3` reported
  epochs, rather than calling the current run fully epoch-equivalent.
- `[ ]` Mamba base-loss regression/root-cause check: revisit the pure-Mamba
  pretrained/base candidates where held-out eval loss increased or failed to
  improve monotonically. Before treating Mamba as fairly trained but weak,
  inspect whether the loss rise points to recipe instability, data-ordering or
  streaming/chunking mismatch, checkpoint-selection issues, LR/warmup/weight
  decay problems, implementation differences in the HF Mamba path, or a real
  architecture/base-quality limitation.
- `[ ]` Official Mamba implementation pretraining check: pre-train a comparable
  Mamba base using the official Mamba codepath rather than the Hugging Face
  Mamba implementation, then compare held-out pretrain loss, clean generation
  loss, downstream-relevant diagnostics, runtime stability, tokenizer/data
  exposure, and checkpoint-selection behavior against the current HF-codepath
  Mamba base. Purpose: determine whether the weak/loss-regressing Mamba story
  is partly an implementation-path artifact before making architecture-level
  claims.

#### xLSTM strict-125M screen

Status: `[x]` complete / ambiguous-promising.

- Jobs: `861747` 10k streaming train -> `861748` clean generation-loss audit.
- Shape: HF xLSTM `h736_l12_h4_chunk64`, `126,901,952` parameters.
- Trainer eval loss improved from `3.607671` at checkpoint-5000 to `3.310756`
  at checkpoint-10000.
- Clean generation-loss checkpoint-10000 PPLs:
  - T2X Xho: xLSTM `283.95`, current Mamba base `347.77`, LLaMA base `64.20`;
  - AfriHG Xho: xLSTM `627.67`, current Mamba base `438.50`, LLaMA base
    `205.48`;
  - AfriHG Zul: xLSTM `852.06`, current Mamba base `605.85`, LLaMA base
    `267.32`.
- Decision: xLSTM is not a failed base screen. It beats Mamba on T2X clean
  loss but not AfriHG, and remains behind LLaMA on all three tasks.
- Carry-forward tag: `Ambiguous/promising architecture follow-up`.
- Next: prioritize xLSTM downstream adaptation or a longer xLSTM base screen
  before spending more compute on low-probability cheap Mamba rescue arms.

User reminder, 2026-05-22: do not let the Mamba rescue focus hide methods that
also improve LLaMA. If a decoder-only change helps both architectures, record
it as a shared protocol improvement and carry it into the later optimized
LLaMA rerun where compatible. The final downstream comparison must be fair
between the best defensible Mamba and best defensible LLaMA recipes, not just
between new Mamba and old LLaMA rows.

Review reminder, 2026-05-22 01:20 SAST: every rescue decision should say
whether the change improves current Mamba, current LLaMA, both, or neither.
Shared improvements become part of the future LLaMA carry-forward backlog; they
should not be counted as Mamba-only evidence, and final downstream claims should
wait until the best defensible Mamba and best defensible LLaMA recipes are
evaluated under matched conditions.

Advisor-facing reminder, 2026-05-22 01:59 SAST: once this checklist yields a
defensible Mamba base/recipe, downstream evaluation must be a fair comparison
between optimized Mamba and optimized LLaMA, not a comparison against stale
control rows. Any improvement found during Mamba rescue that also helps LLaMA
should be noted immediately and applied to the later LLaMA rerun where it keeps
the comparison valid.

Per-experiment review questions, 2026-05-22 02:30 SAST:

- Did this improve over the previous best Mamba result for the same task?
- Did the matched LLaMA control improve, degrade, or stay unchanged under the
  same decoder-only change?
- Is the result `Mamba-only`, `LLaMA-only`, `Shared`, or `Negative`?
- If it is `Shared`, has it been added to the LLaMA carry-forward backlog so
  the later optimized LLaMA rerun can use it where compatible?
- If it is `Mamba-only`, has the final-comparison note made clear that LLaMA
  should use its own best defensible recipe rather than the Mamba-specific
  setting?
- Before final downstream claims, have best defensible Mamba and best
  defensible LLaMA been evaluated fairly with matched splits, prompts, metric
  scripts, decoding/result-selection rules, and explicitly labeled
  model-specific exceptions?

## Recently Closed Background Gate

### [x] D1. Wide Pure-Mamba2 Base Gate

Question: does a wider public-Mamba-like pure-Mamba2 shape improve base quality?

Current jobs:

- `854987`: canary, completed successfully.
- `854988`: full 20k pretraining run, completed successfully.
- `854989`: clean generation-loss audit, failed with a Mamba2 fast-kernel
  channel-last stride error.
- `855987`: repaired clean generation-loss audit with `--mamba-torch-forward`,
  completed successfully.

Hypothesis:

- If the current Mamba base is under-optimized, a better pure-Mamba2 shape may
  reduce held-out pretraining loss and downstream clean generation loss.

Success criteria:

- Better held-out pretraining eval-loss trajectory than the failed/current
  fresh 20k run.
- Clean generation-loss closer to current Mamba base and ideally toward LLaMA
  base on T2X/AfriHG.
- No implementation-only failure.

Observations:

- 2026-05-21: torch-eval canary passed with fast CUDA training and cheaper eval.
- 2026-05-21 22:48 SAST: full run `854988` is still healthy and training
  around step `7750/20000` (`39%`), with recent losses roughly `5.2`-`5.4`.
  Clean-loss audit job `854989` remains pending on the dependency.
- 2026-05-21 23:08 SAST: full run `854988` remains healthy around step
  `8700/20000` (`44%`). Pulled only the small `checkpoint-5000`
  `trainer_state.json` locally; no weights were pulled. Checkpoint-5000
  train loss is `5.5603` and eval loss is `5.6133`.
- 2026-05-21 23:10 SAST: still running around step `8896/20000`; no new
  checkpoint artifact beyond checkpoint-5000 and no clean-loss audit yet.
- 2026-05-21 23:15 SAST: quota-first HEX check shows scratch still at
  `86.9%` (`checkpoints` `54G`, `results` `15G`, `logs` `92M`). Job `854988`
  is still running on L40S around step `9140/20000` (`46%`), and job `854989`
  remains dependency-pending. No new artifact beyond
  `checkpoint-5000/trainer_state.json`.
- 2026-05-21 23:22 SAST: quota-first HEX check is unchanged on scratch
  (`86.9%`, top consumers `checkpoints` `54G`, `results` `15G`, `logs`
  `92M`). Job `854988` is still running on L40S around step `9528/20000`
  (`48%`), and `854989` remains dependency-pending. No new artifact beyond
  `checkpoint-5000/trainer_state.json`.
- 2026-05-21 23:25 SAST: quota-first HEX check remains unchanged on scratch.
  Job `854988` is still running on L40S around step `9670/20000` (`48%`);
  `854989` remains dependency-pending. Still no artifact beyond
  `checkpoint-5000/trainer_state.json`.
- 2026-05-21 23:27 SAST: quota-first HEX check remains unchanged on scratch.
  Job `854988` is still running on L40S around step `9750/20000` (`49%`);
  `854989` remains dependency-pending. Still no artifact beyond
  `checkpoint-5000/trainer_state.json`.
- 2026-05-21 23:29 SAST: quota-first HEX check remains unchanged on scratch.
  Job `854988` is still running on L40S around step `9834/20000` (`49%`);
  `854989` remains dependency-pending. Still no artifact beyond
  `checkpoint-5000/trainer_state.json`.
- 2026-05-21 23:33 SAST: quota-first HEX check still shows scratch at
  `86.9%` with top consumers `checkpoints` `54G`, `results` `15G`, and
  `logs` `92M`. Job `854988` remains `RUNNING` on `srvrocgpu012` after about
  `03:20:48`; job `854989` remains dependency-pending. Artifact search still
  finds only `checkpoint-5000/trainer_state.json`. The `854988` tail is active
  inside an eval loop and shows no traceback, but no checkpoint-10000
  `trainer_state.json` or final summary exists yet.
- 2026-05-21 23:35 SAST: checkpoint-10000 `trainer_state.json` appeared and
  was pulled locally. Scratch increased to `87.6%` with checkpoints now `55G`.
  Job `854988` continued training after the 10k eval and `854989` remains
  dependency-pending.
- 2026-05-21 23:38 SAST: quota-first HEX check still shows scratch at
  `87.6%` with top consumers `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`. Job `854988` is still `RUNNING` on `srvrocgpu012` after about
  `03:25:37` and the log tail is around step `10212/20000` (`51%`) after the
  checkpoint-10000 eval. Job `854989` remains dependency-pending. No new
  artifact exists beyond checkpoint-5000 and checkpoint-10000 trainer states.
- 2026-05-21 23:39 SAST: quota-first HEX check remains at scratch `87.6%`
  with top consumers unchanged (`checkpoints` `55G`, `results` `15G`, `logs`
  `92M`). Job `854988` is still `RUNNING` on `srvrocgpu012` after about
  `03:27:14`; the log tail has advanced to about step `10298/20000` (`51%`).
  Job `854989` remains dependency-pending. No new artifact exists beyond
  checkpoint-5000 and checkpoint-10000 trainer states.
- 2026-05-21 23:41 SAST: quota-first HEX check remains at scratch `87.6%`
  with top consumers unchanged. Job `854988` is still `RUNNING` on
  `srvrocgpu012` after about `03:28:47`; the log tail has advanced to about
  step `10382/20000` (`52%`). Job `854989` remains dependency-pending. No new
  artifact exists beyond checkpoint-5000 and checkpoint-10000 trainer states.
- 2026-05-21 23:43 SAST: quota-first HEX check remains at scratch `87.6%`
  with top consumers unchanged. Job `854988` is still `RUNNING` on
  `srvrocgpu012` after about `03:30:35`; the log tail has advanced to about
  step `10470/20000` (`52%`). Job `854989` remains dependency-pending. No new
  artifact exists beyond checkpoint-5000 and checkpoint-10000 trainer states.
- 2026-05-21 23:48 SAST: quota-first HEX check remains at scratch `87.6%`
  with top consumers `checkpoints` `55G`, `results` `15G`, and `logs` `92M`.
  Job `854988` is still `RUNNING` on `srvrocgpu012` after about `03:35:26`;
  the live log tail has advanced to about step `10736/20000` (`54%`). Job
  `854989` remains dependency-pending. No new artifact exists beyond
  checkpoint-5000 and checkpoint-10000 trainer states, so there is nothing new
  to pull or summarize locally yet.
- 2026-05-21 23:51 SAST: quota-first HEX check remains at scratch `87.6%`
  with top consumers unchanged. Job `854988` is still `RUNNING` on
  `srvrocgpu012` after about `03:38:40`; the live log tail shows recent train
  losses around `5.10`-`5.32` and has advanced to about step `10885/20000`
  (`54%`). Job `854989` remains dependency-pending. Artifact search still
  finds only checkpoint-5000 and checkpoint-10000 trainer states.
- 2026-05-21 23:52 SAST: quota-first HEX check remains at scratch `87.6%`
  with top consumers unchanged. Job `854988` is still `RUNNING` on
  `srvrocgpu012` after about `03:40:16`; the live log tail has advanced to
  about step `10972/20000` (`55%`). Job `854989` remains dependency-pending.
  Artifact search still finds only checkpoint-5000 and checkpoint-10000 trainer
  states; no diagnostics exist yet.
- 2026-05-21 23:54 SAST: quota-first HEX check remains at scratch `87.6%`
  with top consumers unchanged. Job `854988` is still `RUNNING` on
  `srvrocgpu012` after about `03:41:49`; the live log tail has advanced to
  about step `11050/20000` (`55%`). Job `854989` remains dependency-pending.
  Artifact search still finds only checkpoint-5000 and checkpoint-10000 trainer
  states; no D1 diagnostics exist yet.
- 2026-05-21 23:55 SAST: quota-first HEX check remains at scratch `87.6%`
  with top consumers unchanged. Job `854988` is still `RUNNING` on
  `srvrocgpu012` after about `03:43:23`; the live log tail has advanced to
  about step `11124/20000` (`56%`). Job `854989` remains dependency-pending.
  Artifact search still finds only checkpoint-5000 and checkpoint-10000 trainer
  states. Next manual decision should wait for checkpoint-15000/final/audit
  artifacts rather than continuing minute-by-minute checks.
- 2026-05-22 00:01 SAST: quota-first HEX check remains at scratch `87.6%`.
  Top consumers: checkpoints `55G`, results `15G`, logs `92M`; checkpoint
  detail has `base_hpo` at `8.8G`, `fullft` at `14G`, `final_mamba` at
  `7.4G`, and `base_recovery` at `5.0G`. Job `854988` is still `RUNNING` on
  L40S `srvrocgpu012` after about `03:48:29`; the live log reached about step
  `11407/20000` (`57%`) with no visible traceback. Job `854989` remains
  dependency-pending. Artifact search still finds only checkpoint-5000 and
  checkpoint-10000 trainer states; no checkpoint-15000, final summary, or
  clean-loss diagnostics exist yet.
- 2026-05-22 00:04 SAST: quota-first HEX check still shows scratch `87.6%`.
  Job `854988` remains `RUNNING` on L40S after about `03:52:08`, with log
  progress around step `11572/20000` (`58%`) and recent train losses mostly
  around `5.1`-`5.3`. Job `854989` remains dependency-pending. Artifact search
  still finds only checkpoint-5000 and checkpoint-10000 trainer states; no
  checkpoint-15000, final summary, or clean-loss diagnostics exist yet.
- 2026-05-22 00:06 SAST: quota-first HEX check still shows scratch `87.6%`,
  with top-level consumers unchanged at checkpoints `55G`, results `15G`,
  logs `92M`. Job `854988` remains `RUNNING` on L40S after about `03:53:34`;
  tail progress reached about step `11644/20000` (`58%`). Job `854989`
  remains dependency-pending. Artifact search still finds only checkpoint-5000
  and checkpoint-10000 trainer states; no checkpoint-15000, final summary, or
  clean-loss diagnostics exist yet.
- 2026-05-22 00:12 SAST: quota-first HEX check still shows scratch `87.6%`,
  with top-level consumers `checkpoints` `55G`, `results` `15G`, and `logs`
  `92M`. Job `854988` remains `RUNNING` on L40S `srvrocgpu012` after about
  `04:00:08`; `scontrol` confirms it reserves one `l40s` GPU only and has a
  `12:00:00` wall limit. Job `854989` remains dependency-pending. Artifact
  search still finds only checkpoint-5000 and checkpoint-10000 trainer states;
  no checkpoint-15000, final summary, or clean-loss diagnostics exist yet.
  The log tail has advanced to about step `11985/20000` (`60%`) with no
  visible traceback.
- 2026-05-22 00:16 SAST: quota-first HEX check remains stable: home `12.8%`,
  scratch `87.6%`, top consumers `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`. Job `854988` is still `RUNNING` on L40S after about
  `04:05:01`; job `854989` is still dependency-pending. Artifact search still
  finds only checkpoint-5000 and checkpoint-10000 trainer states; no
  checkpoint-15000, final summary, or clean-loss diagnostics exist yet. The
  log tail has advanced to roughly step `12214/20000` (`61%`) with no visible
  traceback.
- 2026-05-22 00:18 SAST: quota-first HEX check remains stable: home `12.8%`,
  scratch `87.6%`, top consumers `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`. Job `854988` is still `RUNNING` on L40S after about
  `04:06:58`; job `854989` remains dependency-pending. Artifact search is
  unchanged with only checkpoint-5000 and checkpoint-10000 trainer states; no
  checkpoint-15000, final summary, or clean-loss diagnostics exist yet.
- 2026-05-22 00:20 SAST: quota-first HEX check remains stable: home `12.8%`,
  scratch `87.6%`, top consumers `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`. Job `854988` is still `RUNNING` on L40S after about
  `04:08:38`; job `854989` remains dependency-pending. Artifact search is
  unchanged with only checkpoint-5000 and checkpoint-10000 trainer states; no
  checkpoint-15000, final summary, or clean-loss diagnostics exist yet.
- 2026-05-22 00:22 SAST: quota-first HEX check remains stable: home `12.8%`,
  scratch `87.6%`, top consumers `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`. Job `854988` is still `RUNNING` on L40S `srvrocgpu012` after
  about `04:10:34`; job `854989` remains dependency-pending. Artifact search
  is unchanged with only checkpoint-5000 and checkpoint-10000 trainer states;
  no checkpoint-15000, final summary, or clean-loss diagnostics exist yet. The
  live log had advanced to about step `12507/20000` (`63%`) with no visible
  traceback.
- 2026-05-22 00:27 SAST: quota-first HEX check remains stable: home `12.8%`,
  scratch `87.6%`, top consumers `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`. `qstat`/`sacct` show job `854988` still `RUNNING` on L40S
  `srvrocgpu012` after about `04:14:23`; job `854989` remains
  dependency-pending. Artifact search is still unchanged with only
  checkpoint-5000 and checkpoint-10000 trainer states; no checkpoint-15000,
  final summary, or clean-loss diagnostics exist yet. `scontrol` confirmed
  stdout at `/home/lmbanr001/masters/sallm/slurm-854988.out`, and that tail
  shows active training around step `12700/20000` (`64%`) with no visible
  traceback.
- 2026-05-22 00:29 SAST: quota-first HEX check is still home `12.8%`, scratch
  `87.6%`. Required top-consumer check remains stable at `checkpoints` `55G`,
  `results` `15G`, `logs` `92M`. `qstat`/`sacct` show job `854988` still
  `RUNNING` on L40S after about `04:17:00`; `854989` remains
  dependency-pending. Artifact search still finds only checkpoint-5000 and
  checkpoint-10000 trainer states; the diagnostics directory does not yet
  exist. The stdout tail shows active training around step `12841/20000`
  (`64%`) with no visible traceback.
- 2026-05-22 00:31 SAST: quota-first HEX check remains home `12.8%`, scratch
  `87.6%`; top consumers are unchanged at `checkpoints` `55G`, `results`
  `15G`, `logs` `92M`. `qstat`/`sacct` show job `854988` still `RUNNING`
  after about `04:18:31`; `854989` remains dependency-pending. Artifact search
  remains unchanged with only checkpoint-5000 and checkpoint-10000 trainer
  states; no D1 diagnostics directory exists yet. The stdout tail shows active
  training around step `12891/20000` (`64%`) with no traceback.
- 2026-05-22 00:34 SAST: quota-first HEX check remains home `12.8%`, scratch
  `87.6%`; top consumers are unchanged at `checkpoints` `55G`, `results`
  `15G`, `logs` `92M`. `qstat`/`sacct` show job `854988` still `RUNNING`
  after about `04:22:03`; `854989` remains dependency-pending. Artifact search
  remains unchanged with only checkpoint-5000 and checkpoint-10000 trainer
  states; no clean-loss diagnostics files exist yet.
- 2026-05-22 00:38 SAST: quota-first HEX check remains home `12.8%`, scratch
  `87.6%`; mandatory top-consumer inspection is unchanged at `checkpoints`
  `55G`, `results` `15G`, and `logs` `92M`, with checkpoint detail still led
  by `fullft` `14G`, `base_hpo` `8.8G`, `final_mamba` `7.4G`, and
  `base_recovery` `5.0G`. `qstat`/`sacct` show `854988` still `RUNNING` on
  L40S `srvrocgpu012` after about `04:27:35`; `854989` remains
  dependency-pending. Artifact search still finds only checkpoint-5000 and
  checkpoint-10000 trainer states. The diagnostics directory does not exist
  yet, and there is no checkpoint-15000, `fresh_pretrain_summary.json`, or
  clean generation-loss audit. The live stdout tail reached about
  `13380/20000` (`67%`) with recent train losses around `5.05`-`5.30` and no
  visible traceback.
- 2026-05-22 00:42 SAST: quota-first HEX check remains home `12.8%`, scratch
  `87.6%`. Top consumers remain `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`; checkpoint detail remains `base_hpo` `8.8G`, `fullft` `14G`,
  `final_mamba` `7.4G`, `base_recovery` `5.0G`, `rerank` `1.2G`, and
  `final_llama` `950M`. `qstat`/`sacct` show `854988` still `RUNNING` on L40S
  after about `04:29:59`; `854989` remains dependency-pending. Artifact search
  still finds only checkpoint-5000 and checkpoint-10000 trainer states. The
  diagnostics directory still does not exist, and no checkpoint-15000 or final
  summary exists. The live stdout tail reached about `13455/20000` (`67%`),
  with recent train losses around `5.01`-`5.23` and no visible traceback.
- 2026-05-22 00:44 SAST: quota-first HEX check remains home `12.8%`, scratch
  `87.6%`; top consumers remain `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`, with checkpoint detail unchanged. `qstat`/`sacct` show
  `854988` still `RUNNING` on L40S after about `04:29:59`; `854989` remains
  dependency-pending. Artifact search still finds only checkpoint-5000 and
  checkpoint-10000 trainer states, the diagnostics directory does not exist,
  and no checkpoint-15000/final summary/clean-loss audit exists yet. The live
  stdout tail reached about `13455/20000` (`67%`) with recent train losses
  around `5.01`-`5.23` and no visible traceback. No new artifact was pulled.
- 2026-05-22 00:46 SAST: quota-first HEX check remains home `12.8%`, scratch
  `87.6%`; top consumers remain `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`, with checkpoint detail unchanged. `qstat`/`sacct` show
  `854988` still `RUNNING` on L40S after about `04:33:42`; `854989` remains
  dependency-pending. Artifact search still finds only checkpoint-5000 and
  checkpoint-10000 trainer states. The diagnostics directory still does not
  exist, and no checkpoint-15000/final summary/clean-loss audit exists yet. The
  live stdout tail reached about `13651/20000` (`68%`) with recent train losses
  around `5.01`-`5.31` and no visible traceback. No new artifact was pulled.
- 2026-05-22 00:48 SAST: quota-first HEX check remains home `12.8%`, scratch
  `87.6%`; top consumers remain `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`, with checkpoint detail unchanged. `qstat`/`sacct` show
  `854988` still `RUNNING` on L40S after about `04:35:35`; `854989` remains
  dependency-pending. Artifact search still finds only checkpoint-5000 and
  checkpoint-10000 trainer states. The diagnostics directory still does not
  exist, and no checkpoint-15000/final summary/clean-loss audit exists yet. The
  live stdout tail reached about `13765/20000` (`69%`) with no visible
  traceback. No new artifact was pulled.
- 2026-05-22 00:49 SAST: quota-first HEX check remains home `12.8%`, scratch
  `87.6%`; top consumers remain `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`, with checkpoint detail unchanged. `qstat`/`sacct` show
  `854988` still `RUNNING` on L40S after about `04:37:20`; `854989` remains
  dependency-pending. Artifact search still finds only checkpoint-5000 and
  checkpoint-10000 trainer states. The diagnostics directory still does not
  exist, and no checkpoint-15000/final summary/clean-loss audit exists yet. The
  live stdout tail reached about `13840/20000` (`69%`) with no visible
  traceback. No new artifact was pulled.
- 2026-05-22 00:55 SAST: quota-first HEX check remains home `12.8%`, scratch
  `87.6%`; top consumers remain `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`, with checkpoint detail unchanged. `qstat`/`sacct` show
  `854988` still `RUNNING` on L40S after about `04:42:39`; `854989` remains
  dependency-pending. Artifact search still finds only checkpoint-5000 and
  checkpoint-10000 trainer states. The diagnostics path yielded no files/no
  pull target, and no checkpoint-15000/final summary/clean-loss audit exists
  yet. The stdout tail reached about `14091/20000` (`70%`) with recent train
  losses around `5.04`-`5.22` and no visible traceback. No new artifact was
  pulled.
- 2026-05-22 00:57 SAST: quota-first HEX check remains home `12.8%`, scratch
  `87.6%`; top consumers remain `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`, with checkpoint detail unchanged. `qstat`/`sacct` show
  `854988` still `RUNNING` on L40S after about `04:44:50`; `854989` remains
  dependency-pending. Artifact search still finds only checkpoint-5000 and
  checkpoint-10000 trainer states; diagnostics still has no files. The stdout
  tail reached about `14210/20000` (`71%`) with no visible traceback. No new
  artifact was pulled.
- 2026-05-22 01:00 SAST: quota-first HEX check remains home `12.8%`, scratch
  `87.6%`; top consumers remain `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`, with checkpoint detail unchanged. `qstat`/`sacct` show
  `854988` still `RUNNING` on L40S after about `04:47:25`; `854989` remains
  dependency-pending. Artifact search still finds only checkpoint-5000 and
  checkpoint-10000 trainer states; diagnostics still has no files. The stdout
  tail reached about `14355/20000` (`72%`) with recent train losses around
  `5.00`-`5.24` and no visible traceback. No new artifact was pulled.
- 2026-05-22 01:01 SAST: quota-first HEX check remains home `12.8%`, scratch
  `87.6%`; top consumers remain `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`, with checkpoint detail unchanged. `qstat`/`sacct` show
  `854988` still `RUNNING` on L40S after about `04:49:17`; `854989` remains
  dependency-pending. Artifact search still finds only checkpoint-5000 and
  checkpoint-10000 trainer states; diagnostics still has no files. The stdout
  tail reached about `14438/20000` (`72%`) with no visible traceback. No new
  artifact was pulled.
- 2026-05-22 01:05 SAST: quota-first HEX check remains home `12.8%`, scratch
  `87.6%`; top consumers remain `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`, with checkpoint detail unchanged (`base_hpo` `8.8G`,
  `fullft` `14G`, `final_mamba` `7.4G`, `base_recovery` `5.0G`, `rerank`
  `1.2G`, `final_llama` `950M`). `qstat`/`sacct` show `854988` still
  `RUNNING` on L40S after about `04:52:36`; `854989` remains
  dependency-pending and `854987` remains completed `0:0`. Artifact search
  still finds only checkpoint-5000 and checkpoint-10000 trainer states;
  diagnostics still has no files. The stdout tail reached about
  `14609/20000` (`73%`) with recent train losses around `5.0`-`5.24` and no
  visible traceback. No new artifact was pulled.
- 2026-05-22 01:06 SAST: quota-first HEX check remains home `12.8%`, scratch
  `87.6%`; top consumers and checkpoint detail are unchanged. `qstat`/`sacct`
  show `854988` still `RUNNING` on L40S after about `04:54:13`; `854989`
  remains dependency-pending and `854987` remains completed `0:0`. Artifact
  search still finds only checkpoint-5000 and checkpoint-10000 trainer states;
  diagnostics still has no files. The stdout tail reached about
  `14689/20000` (`73%`) with no visible traceback. No new artifact was pulled.
- 2026-05-22 01:08 SAST: quota-first HEX check remains home `12.8%`, scratch
  `87.6%`; top consumers remain `checkpoints` `55G`, `results` `15G`, and
  `logs` `92M`, with checkpoint detail unchanged. `qstat`/`sacct` show
  `854988` still `RUNNING` on L40S after about `04:56:17`; `854989` remains
  dependency-pending and `854987` remains completed `0:0`. Artifact search
  still finds only checkpoint-5000 and checkpoint-10000 trainer states;
  diagnostics still has no files. The stdout tail reached about
  `14799/20000` (`74%`) with no visible traceback. No new artifact was pulled.
- 2026-05-22 01:16 SAST: `ssh hex` alias briefly failed DNS resolution, so
  the direct fallback `lmbanr001@137.158.158.180` with `HostKeyAlias` was used.
  Quota-first check succeeded: home `12.8%`, scratch increased to `88.3%`.
  Top consumers remain `checkpoints` `55G`, `results` `15G`, and `logs`
  `92M`. `qstat`/`sacct` show `854988` still `RUNNING` on L40S after about
  `05:02:59`; `854989` remains dependency-pending. `checkpoint-15000` has now
  landed, while diagnostics still has no files. Pulled only lightweight
  trainer states locally to
  `outputs/eval/diagnostics/mamba_fresh_base_wide_e2_s128_torcheval3_20k_20260521/checkpoint_states/`.
  The stdout tail reached about `15063/20000` (`75%`) with no visible
  traceback.
- 2026-05-22 01:18 SAST: quota-first direct-HEX check remains home `12.8%`,
  scratch `88.3%`; top consumers remain `checkpoints` `55G`, `results`
  `15G`, and `logs` `92M`. Checkpoint detail shows `base_hpo` has grown from
  `8.8G` to `9.6G`, consistent with the checkpoint-15000 save. `qstat`/`sacct`
  show `854988` still `RUNNING` on L40S after about `05:06:12`; `854989`
  remains dependency-pending. Artifact search shows checkpoint-5000,
  checkpoint-10000, and checkpoint-15000 trainer states only; no final summary
  or diagnostics yet. The stdout tail reached about `15230/20000` (`76%`) with
  no visible traceback. No new artifact was pulled.
- 2026-05-22 01:23 SAST: quota-first direct-HEX check remains home `12.8%`,
  scratch `88.3%`; top consumers remain `checkpoints` `55G`, `results` `15G`,
  and `logs` `92M`, with checkpoint detail still led by `fullft` `14G`,
  `base_hpo` `9.6G`, `final_mamba` `7.4G`, and `base_recovery` `5.0G`.
  `qstat`/`sacct` show `854988` still `RUNNING` on L40S after about
  `05:10:02`; `854989` remains dependency-pending. Artifact search still shows
  only checkpoint-5000, checkpoint-10000, and checkpoint-15000 trainer states;
  no checkpoint-20000, `fresh_pretrain_summary.json`, or diagnostics files
  exist yet. No new artifact was pulled.
- 2026-05-22 01:25 SAST: quota-first direct-HEX check remains home `12.8%`,
  scratch `88.3%`; top consumers remain `checkpoints` `55G`, `results` `15G`,
  and `logs` `92M`, with checkpoint detail unchanged. `qstat`/`sacct` show
  `854988` still `RUNNING` on L40S after about `05:12:51`; `854989` remains
  dependency-pending. Artifact search is unchanged with only checkpoint-5000,
  checkpoint-10000, and checkpoint-15000 trainer states. The stdout tail shows
  active training around step `15580/20000` (`78%`) with no visible traceback;
  no checkpoint-20000, final summary, or diagnostics files exist yet.
- 2026-05-22 01:27 SAST: quota-first direct-HEX check remains home `12.8%`,
  scratch `88.3%`; top consumers remain `checkpoints` `55G`, `results` `15G`,
  and `logs` `92M`. `qstat`/`sacct` show `854988` still `RUNNING` on L40S
  after about `05:15:14`; `854989` remains dependency-pending. Artifact search
  is unchanged with only checkpoint-5000, checkpoint-10000, and
  checkpoint-15000 trainer states. The stdout tail shows active training
  around step `15700/20000` (`78%`) with no visible traceback; no
  checkpoint-20000, final summary, or diagnostics files exist yet.
- 2026-05-22 01:30 SAST: quota-first direct-HEX check remains home `12.8%`,
  scratch `88.3%`; top consumers remain `checkpoints` `55G`, `results` `15G`,
  and `logs` `92M`. `qstat`/`sacct` show `854988` still `RUNNING` on L40S
  after about `05:18:49`; `854989` remains dependency-pending. Artifact search
  is unchanged with only checkpoint-5000, checkpoint-10000, and
  checkpoint-15000 trainer states. The stdout tail shows active training
  around step `15874/20000` (`79%`) with no visible traceback; no
  checkpoint-20000, final summary, or diagnostics files exist yet.
- 2026-05-22 01:36 SAST: quota-first direct-HEX check remains home `12.8%`,
  scratch `88.3%`; top consumers remain `checkpoints` `55G`, `results` `15G`,
  and `logs` `92M`. `qstat`/`sacct` show `854988` still `RUNNING` on L40S
  after about `05:24:07`; `854989` remains dependency-pending. Artifact search
  is unchanged with only checkpoint-5000, checkpoint-10000, and
  checkpoint-15000 trainer states. The stdout tail shows active training
  around step `16127/20000` (`81%`) with no visible traceback; no
  checkpoint-20000, final summary, or diagnostics files exist yet.
- 2026-05-22 01:40 SAST: quota-first direct-HEX check is materially unchanged:
  home `12.8%`, scratch `88.3%`, top consumers `checkpoints` `55G`, `results`
  `15G`, and `logs` `92M`. `qstat`/`sacct` show `854988` still `RUNNING` on
  L40S after about `05:27:52`; `854989` remains dependency-pending. Artifact
  search is unchanged with only checkpoint-5000, checkpoint-10000, and
  checkpoint-15000 trainer states. The stdout tail shows active training
  around step `16322/20000` (`82%`) with no visible traceback; no
  checkpoint-20000, final summary, or diagnostics files exist yet.
- 2026-05-22 01:52 SAST: quota-first direct-HEX check remains home `12.8%`,
  scratch `88.3%`, with top consumers `checkpoints` `55G`, `results` `15G`,
  and `logs` `92M`. `qstat`/`sacct` show `854988` still `RUNNING` on L40S
  after about `05:39:47`; `854989` remains dependency-pending. Artifact search
  is unchanged with only checkpoint-5000, checkpoint-10000, and
  checkpoint-15000 trainer states. The stdout tail shows active training
  around step `16930/20000` (`85%`) with no visible traceback. Local
  next-gate readiness was rechecked while waiting: B3/C3/E1/D3 submitters pass
  `bash -n`, B3 and E1 CLI help loads, Ruff passes for the B3/E1 scripts, and
  B3/C3/E1 remain explicit one-GPU L40S jobs while D3 remains gated by
  `D3_BASE_CANDIDATE_PATH`.
- 2026-05-22 01:55 SAST: quota-first direct-HEX check remains home `12.8%`,
  scratch `88.3%`, with the same top consumers. `854988` is still `RUNNING`
  after about `05:42:50`, with the tail around step `17080/20000` (`85%`).
  `854989` remains dependency-pending and there are still no checkpoint-20000,
  `fresh_pretrain_summary.json`, or clean-loss diagnostic files. Local
  final-summary paths were inspected: the training script writes
  `fresh_pretrain_summary.json` after saving `final_model`; the clean-loss
  audit writes `clean_generation_loss/summary.json` and
  `clean_generation_loss/loss_rows.jsonl`, covering checkpoints 5k/10k/15k/20k,
  final_model, prior failed/current 20k pure-Mamba baseline, current Mamba
  base, and LLaMA base on T2X Xho, AfriHG Xho, and AfriHG Zul validation.
- 2026-05-22 01:59 SAST: quota-first direct-HEX check remains home `12.8%`,
  scratch `88.3%`, with top consumers unchanged at checkpoints `55G`, results
  `15G`, and logs `92M`. `854988` is still `RUNNING` on L40S after about
  `05:48:20`; `854989` remains dependency-pending. Artifact search still finds
  only checkpoint-5000, checkpoint-10000, and checkpoint-15000 trainer states.
  No checkpoint-20000, `fresh_pretrain_summary.json`, or diagnostics files
  exist yet; the stdout tail shows active training around step `17352/20000`
  (`87%`) with no visible traceback.
- 2026-05-22 02:03 SAST: quota-first direct-HEX check is materially unchanged.
  `854988` remains `RUNNING` on L40S after about `05:50:50`, with stdout around
  step `17480/20000` (`87%`); `854989` remains dependency-pending. Scratch is
  still `88.3%` with top consumers checkpoints `55G`, results `15G`, and logs
  `92M`. Artifact search still finds only checkpoint-5000, checkpoint-10000,
  and checkpoint-15000 trainer states; no checkpoint-20000,
  `fresh_pretrain_summary.json`, or diagnostics files exist yet.
- 2026-05-22 02:05 SAST: quota-first direct-HEX check is still materially
  unchanged. `854988` remains `RUNNING` on L40S after about `05:52:34`, with
  stdout around step `17565/20000` (`88%`); `854989` remains
  dependency-pending. Scratch remains `88.3%` with top consumers checkpoints
  `55G`, results `15G`, and logs `92M`. Artifact search still finds only
  checkpoint-5000, checkpoint-10000, and checkpoint-15000 trainer states; no
  checkpoint-20000, `fresh_pretrain_summary.json`, or diagnostics files exist
  yet.
- 2026-05-22 02:06 SAST: quota-first direct-HEX check remains materially
  unchanged. `854988` is `RUNNING` on L40S after about `05:54:11`, with stdout
  around step `17646/20000` (`88%`); `854989` remains dependency-pending.
  Scratch remains `88.3%` with top consumers checkpoints `55G`, results `15G`,
  and logs `92M`. Artifact search still finds only checkpoint-5000,
  checkpoint-10000, and checkpoint-15000 trainer states; no checkpoint-20000,
  `fresh_pretrain_summary.json`, or diagnostics files exist yet.
- 2026-05-22 02:08 SAST: quota-first direct-HEX check remains stable.
  `854988` is `RUNNING` on L40S after about `05:55:45`, with stdout around
  step `17731/20000` (`89%`); `854989` remains dependency-pending. Scratch
  remains `88.3%` with top consumers checkpoints `55G`, results `15G`, and
  logs `92M`. Artifact search still finds only checkpoint-5000,
  checkpoint-10000, and checkpoint-15000 trainer states; no checkpoint-20000,
  `fresh_pretrain_summary.json`, or diagnostics files exist yet.
- 2026-05-22 02:09 SAST: quota-first direct-HEX check remains stable.
  `854988` is `RUNNING` on L40S after about `05:57:19`, with stdout around
  step `17813/20000` (`89%`); `854989` remains dependency-pending. Scratch
  remains `88.3%` with top consumers checkpoints `55G`, results `15G`, and
  logs `92M`. Artifact search still finds only checkpoint-5000,
  checkpoint-10000, and checkpoint-15000 trainer states; no checkpoint-20000,
  `fresh_pretrain_summary.json`, or diagnostics files exist yet.
- 2026-05-22 02:11 SAST: quota-first direct-HEX check remains stable.
  `854988` is `RUNNING` on L40S after about `05:58:53`, with stdout around
  step `17868/20000` (`89%`); `854989` remains dependency-pending. Scratch
  remains `88.3%` with top consumers checkpoints `55G`, results `15G`, and
  logs `92M`. Artifact search still finds only checkpoint-5000,
  checkpoint-10000, and checkpoint-15000 trainer states; no checkpoint-20000,
  `fresh_pretrain_summary.json`, or diagnostics files exist yet.
- 2026-05-22 02:16 SAST: quota-first direct-HEX check remains stable.
  `854988` is `RUNNING` on L40S after about `06:03:13`, with stdout around
  step `18078/20000` (`90%`); `854989` remains dependency-pending. Scratch
  remains `88.3%` with top consumers checkpoints `55G`, results `15G`, and
  logs `92M`. Artifact search still finds only checkpoint-5000,
  checkpoint-10000, and checkpoint-15000 trainer states; no checkpoint-20000,
  `fresh_pretrain_summary.json`, or diagnostics files exist yet.
- 2026-05-22 02:17 SAST: quota-first direct-HEX check remains stable.
  `854988` is `RUNNING` on L40S after about `06:05:11`, with stdout around
  step `18185/20000` (`91%`); `854989` remains dependency-pending. Scratch
  remains `88.3%` with top consumers checkpoints `55G`, results `15G`, and
  logs `92M`. Artifact search still finds only checkpoint-5000,
  checkpoint-10000, and checkpoint-15000 trainer states; no checkpoint-20000,
  `fresh_pretrain_summary.json`, or diagnostics files exist yet.
- 2026-05-22 02:24 SAST: added a local-only D1 artifact summarizer,
  `scripts/summarize_mamba_base_gate_artifacts.py`, so the final pull can be
  converted into a decision-ready Markdown report without W&B, model inference,
  or HEX login-node analysis. Verified against the current partial D1 directory:
  it reports checkpoint-5000/10000/15000 train/eval rows, correctly marks
  `fresh_pretrain_summary.json` and `clean_generation_loss/summary.json` as
  pending, and passes `python3 -m py_compile` plus
  `uv run ruff check scripts/summarize_mamba_base_gate_artifacts.py`.
  Command to run after final artifacts are pulled:
  `python3 scripts/summarize_mamba_base_gate_artifacts.py outputs/eval/diagnostics/mamba_fresh_base_wide_e2_s128_torcheval3_20k_20260521`.
- 2026-05-22 02:26 SAST: quota-first direct-HEX check remains stable.
  `854988` is `RUNNING` on L40S after about `06:14:11`, with stdout around
  step `18643/20000` (`93%`); `854989` remains dependency-pending. Scratch
  remains `88.3%` with top consumers checkpoints `55G`, results `15G`, and
  logs `92M`. Artifact search still finds only checkpoint-5000,
  checkpoint-10000, and checkpoint-15000 trainer states; no checkpoint-20000,
  `fresh_pretrain_summary.json`, or diagnostics files exist yet. Updated
  heartbeat `sallm-mamba-wide-torch-eval-gate-watch` to include the current
  state and the exact local `base_gate_summary.md` generation command once
  final/audit artifacts are pulled.
- 2026-05-22 02:33 SAST: quota-first direct-HEX check remains stable.
  `854988` is still `RUNNING` on L40S after about `06:21:05`, with stdout
  around step `18995/20000` (`95%`); `854989` remains dependency-pending.
  Scratch remains `88.3%` with top consumers checkpoints `55G`, results
  `15G`, and logs `93M`. Artifact search still finds only checkpoint-5000,
  checkpoint-10000, and checkpoint-15000 trainer states; no checkpoint-20000,
  `fresh_pretrain_summary.json`, or diagnostics files exist yet. No artifact
  pull or D1 classification is available.
- 2026-05-22 02:40 SAST: quota-first direct-HEX check remains stable.
  `854988` is still `RUNNING` on L40S after about `06:27:11`, with stdout
  around step `19299/20000` (`96%`); `854989` remains dependency-pending.
  Scratch remains `88.3%` with top consumers checkpoints `55G`, results
  `15G`, and logs `93M`. Artifact search still finds only checkpoint-5000,
  checkpoint-10000, and checkpoint-15000 trainer states; no checkpoint-20000,
  `fresh_pretrain_summary.json`, or diagnostics files exist yet. The local
  partial summary remains decision-not-ready because final and clean-loss
  artifacts are missing.
- 2026-05-22 03:11 SAST: quota-first direct-HEX check found `854988`
  completed successfully (`0:0`) and `854989` failed (`1:0`). Scratch is now
  `89.3%`, with top consumers checkpoints `56G`, results `15G`, and logs
  `93M`. Pulled `fresh_pretrain_summary.json`, checkpoint-20000
  `trainer_state.json`, and the clean-loss diagnostics directory locally.
  The clean-loss directory contains only a zero-byte `loss_rows.jsonl`; no
  `summary.json` was produced. `854989` failed before writing useful rows with
  the Mamba2 fast-kernel error:
  `causal_conv1d with channel last layout requires strides (x.stride(0) and
  x.stride(2)) to be multiples of 8`.
- 2026-05-22 10:12 SAST: repaired the clean-loss audit locally by adding a
  default `--mamba-torch-forward` path to `scripts/run_generation_loss_audit_clean.py`.
  This mirrors the D1 training eval workaround: Mamba2 uses HF's torch path
  during eval, while non-Mamba models are unchanged. Added repair-only submitter
  `scripts/resubmit_d1_wide_torcheval_clean_loss_2026_05_22.sh`. Local checks
  passed: `python3 -m py_compile`, `uv run ruff check`, CLI `--help`, and
  `bash -n`. Synced the scripts to HEX, checked quota/top consumers first
  (`scratch` `89.3%`, checkpoints `56G`, results `15G`, logs `93M`), then
  submitted repaired one-GPU L40S audit job `855987`. A post-submit check found
  `855987` running after `00:01:40`; the log shows `--mamba-torch-forward`,
  Mamba kernels available, template expansion progressing, and no immediate
  repeat of the stride traceback. Created heartbeat
  `sallm-mamba-d1-clean-loss-repair-watch` to pull/summarize/classify after
  completion.
- 2026-05-22 10:43 SAST: repaired audit job `855987` completed successfully
  (`0:0`) after `00:23:37`. Pulled final diagnostics and generated
  `outputs/eval/diagnostics/mamba_fresh_base_wide_e2_s128_torcheval3_20k_20260521/base_gate_summary.md`.

Results:

- Final pretraining trajectory, without clean downstream generation-loss yet:
  - checkpoint-5000: train loss `5.5603`, eval loss `5.6133`;
  - checkpoint-10000: train loss `5.2401`, eval loss `5.6522`;
  - checkpoint-15000: train loss `5.0590`, eval loss `5.6954`;
  - checkpoint-20000: train loss `5.1444`, eval loss `5.6942`;
  - trainer best metric still points to checkpoint-5000, so this run's
    validation trajectory worsens while train loss improves;
  - checkpoint-5000 remains better than the earlier failed/current
    expand4/state64 20k run's checkpoint-5000 eval loss `5.9383`, but the
    later checkpoints do not support a simple "train longer fixes the base"
    story;
  - final summary reports best checkpoint as checkpoint-5000, best metric
    `5.6133`, final train loss `5.5535`, and final model path under the D1
    base-HPO directory;
  - the original clean generation-loss audit failed before producing metrics,
    but the repaired audit `855987` completed with full metrics.
- Clean generation-loss audit, best D1 fresh wide checkpoint versus current
  Mamba base and LLaMA base:
  - T2X Xho: best D1 weighted NLL/PPL `9.0443` / `8470.3`; current Mamba base
    `5.8515` / `347.8`; LLaMA base `4.1595` / `64.0`;
  - AfriHG Xho: best D1 `9.3339` / `11314.7`; current Mamba base `6.0834` /
    `438.5`; LLaMA base `5.3260` / `205.6`;
  - AfriHG Zul: best D1 `9.3690` / `11719.9`; current Mamba base `6.4066` /
    `605.9`; LLaMA base `5.5892` / `267.5`.
- Carry-forward tag: `Negative`. The torch-forward audit repair is useful
  Mamba implementation hygiene, but the D1 wide fresh base itself is not a
  candidate to carry forward.

Decision:

- D1 is complete and negative.
- The wide pure-Mamba2 shape plus 20k fresh pretraining does not improve over
  the current public Mamba base on clean downstream generation loss and remains
  far behind the LLaMA base.
- Do not run D3 from this base; there is no credible base candidate.
- Do not start D2 yet. Follow the D1-negative branch: use E1/B3/C3 to reduce
  implementation and formulation uncertainty before spending on another base
  HPO matrix.
- Scratch is `89.3%`; no further submissions without explicit user approval
  for the next low-footprint gate or cleanup. No deletion has been performed.

---

## Phase 1: High-Information Diagnostics We Can Run Now

### [x] A1. POS/NER Per-Token Decoder-Only Label Scoring

Question: can POS/NER recover when we remove free-form generation and make the
decoder-only model score allowed labels token by token?

Hypothesis:

- POS/NER are failing mainly because free generation does not obey the required
  output shape, not because the model has no task signal.

Planned design:

- Score each allowed tag label with causal-LM loglikelihood for each input
  token.
- POS label set: UPOS labels.
- NER label set: BIO or atomic labels.
- Run current best Mamba POS/NER checkpoints.
- Run matched LLaMA controls if available.

Metrics:

- Parseable rate.
- Exact output length.
- POS token accuracy.
- NER BIO/entity F1.
- NER non-O recall.
- All-O rate.
- Runtime/memory.

Success criteria:

- Parseable/exact length should be `100%` by construction.
- Meaningful lift over current free-generation POS/NER results.
- For NER, non-O recall improves without all-O collapse.

Observations:

- 2026-05-21: prior pulled May 2 constrained-tagseq controls already showed
  that output-shape rescue alone is not enough for POS: Mamba Xho POS became
  parseable with exact length by construction, but validation token accuracy
  was only about `0.2509` with summed label scores and `0.2444` with mean label
  scores. LLaMA under the same older diagnostic was similarly weak on this
  specific overfit tagseq control (`0.2072` sum, `0.1903` mean), so the POS
  tagseq control itself needs careful interpretation rather than a simple
  Mamba-only blame.
- 2026-05-21: prior Mamba NER constrained scoring showed the suspected
  all-`O` failure: summed scoring predicted only `o` on both train and
  validation, giving validation token accuracy `0.7420` but validation
  per-sample BIO/entity F1 only `0.1758`. Mean scoring predicted non-`O`
  labels but had poor validation token accuracy (`0.2009`) and BIO/entity F1
  (`0.0308`). LLaMA did not collapse to all-`O` under the same old control but
  still had weak validation entity F1 (`0.0771` sum, `0.0430` mean).
- 2026-05-21: upgraded `scripts/run_constrained_tagseq_eval.py` to add
  explicit NER non-`O` precision/recall/F1, all-`O` sequence rate, all-`O` on
  gold-entity sequence rate, and global BIO entity precision/recall/F1 while
  preserving the older per-sample `bio_entity_f1` metric.
- 2026-05-21: submitted fresh A1 matched-control rerun:
  - `855018`: Mamba POS Xho constrained tag scoring.
  - `855019`: Mamba NER Xho BIO constrained tag scoring.
  - `855020`: LLaMA POS Xho constrained tag scoring.
  - `855021`: LLaMA NER Xho BIO constrained tag scoring.
  - Output root:
    `/scratch/lmbanr001/masters/sallm/results/eval/diagnostics/a1_constrained_tagseq_controls_20260521/`.
  - Local pull target:
    `outputs/eval/diagnostics/a1_constrained_tagseq_controls_20260521/`.
  - Monitoring folded into heartbeat
    `sallm-mamba-wide-torch-eval-gate-watch` because the app allows one
    thread-attached heartbeat.
- 2026-05-21: first A1 wave failed immediately because old diagnostic adapter
  paths under `checkpoints/diagnostics/` had been cleaned from scratch. This is
  an artifact hygiene issue, not a model result. Repaired the A1 submitter to
  use durable current/final model artifacts and the same positive-control
  tag-sequence example files for Mamba and LLaMA:
  - Mamba POS: final Mamba Xho POS adapter from
    `final_mamba/mamba_baseline_all_2026-04-27/mamba_pos_xho/final_adapter`.
  - Mamba NER: final Mamba Xho NER langfix adapter from
    `final_mamba/mamba_baseline_ner_langfix2_2026-04-28/mamba_ner_xho/final_adapter`.
  - LLaMA POS: merged positive-control model at
    `results/eval/final_test/llama_pos_xho/3_qyc5p564/_lm_eval/merged_model`.
  - LLaMA NER: merged positive-control model at
    `results/eval/final_llama/2026-04-11-r7/llama_ner_xho/_lm_eval/merged_model`.
  - Repaired jobs:
    - `855022`: Mamba POS Xho constrained tag scoring.
    - `855023`: Mamba NER Xho BIO constrained tag scoring.
    - `855024`: LLaMA POS Xho constrained tag scoring.
    - `855025`: LLaMA NER Xho BIO constrained tag scoring.
  - Repaired output root:
    `/scratch/lmbanr001/masters/sallm/results/eval/diagnostics/a1_constrained_tagseq_controls_20260521_r2/`.
  - Repaired local pull target:
    `outputs/eval/diagnostics/a1_constrained_tagseq_controls_20260521_r2/`.

Results:

- Repaired A1 r2 completed successfully:
  - jobs `855022`, `855023`, `855024`, `855025` all completed `0:0`;
  - artifacts pulled locally to
    `outputs/eval/diagnostics/a1_constrained_tagseq_controls_20260521_r2/`.
- POS validation:
  - Mamba POS: exact length/parseable `1.0`, token accuracy `0.0024` with
    sum scoring and `0.0924` with mean scoring. Sum scoring collapsed mostly
    to `x`; mean scoring mostly predicted `adp`/`punct`/`propn`.
  - LLaMA POS: exact length/parseable `1.0`, token accuracy `0.0863` with sum
    scoring and `0.0714` with mean scoring. This matched control is also weak,
    so this label-scoring prompt/formulation is not a valid POS rescue by
    itself.
- NER validation:
  - Mamba NER sum scoring: token accuracy `0.7420`, but all-`O` sequence rate
    `1.0`, all-`O` on gold-entity sequence rate `1.0`, non-`O` recall `0.0`,
    global BIO entity F1 `0.0`. This confirms a pure all-`O` collapse hidden
    by high token accuracy.
  - Mamba NER mean scoring: token accuracy `0.0180`, non-`O` recall `0.0684`,
    global BIO entity F1 `0.0018`; it avoids all-`O` by overpredicting labels
    such as `i-date`/`i-org`, not by recovering entities.
  - LLaMA NER sum scoring: token accuracy `0.1074`, non-`O` recall `0.0939`,
    global BIO entity F1 `0.0274`, all-`O` sequence rate `0.0078`.
  - LLaMA NER mean scoring: token accuracy `0.0330`, non-`O` recall `0.1133`,
    global BIO entity F1 `0.0334`. LLaMA avoids all-`O` collapse but still
    overgenerates entity labels and remains weak under this scoring setup.

Decision:

- A1 rejects the simplest "it is only output shape" explanation. Constraining
  output length and label vocabulary makes outputs parseable, but does not
  recover meaningful POS/NER accuracy.
- Mamba NER has a specific all-`O` prior under sum scoring, so A3
  label-prior calibration is still worth running.
- Because LLaMA is also weak under this exact scoring prompt, A1 alone cannot
  justify an architecture claim. POS/NER still need either better matched
  formulation/training or constrained decoding/scoring controls before final
  conclusions.

### [x] A2. POS/NER Constrained Tag-Sequence Decoding

Question: can generation be kept, but constrained to legal labels and exact
sequence shape?

Hypothesis:

- If constrained decoding works, we can keep a generation-style decoder-only
  formulation while preventing malformed tag outputs.

Planned design:

- Constrain decoding to valid POS/NER labels and separators.
- Dynamically cap max-new-tokens from input token count.
- Compare greedy constrained decoding against A1 label scoring.
- Use matched Mamba/LLaMA controls if formulation changes.

Metrics:

- Parseable rate.
- Exact length rate.
- POS token accuracy.
- NER F1 and non-O recall.
- Runtime/memory.

Success criteria:

- Parseable rate near `100%`.
- Accuracy/F1 competitive with A1.

Observations:

- 2026-05-21: added a true prefix-constrained `model.generate` path to
  `scripts/run_constrained_tagseq_eval.py`. Unlike A1's explicit
  per-label continuation scoring, this uses HF generation with
  `prefix_allowed_tokens_fn` so each next token must remain on a legal path
  toward exactly one tag per input token, then EOS.
- 2026-05-21: submitted matched Mamba/LLaMA POS/NER A2 jobs:
  - `855032`: Mamba POS Xho constrained generation.
  - `855033`: Mamba NER Xho BIO constrained generation.
  - `855034`: LLaMA POS Xho constrained generation.
  - `855035`: LLaMA NER Xho BIO constrained generation.
  - Remote output root:
    `/scratch/lmbanr001/masters/sallm/results/eval/diagnostics/a2_constrained_tagseq_generation_20260521/`.
  - Local pull target:
    `outputs/eval/diagnostics/a2_constrained_tagseq_generation_20260521/`.
- 2026-05-21: A2 jobs `855032`-`855035` completed successfully and artifacts
  were pulled locally.

Results:

- Validation metrics:

  | Model | Task | Token acc | Exact length | Parseable | Non-`O` recall | All-`O` seq | Global entity F1 | Label pattern |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
  | Mamba | POS Xho | `0.0612` | `1.0000` | `1.0000` | n/a | n/a | n/a | mostly `cconj`, some `propn` |
  | LLaMA | POS Xho | `0.0797` | `1.0000` | `1.0000` | n/a | n/a | n/a | cycles many tags |
  | Mamba | NER Xho | `0.0213` | `1.0000` | `1.0000` | `0.0765` | `0.0000` | `0.0118` | floods `b-date`/`i-date`/`i-org` |
  | LLaMA | NER Xho | `0.0973` | `1.0000` | `1.0000` | `0.0980` | `0.0039` | `0.0264` | mostly `i-date`, with other entity labels |

- A2 proves the implementation can produce exact-length, parseable tag
  sequences under true decoder generation.
- The task signal remains weak. Mamba POS predicts `cconj` for most tokens.
  Mamba NER predicts `3727` non-`O` tokens on validation for `980` gold
  non-`O` tokens and recovers only `15` complete entities with `1901` false
  positives.
- LLaMA also remains weak under this constrained-generation formulation, so A2
  is not a clean Mamba-only failure claim.

Decision:

- A2 closes the stricter "free generation output shape is the main cause"
  hypothesis. Even when output shape is guaranteed by prefix-constrained
  decoder-only generation, POS/NER do not recover.
- Since both Mamba and LLaMA are weak under this exact formulation, the next
  POS/NER path should be a better formulation/training control rather than
  more output-shape-only decoding.

### [x] A3. NER Label-Prior Calibration

Question: is NER all-O collapse caused by label-prior bias?

Hypothesis:

- The model may have entity signal but over-prefers `O` or generic non-entity
  labels.

Planned design:

- Start from A1 per-token label scores.
- Compare raw label scores to scores normalized by a null/minimal context label
  prior.
- Run Mamba first; run LLaMA control if useful.

Metrics:

- Non-O recall.
- All-O rate.
- Entity F1.
- False-positive rate.

Success criteria:

- Non-O recall improves without large false-positive explosion.

Observations:

- 2026-05-21: A1 r2 established the calibration target. Mamba NER sum scoring
  predicts all `o` on validation (`all_o_sequence_rate=1.0`, non-`O`
  recall `0.0`, global entity F1 `0.0`), while mean scoring overpredicts a few
  non-`O` labels and still has global entity F1 only `0.0018`.
- 2026-05-21: submitted Mamba-only first-pass `O`-bias sweep to test whether
  the all-`O` collapse is a tunable prior:
  - `855026`: `o=-0.5`.
  - `855027`: `o=-1.0`.
  - `855028`: `o=-2.0`.
  - `855029`: `o=-3.0`.
  - `855030`: `o=-4.0`.
  - Remote output root:
    `/scratch/lmbanr001/masters/sallm/results/eval/diagnostics/a3_mamba_ner_label_prior_bias_20260521/`.
  - Local pull target:
    `outputs/eval/diagnostics/a3_mamba_ner_label_prior_bias_20260521/`.
- 2026-05-21: jobs `855026`-`855030` completed successfully and artifacts
  were pulled locally. This sweep used only decoder-only label scoring with a
  scalar penalty on the `o` label.

Results:

- Validation metrics:

  | `O` bias | Token acc | Non-`O` P/R/F1 | All-`O` seq | Entity P/R/F1 | Pred non-`O` | Label pattern |
  | ---: | ---: | ---: | ---: | ---: | ---: | --- |
  | `-0.5` | `0.7420` | `0.0000/0.0000/0.0000` | `1.0000` | `0.0000/0.0000/0.0000` | `0` | all `o` |
  | `-1.0` | `0.7420` | `0.0000/0.0000/0.0000` | `1.0000` | `0.0000/0.0000/0.0000` | `0` | all `o` |
  | `-2.0` | `0.7420` | `0.0000/0.0000/0.0000` | `1.0000` | `0.0000/0.0000/0.0000` | `0` | all `o` |
  | `-3.0` | `0.6554` | `0.0082/0.0020/0.0033` | `0.8516` | `0.0000/0.0000/0.0000` | `244` | mostly `o`, plus `i-date`/`i-org` |
  | `-4.0` | `0.1221` | `0.0166/0.0490/0.0247` | `0.1602` | `0.0024/0.0016/0.0019` | `2900` | floods `i-date`/`i-org`/`b-date` |

- At `o=-3.0`, the model begins predicting entities but recovers only `2`
  correct non-`O` tokens on validation and no complete BIO entities.
- At `o=-4.0`, it predicts `2900` non-`O` tokens for `980` gold non-`O`
  tokens, but recovers only `48` correct non-`O` tokens and `1` complete
  entity. The output shape shifts from all-`O` collapse to uncontrolled
  `i-date`/`i-org` flooding.

Decision:

- Scalar `O`-bias calibration is not a usable NER rescue. It confirms that
  Mamba's NER failure is not only a fixed global preference for `O`: once `O`
  is penalized hard enough, the model does not recover boundaries or entity
  classes, it overpredicts a few labels.
- No matched LLaMA `O`-bias rerun is needed for this exact scalar-bias variant,
  because there is no Mamba candidate worth promoting. A more principled
  prior-normalized scorer could still be tested later, but the next POS/NER
  gates should focus on formulation/training and constrained generation rather
  than this simple bias sweep.

### [x] C1. AfriHG Official Test Rerun for Validation-Selected Beam/Length Settings

Question: do AfriHG validation decoding gains survive on official test?

Hypothesis:

- AfriHG Mamba is partly fixable through decoding length/focus control because
  validation beam/length settings improved chrF and length ratio.

Planned design:

- Use current best AfriHG checkpoints:
  - Xho checkpoint-656.
  - Zul checkpoint-892.
- Run validation-selected beam/length settings on official test.
- Record current checkpoint-selected result as baseline.
- Consider matched LLaMA decode variant if needed for fairness.

Metrics:

- chrF.
- BLEU.
- ROUGE-L.
- Length ratio.
- Repetition.
- Hook/entity preservation sample table.

Success criteria:

- Test chrF improves without making repetition or hallucination worse.
- Length ratio moves closer to reference.

Observations:

- 2026-05-21: launched official test reruns for the validation-selected
  `beam5/lp1.2` setting. This changes only the decoder settings relative to
  the checkpoint-selected baseline and keeps the same decoder-only Mamba
  checkpoints:
  - Xho: checkpoint-656, job `855036`;
  - Zul: checkpoint-892, first job `855037` failed immediately due Hydra
    `peft_adapter` override hygiene, then repaired/resubmitted as job
    `855040`;
  - remote output root:
    `/scratch/lmbanr001/masters/sallm/results/eval/diagnostics/c1_mamba_afrihg_beam_lp12_test_20260521/`;
  - local pull target:
    `outputs/eval/diagnostics/c1_mamba_afrihg_beam_lp12_test_20260521/`.
- Baselines to beat:
  - Xho checkpoint-selected test chrF `11.5335` with beam5/lp0.7;
  - Zul checkpoint-selected test chrF `13.5726` with beam5/lp0.7.

Results:

- Xho completed: chrF `15.1702`, BLEU `0.0057`, ROUGE-L `0.0572`. This is a
  strong improvement over the checkpoint-selected Xho test baseline chrF
  `11.5335`.
- Zul repaired job `855040` completed: chrF `17.1245`, BLEU `0.0078`,
  ROUGE-L `0.0736`. This is a strong improvement over the checkpoint-selected
  Zul test baseline chrF `13.5726`.
- Output-shape checks from pulled examples:
  - Xho mean prediction length `3.49` tokens vs mean reference length `4.85`,
    length ratio `0.72`, empty rate `0.006`, repeated-bigram rate `0.012`;
  - Zul mean prediction length `3.60` tokens vs mean reference length `4.76`,
    length ratio `0.76`, empty rate `0.005`, repeated-bigram rate `0.003`.

Decision:

- Keep beam5/lp1.2 as the current best Mamba AfriHG decoding recipe.
- This is a Mamba recipe improvement, not an architecture conclusion by
  itself. Before final fair Mamba-vs-LLaMA claims, run or plan a matched LLaMA
  AfriHG decode-variant check and carry forward the setting only if the final
  comparison remains fair.

### [x] C1b. Matched LLaMA AfriHG Beam/Lp Decode Check

Question: is beam5/lp1.2 a Mamba-specific rescue, or a general AfriHG decoding
improvement that should also be carried forward to LLaMA?

Planned design:

- Run current LLaMA AfriHG Xho/Zul test configs with the same decode override:
  beam5, length penalty `1.2`, early stopping `false`.
- Compare against the current LLaMA AfriHG baseline rows and against the new
  C1 Mamba rows.

Metrics:

- chrF.
- BLEU.
- ROUGE-L.
- Length ratio and repetition from examples.

Observations:

- 2026-05-21: launched matched LLaMA decode check:
  - first Xho job `855049` failed immediately because the config default
    pointed to a stale LLaMA checkpoint path;
  - dependent Zul job `855050` was cancelled/stranded;
  - patched submitter to use durable task-specific full-FT merged model paths;
  - repaired Xho job `855054`;
  - repaired Zul job `855055`, serially dependent on `855054`;
  - remote output root:
    `/scratch/lmbanr001/masters/sallm/results/eval/diagnostics/c1b_llama_afrihg_beam_lp12_test_20260521/`;
  - local pull target:
    `outputs/eval/diagnostics/c1b_llama_afrihg_beam_lp12_test_20260521/`.
- 2026-05-21 22:18 SAST: repaired Xho job `855054` completed successfully
  and was pulled locally. Repaired Zul job `855055` is running.
- 2026-05-21 22:48 SAST: repaired Zul job `855055` is still running on L40S,
  loaded the intended task-specific merged checkpoint, prepared `1776` test
  examples, and shows no final artifacts yet.
- 2026-05-21 23:01 SAST: repaired Zul job `855055` completed successfully
  (`0:0`, elapsed `00:39:50`) and artifacts were pulled locally.

Results:

- LLaMA AfriHG Xho beam5/lp1.2:
  - chrF `13.1149`, ROUGE-L `0.0484`, BLEU `0.0011`;
  - below the current-stack LLaMA Xho parity value previously tracked
    for AfriHG Xho (`14.2017`) and below the C1 Mamba Xho beam5/lp1.2 value
    (`15.1702`);
  - whitespace mean prediction length `26.19` vs reference `4.85`, empty rate
    `0.0000`; the matched LLaMA decode setting over-generates long
    excerpt-like outputs on Xho rather than concise headlines.
- LLaMA AfriHG Zul beam5/lp1.2:
  - chrF `20.8709`, ROUGE-L `0.0999`, BLEU `0.0040`;
  - below the current-stack LLaMA Zul parity value previously tracked
    (`21.5552`) and older best (`23.0041`), but still above the C1 Mamba Zul
    beam5/lp1.2 value (`17.1245`);
  - whitespace mean prediction length `11.74` vs reference `4.76`, empty rate
    `0.0000`; the matched LLaMA decode setting again over-generates relative
    to the headline target.

Decision:

- Complete. Beam5/lp1.2 is a useful Mamba AfriHG rescue but not a shared
  LLaMA/Mamba AfriHG decoding protocol. It improves Mamba Xho/Zul over the
  previous checkpoint-selected Mamba baselines, while the matched LLaMA runs
  fall below their tracked LLaMA baselines and over-generate relative to the
  headline target.
- Carry-forward tag: `Mamba-only` for current AfriHG decoding. Final fair
  comparison must label this as a Mamba-specific decode fix, not apply it to
  LLaMA unless a separate LLaMA-optimized decode setting is selected and
  documented.

### [x] C2. AfriHG Hook/Focus Error Table

Question: are weak headlines missing main entities/events, or are they mostly
too short?

Hypothesis:

- Mamba AfriHG failures split into generic-short headlines and wrong-focus
  headlines; knowing the split determines whether to use decoding or prompt
  formulation.

Planned design:

- Build a 20-example Xho and 20-example Zul manual table.
- Include article hook, reference, Mamba output, LLaMA output, error type.
- Add crude entity/date/number preservation flags.

Metrics:

- Generic headline rate.
- Main entity preserved.
- Main event/action preserved.
- Date/number preserved.
- Wrong-focus rate.

Success criteria:

- Clear dominant error class identified.

Observations:

- 2026-05-21: started C2 using completed Mamba C1 official test outputs and
  older LLaMA AfriHG baseline outputs as a temporary comparison column. Full
  note:
  `notes/2026-05-21-c2-afrihg-hook-focus-error-table.md`.
- 2026-05-21: replaced/augmented the Xho comparison with matched C1b LLaMA
  beam5/lp1.2 outputs after job `855054` completed and was pulled. Zul still
  waits for job `855055`.
- 2026-05-21: replaced/augmented the Zul comparison with matched C1b LLaMA
  beam5/lp1.2 outputs after job `855055` completed and was pulled.

Results:

- Preliminary heuristic buckets for Mamba C1 outputs:
  - Xho, n=`1305`: off-topic/generic `37.2%`, too short/generic `31.9%`,
    wrong focus `9.4%`, repetition/loop `6.2%`, reasonable partial only
    `3.4%`, empty `0.6%`.
  - Zul, n=`1776`: off-topic/generic `54.7%`, too short/generic `14.9%`,
    wrong focus `7.5%`, repetition/loop `3.7%`, reasonable partial only
    `2.4%`, empty `0.5%`.
- Beam5/lp1.2 has mostly removed the worst empty-output problem but has not
  solved headline planning. Residual errors are dominated by generic/off-topic
  or wrong-focus headline selection, with some repetition.
- Matched Xho C1b comparison: LLaMA often captures more article content and
  entities than Mamba, but beam5/lp1.2 makes LLaMA over-generate long
  excerpt-like or noisy outputs. This supports the C1b Xho finding that
  beam5/lp1.2 is not a shared AfriHG decode improvement for LLaMA Xho.
- Matched Zul C1b comparison: LLaMA remains stronger in chrF (`20.8709` vs
  Mamba `17.1245`) and often includes more article content, but the same
  beam5/lp1.2 decode setting over-generates relative to concise headlines and
  remains below tracked LLaMA Zul baselines.

Decision:

- Complete. C2 supports the view that AfriHG needs better semantic
  focus/headline planning and likely better base quality and/or task
  formulation, not only more output-length control. Beam5/lp1.2 is a
  Mamba-specific length rescue, but the remaining Mamba errors are mostly
  generic/off-topic or wrong-focus headline choices.
- Carry-forward tag: `Mamba-only` for beam5/lp1.2, with C3 focus-prompt
  validation as the next formulation gate when the active base run releases
  the queue.

### [x] B1. T2X Token-Class Conditional-Loss Audit

Question: does Mamba assign disproportionately worse likelihood to source
entity/value tokens?

Hypothesis:

- T2X underperformance is driven by copy/coverage difficulty, especially for
  proper names and structured values.

Planned design:

- Use T2X prompts/references.
- Label reference tokens as source-entity, source-value, relation/verbalization,
  or other.
- Compute teacher-forced NLL per class for Mamba and LLaMA.

Metrics:

- NLL/PPL by token class.
- Mamba-minus-LLaMA NLL gap by class.
- Relationship between token-class NLL and preservation failures.

Success criteria:

- Identifies whether entity/value tokens are the disproportionate weakness.

Observations:

- 2026-05-21: prepared the B1 diagnostic implementation but did not submit it
  yet because scratch is `86.9%` and C1b/D1 L40S jobs are still active.
- Added `scripts/run_t2x_token_class_loss_audit.py`.
  - Loads the same generation task config path as the clean loss audits.
  - Uses the chat-template assistant mask to score only target/reference tokens.
  - Computes per-token teacher-forced NLL and buckets target tokens by crude
    exact source-token overlap: `source_entity`, `source_value`,
    `source_relation`, or `other`.
  - Disables cache for forward scoring where the model exposes `use_cache`.
  - Records `token_loss_rows.jsonl` plus `summary.json`.
- Added `scripts/submit_b1_t2x_token_class_loss_audit_2026_05_21.sh`.
  - One L40S GPU.
  - Compares current Mamba T2X continuation checkpoint against current LLaMA
    T2X opt-chrF checkpoint.
  - Uses validation split and `max_samples=256`.
- Local validation:
  - `uv run ruff check scripts/run_t2x_token_class_loss_audit.py` passed;
  - `bash -n scripts/submit_b1_t2x_token_class_loss_audit_2026_05_21.sh`
    passed;
  - `uv run python scripts/run_t2x_token_class_loss_audit.py --help` passed.
- 2026-05-21 22:13 SAST: submitted B1 as job `855076` after checking scratch
  and active jobs. It is running on L40S `srvrocgpu013`.
  - Remote result path:
    `/scratch/lmbanr001/masters/sallm/results/eval/diagnostics/b1_t2x_token_class_loss_audit_20260521/`.
  - Immediate log tail showed Mamba fast path CUDA kernels available and the
    intended diagnostic command started without immediate failure.

Results:

- Completed as job `855076` and pulled locally to
  `outputs/eval/diagnostics/b1_t2x_token_class_loss_audit_20260521/`.
- Validation split, `256` examples, `7009` scored target tokens.
- Mamba vs LLaMA teacher-forced NLL/PPL by exact source-token class:

  | Class | Tokens | Mamba NLL | Mamba PPL | LLaMA NLL | LLaMA PPL | Mamba-LLaMA NLL Gap | PPL Ratio |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `other` | `6594` | `2.8759` | `17.74` | `1.6975` | `5.46` | `+1.1783` | `3.25x` |
  | `source_entity` | `212` | `3.6418` | `38.16` | `0.8762` | `2.40` | `+2.7656` | `15.89x` |
  | `source_value` | `196` | `3.7529` | `42.65` | `1.4464` | `4.25` | `+2.3065` | `10.04x` |
  | `source_relation` | `7` | `8.9985` | `8090.91` | `6.0458` | `422.32` | `+2.9527` | `19.16x` |

Decision:

- Complete. B1 supports the B3 generated-output finding: Mamba's T2X weakness
  is not only decoding. Under teacher-forced scoring, Mamba is worse than
  LLaMA on all target tokens, but the gap is disproportionately large on
  exact source entity/value tokens.
- Treat `source_relation` as low-confidence because only `7` tokens matched
  that crude class.
- Caveat: exact source-token matching undercounts legitimate translations and
  paraphrases. The result is still useful because both architectures are scored
  on the same references and class labels.
- Carry-forward tag: `Mamba-only` root-cause evidence. The next T2X rescue
  should test B2 placeholder/reinsertion with matched Mamba and LLaMA controls.

### [x] B2. T2X Delexicalized Placeholder/Reinsertion Diagnostic

Question: is Mamba bad at relation verbalization, or at copying names/values?

Hypothesis:

- If entity/value copying is the root issue, replacing them with placeholders
  should make the relation-verbalization task easier and improve metrics after
  deterministic reinsertion.

Planned design:

- Replace source entity/value with placeholders like `ENTITY_A`, `VALUE_A`.
- Generate verbalization with placeholders.
- Deterministically reinsert original entity/value.
- Run matched Mamba and LLaMA controls.

Metrics:

- Placeholder preservation.
- chrF/BLEU/ROUGE after reinsertion.
- Entity/value preservation after reinsertion.
- Relation correctness sample table.

Success criteria:

- Mamba improves substantially, especially on value/entity preservation.

Observations:

- 2026-05-21: added
  `scripts/run_t2x_placeholder_reinsertion_diagnostic.py`.
  - Loads the same T2X generation task config as B1.
  - Replaces exact source entity/value strings in prompt and reference with
    stable placeholders such as `ENTITY_A` and `VALUE_A`.
  - Generates placeholder verbalizations with the same decoder-only model
    interface.
  - Deterministically reinserts original entity/value strings before scoring
    final text metrics.
  - Reports placeholder recall/exact-set rate, entity/value preservation after
    reinsertion, and normal chrF/BLEU/ROUGE metrics.
- Added `scripts/submit_b2_t2x_placeholder_reinsertion_2026_05_21.sh`.
  - One L40S GPU, validation split, `max_samples=256`.
  - Matched current Mamba T2X continuation checkpoint against current LLaMA
    T2X opt-chrF checkpoint.
- Local validation:
  - `uv run ruff check scripts/run_t2x_placeholder_reinsertion_diagnostic.py`
    passed;
  - `uv run python scripts/run_t2x_placeholder_reinsertion_diagnostic.py --help`
    passed;
  - `bash -n scripts/submit_b2_t2x_placeholder_reinsertion_2026_05_21.sh`
    passed.
- Submitted as job `855304` after quota/top-consumer check:
  - scratch `86.9%`;
  - top consumers checkpoints `54G`, results `15G`, logs `91M`;
  - job started running on L40S `srvrocgpu013`.
- 2026-05-21: job `855304` completed successfully in `00:04:29` and artifacts
  were pulled locally to
  `outputs/eval/diagnostics/b2_t2x_placeholder_reinsertion_20260521/`.
- Post-hoc hygiene: the generated `summary.json` averages placeholder recall
  over all examples, including examples where the placeholdered reference has
  no exact placeholders. The decision below uses the cleaner
  expected-placeholder-only recall computed from `examples.jsonl`.

Results:

- Validation split, `256` examples.

  | Model | Reinserted chrF | Reinserted ROUGE-L | Reinserted BLEU | Expected-placeholder examples | Placeholder any on expected | Placeholder recall on expected | Exact placeholder set on expected | Entity coverage | Value coverage |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | Mamba | `8.2455` | `0.0773` | `0.0000` | `216` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
  | LLaMA | `27.2327` | `0.2604` | `0.0643` | `216` | `0.8750` | `0.4815` | `0.2963` | `0.8867` | `0.1299` |

Decision:

- Complete. B2 is negative as a Mamba rescue: Mamba did not generate any
  expected placeholders, so deterministic reinsertion had nothing useful to
  reinsert.
- This means the T2X gap is not solved by simply removing exact entity/value
  copy burden. Mamba also struggles with following the placeholder abstraction
  and binding the relation output to placeholders.
- LLaMA is an important matched control: it can use the placeholder abstraction
  substantially better, especially for entity placeholders, although value
  preservation remains weak and unexpected placeholders occur.
- Carry-forward tag: `LLaMA-only` diagnostic improvement / `Negative` Mamba
  rescue. Do not promote placeholdering as a Mamba recipe without a better
  formulation, for example natural-language slot labels or training exposure.

### [x] B3. T2X Source-Preservation Decoding Diagnostic

Question: can prompting/decoding improve entity/value preservation without
retraining?

Hypothesis:

- Mamba may preserve more source content if decoding discourages repetition and
  the prompt explicitly asks to include entity and value.

Planned design:

- Validation-only small sweep:
  - current best decoding;
  - greedy/repetition penalty;
  - conservative beam;
  - prompt checklist requiring entity and value.
- Track source preservation and standard metrics.

Metrics:

- Entity word preservation.
- Value word preservation.
- Repetition.
- chrF/BLEU/ROUGE.

Success criteria:

- Preservation and chrF both improve. If only preservation improves, diagnostic
  only.

Observations:

- 2026-05-21: started B3 with a local forensic pass over already-pulled current
  best Mamba T2X outputs and current-stack LLaMA T2X parity outputs. Full note:
  `notes/2026-05-21-b3-t2x-source-preservation-forensics.md`.
- This pass uses crude exact source-token overlap for entity/value preservation.
  It undercounts legitimate translated/paraphrased values, but is useful as a
  relative diagnostic because the same rule is applied to both architectures.
- 2026-05-21 23:15 SAST: prepared the actual B3 prompt/decoding gate locally
  as `scripts/run_t2x_source_preservation_decoding_diagnostic.py` plus
  `scripts/submit_b3_t2x_source_preservation_decode_2026_05_21.sh`.
  The gate is matched Mamba/LLaMA on validation and tests four decoder-only
  variants: baseline beam3/lp1.2, greedy with repetition controls, conservative
  beam with repetition controls, and a source-preservation checklist prompt.
  Local checks passed:
  - `python3 -m py_compile scripts/run_t2x_source_preservation_decoding_diagnostic.py`;
  - `uv run ruff check scripts/run_t2x_source_preservation_decoding_diagnostic.py`;
  - `uv run python scripts/run_t2x_source_preservation_decoding_diagnostic.py --help`;
  - `bash -n scripts/submit_b3_t2x_source_preservation_decode_2026_05_21.sh`.
  It has not been submitted because D1 is active and scratch is still above
  `85%`.
- 2026-05-21 23:48 SAST: revalidated the runnable B3 script and submitter
  against the current worktree:
  - `python3 -m py_compile scripts/run_t2x_source_preservation_decoding_diagnostic.py`;
  - `uv run ruff check scripts/run_t2x_source_preservation_decoding_diagnostic.py`;
  - `bash -n scripts/submit_b3_t2x_source_preservation_decode_2026_05_21.sh`.
  All passed. Still not submitted because D1 remains active and scratch is
  above `85%`.
- 2026-05-22 01:32 SAST: revalidated B3 locally while D1 continued running:
  - `bash -n` still passes for the prepared submitter;
  - `uv run python scripts/run_t2x_source_preservation_decoding_diagnostic.py --help`
    loads and shows the expected `--model`, `--variant`, and `--output-dir`
    interface;
  - `uv run ruff check scripts/run_t2x_source_preservation_decoding_diagnostic.py`
    passes.
- 2026-05-22 11:04 SAST: submitted B3 as jobs `856266`, `856267`, and repaired
  successful job `856269`. The first two failed immediately because the script
  assigned raw dicts into OmegaConf structured fields (`TemplateRef`, then
  `DecodingConfig`). Patched the script to assign `TemplateRef(...)` and
  `DecodingConfig.from_any(...)`; local py_compile and Ruff passed; `856269`
  completed successfully and artifacts were pulled to
  `outputs/eval/diagnostics/b3_t2x_source_preservation_decode_20260521/`.

Results:

- Current best Mamba checkpoint-selected T2X test: chrF `32.6125`, BLEU
  `0.0580`, ROUGE-L `0.2966`.
- Current-stack LLaMA T2X parity output: chrF `53.4796`, BLEU `0.2155`,
  ROUGE-L `0.5053`.
- Exact source-token preservation:
  - Mamba mean entity coverage `0.241`, any entity token `0.397`, complete
    entity `0.106`;
  - LLaMA mean entity coverage `0.511`, any entity token `0.762`, complete
    entity `0.220`;
  - Mamba mean value coverage `0.202`, any value token `0.294`, complete value
    `0.106`;
  - LLaMA mean value coverage `0.437`, any value token `0.563`, complete value
    `0.272`;
  - Mamba repetition rate `0.082` vs LLaMA `0.034`.
- Worst Mamba value-preservation relation types include `areaTotal`, `award`,
  `cityServed`, `league`, `mediaType`, `status`, date fields, and
  `runwayLength`.
- B3 runnable gate, validation split, 256 examples:
  - Mamba baseline beam3/lp1.2: chrF `28.064`, entity coverage `0.410`,
    value coverage `0.320`, repetition `0.108`;
  - Mamba source-checklist beam3/lp1.2: chrF `28.148`, entity coverage
    `0.393`, value coverage `0.281`, repetition `0.105`;
  - Mamba greedy/repetition-control: chrF `20.141`, entity coverage `0.264`,
    value coverage `0.169`, repetition `0.027`;
  - Mamba conservative beam/repetition-control: chrF `21.737`, entity coverage
    `0.329`, value coverage `0.269`, repetition `0.067`;
  - LLaMA baseline beam3/lp1.2: chrF `45.885`, entity coverage `0.763`,
    value coverage `0.594`, repetition `0.047`;
  - LLaMA source-checklist: chrF `44.817`, entity coverage `0.765`, value
    coverage `0.573`, repetition `0.055`.

Decision:

- Complete. Existing outputs plus B1/B2/B3 support entity/value preservation as
  a real T2X gap:
  - B1 shows a disproportionate Mamba-vs-LLaMA likelihood gap on exact source
    entity/value target tokens;
  - B2 shows placeholdering is negative for Mamba but usable by LLaMA.
- B3 shows direct decoder-only prompt/decoding changes do not rescue Mamba T2X
  source preservation. The source-checklist prompt gives only a tiny chrF gain
  while lowering entity/value preservation, and repetition-control variants
  collapse quality.
- Carry-forward tag: `Negative` for the prompt/decoding changes; `Mamba-only`
  root-cause evidence for source-preservation weakness.
- Do not carry source-checklist or repetition-control variants into final
  Mamba/LLaMA reruns.

---

## Phase 2: Formulation and Parity Checks

### [x] C3. AfriHG Prompted Headline Focus Variant

Question: can a stricter headline prompt reduce generic Mamba headlines?

Hypothesis:

- Mamba may need more explicit instruction to include the main actor/event and
  avoid generic country-level headlines.

Planned design:

- Validation-only matched Mamba/LLaMA diagnostic.
- Try a prompt such as: "Write a 4-8 word headline. Include the main person,
  team, organization, or event when present. Output only the headline."

Metrics:

- chrF/BLEU/ROUGE.
- Length ratio.
- Hook/entity preservation.
- Generic headline rate.

Success criteria:

- Mamba improves in both metric and hook preservation without unfairly changing
  only Mamba's formulation.

Observations:

- 2026-05-21 22:48 SAST: prepared but did not submit this gate because C1b
  Zul and D1 were still active and scratch remained above `85%`.
- 2026-05-21 23:01 SAST: C1b is now closed, but D1 is still active and scratch
  remains above `85%`, so C3 stays prepared only.
- Added template `afrihg_headline/focus_v1`, which asks the model to focus on
  the central event/action/outcome, keep central entities, avoid whole-article
  summarization, and output only the headline.
- Added ready-to-submit script
  `scripts/submit_c3_afrihg_focus_prompt_val_2026_05_21.sh`.
  - Validation-only, `max_samples_per_lang=256`.
  - Matched Mamba Xho/Zul and LLaMA Xho/Zul controls.
  - Uses current Mamba AfriHG checkpoint-selected checkpoints and
    task-specific LLaMA merged checkpoints.
  - Keeps beam5/lp1.2 so the main changed variable is the prompt template.
  - Serial L40S one-GPU chain.
- Local validation passed:
  - `bash -n scripts/submit_c3_afrihg_focus_prompt_val_2026_05_21.sh`;
  - template registry lookup for `afrihg_headline/focus_v1`.
- 2026-05-21 23:48 SAST: re-ran
  `bash -n scripts/submit_c3_afrihg_focus_prompt_val_2026_05_21.sh`; it
  passed. Still not submitted while D1 is active and scratch is above `85%`.
- 2026-05-22 11:08 SAST: submitted first C3 chain `856285` -> `856288`.
  `856285` failed immediately because Hydra required
  `+eval.evaluation.generation_tasks.0.decoding.max_batch_size=8` for the
  previously absent `max_batch_size` key. Patched the submitter, cancelled
  stranded dependency jobs where possible, and resubmitted repaired chain
  `856290` -> `856293`. Heartbeat
  `sallm-mamba-c3-afrihg-focus-watch` now watches the repaired chain.

Results:

- Completed as repaired serial L40S chain `856290` -> `856293`.
- Artifacts pulled locally to
  `outputs/eval/diagnostics/c3_afrihg_focus_prompt_val_20260521/`.
- Mamba validation results versus the prior Mamba validation beam5/lp1.2
  comparators from 2026-05-19:

| Task | Prior val chrF / ROUGE-L | Focus val chrF / ROUGE-L | Delta |
| --- | ---: | ---: | ---: |
| AfriHG Xho ckpt-656 | `12.8140` / `0.0434` | `13.1258` / `0.0439` | chrF `+0.3118`, ROUGE-L `+0.0005` |
| AfriHG Zul ckpt-892 | `17.0020` / `0.0839` | `18.4924` / `0.0908` | chrF `+1.4904`, ROUGE-L `+0.0069` |

- Matched focus-prompt LLaMA controls:
  - Xho chrF `13.2699`, ROUGE-L `0.0561`;
  - Zul chrF `19.2154`, ROUGE-L `0.0893`.
- The LLaMA focus controls are useful but not enough to call this a shared
  protocol improvement, because the pulled notes do not include a matched LLaMA
  non-focus validation baseline under the same split and decode setting.
- First-example spot checks still show unresolved focus errors:
  - Mamba Xho predicted `Ubhubhane uNothemba ubhubhile` for reference
    `Ubekwa kwikhaya lokugqibela uAunt Laura Mphahlwa`;
  - Mamba Zul predicted `Kukhuthaza eyokuceltic` for reference
    `Banohlelo lwentsha namakhondomu`.

Decision:

- Complete. C3 is a positive Mamba validation candidate, strongest on Zul, but
  not a final recipe until it survives official test confirmation.
- Carry-forward tag: `Mamba validation candidate / shared status ambiguous`.
- Do not carry the focus prompt to final LLaMA comparisons unless a matched
  LLaMA validation baseline proves it also helps LLaMA.
- Next defensible AfriHG step: official Mamba test rerun with focus prompt,
  beam5/lp1.2, pinned/conservative generation settings, and documented
  cache/batch behavior.

### [x] C4. AfriHG Focus Prompt Official Test Confirmation

Question: does the C3 validation focus-prompt gain survive on the official
AfriHG test split?

Hypothesis:

- If C3 is a real prompt/formulation improvement rather than validation noise,
  it should improve or at least preserve the C1 official test beam5/lp1.2
  metrics for both AfriHG languages.

Design:

- Mamba-only official test confirmation.
- Use current checkpoint-selected Mamba AfriHG checkpoints:
  - Xho checkpoint-656;
  - Zul checkpoint-892.
- Keep beam5/lp1.2 and `early_stopping=false`.
- Change the prompt template to `afrihg_headline/focus_v1`.
- Keep the C3 explicit `max_batch_size=8` and document the E2 cache/batch
  caveat when interpreting results.

Success criteria:

- Positive: official test chrF and ROUGE-L improve over C1 for both Xho and
  Zul, or improve one language clearly without harming the other.
- Ambiguous: small mixed changes or metric gain with qualitatively worse focus.
- Negative: official test metrics regress or examples show worse generic/wrong
  focus despite validation gain.

Observations:

- 2026-05-22 11:26 SAST: added
  `scripts/submit_c4_mamba_afrihg_focus_prompt_test_2026_05_22.sh`; `bash -n`
  passes.
- Started HEX pass with `/scratch/slurm/bin/purequota`; scratch was `89.3%`.
  Checked top consumers before submission: checkpoints `56G`, results `15G`,
  logs `93M`.
- First rsync flattened the C4 script into the remote repo root; corrected with
  targeted syncs into `scripts/` and `src/conf/templates/afrihg_headline/`.
  No deletion was performed.
- Submitted jobs:
  - `856306` Mamba AfriHG Xho focus official test;
  - `856307` Mamba AfriHG Zul focus official test, dependency
    `afterok:856306`.
- Created heartbeat `sallm-c4-mamba-afrihg-focus-test-watch`.

Results:

- Completed successfully:
  - `856306` Xho: `COMPLETED`, `0:0`, `00:02:44`;
  - `856307` Zul: `COMPLETED`, `0:0`, `00:02:54`.
- Artifacts pulled to
  `outputs/eval/diagnostics/c4_mamba_afrihg_focus_prompt_test_20260522/`.
- Local summary written to
  `outputs/eval/diagnostics/c4_mamba_afrihg_focus_prompt_test_20260522/c4_summary.md`.

| Task | C1 beam5/lp1.2 chrF | C4 focus_v1 chrF | Delta chrF | C1 ROUGE-L | C4 ROUGE-L | Delta ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Xho | `15.1702` | `14.4983` | `-0.6720` | `0.0572` | `0.0553` | `-0.0019` |
| Zul | `17.1245` | `17.3358` | `+0.2112` | `0.0736` | `0.0782` | `+0.0045` |

- Qualitative spot-check still shows wrong/generic focus:
  - Xho `Ikhulule` versus reference `Ingxubakaxaka ngo-AB de Villiers`;
  - Xho `Zwelonke Zwelonke: Zwelonke Sigcawu!` versus reference
    `Thabo Mbeki: Siziva siziinkedama!`;
  - Zul `U-] Izinkunziqu zikamama kwi-FirstMan]` versus reference
    `I First Man ngeyabathanda ukwazi ngomhlaba`.

Decision:

- Complete. C4 is ambiguous-to-negative as a final recipe gate.
- Do not adopt `focus_v1` as the final AfriHG prompt recipe because Xho
  official test regresses and Zul gains are small.
- Keep C1 beam5/lp1.2 as the current default Mamba AfriHG official-test recipe.
- Carry-forward tag: `Validation-only candidate / language-specific possible
  prompt idea / not final`.
- Root-cause implication: clearer prompting can help validation but does not
  fully fix AfriHG headline focus; residual wrong-focus/generic headlines likely
  need model/task learning improvements, not just a stricter prompt.

### [x] E1. HF-vs-Official Mamba Logits Parity

Question: is the HF Mamba implementation equivalent enough to official
`mamba_ssm` for our checkpoints?

Hypothesis:

- Some instability may come from implementation/caching/eval path quirks rather
  than Mamba quality itself.

Planned design:

- Load a representative Mamba checkpoint in both paths if feasible.
- Same tokenizer, dtype, prompt, no generation cache.
- Compare next-token logits/top-k tokens.

Metrics:

- Mean/max logit delta.
- Top-k overlap.
- Same greedy next token rate.

Success criteria:

- Close enough parity to remove implementation path as a major concern.

Observations:

- 2026-05-21 22:55 SAST: prepared but did not submit E1 because C1b/D1 were
  active and scratch was above `85%`.
- 2026-05-21 23:01 SAST: C1b is now closed, but D1 is still active and scratch
  remains above `85%`, so E1 stays prepared only.
- Existing script `scripts/run_mamba_ssm_logits_parity_probe.py` was tightened
  to report:
  - max/mean absolute logit deltas;
  - last-token deltas;
  - same greedy next-token flag;
  - top-k next-token overlap and decoded top tokens.
- Added ready-to-submit wrapper
  `scripts/submit_e1_mamba_hf_official_logits_parity_2026_05_21.sh`.
  It runs a small prompt set covering AfriHG-like headline text, T2X-like
  triple verbalization, and tag tokens.
- Local validation passed:
  - `uv run ruff check scripts/run_mamba_ssm_logits_parity_probe.py`;
  - `uv run python scripts/run_mamba_ssm_logits_parity_probe.py --help`;
  - `bash -n scripts/submit_e1_mamba_hf_official_logits_parity_2026_05_21.sh`.
- 2026-05-21 23:48 SAST: revalidated E1 against the current worktree:
  - `python3 -m py_compile scripts/run_mamba_ssm_logits_parity_probe.py`;
  - `uv run ruff check scripts/run_mamba_ssm_logits_parity_probe.py`;
  - `bash -n scripts/submit_e1_mamba_hf_official_logits_parity_2026_05_21.sh`.
  All passed. Still not submitted while D1 is active and scratch is above
  `85%`.
- 2026-05-22 01:32 SAST: revalidated E1 locally while D1 continued running:
  - `bash -n` still passes for the prepared submitter;
  - `uv run python scripts/run_mamba_ssm_logits_parity_probe.py --help` loads
    and shows the expected `--checkpoint`, `--output`, `--prompt`, `--top-k`,
    and `--device` interface;
  - `uv run ruff check scripts/run_mamba_ssm_logits_parity_probe.py` passes.
- 2026-05-22 10:48 SAST: revalidated E1 again after D1 closed negative:
  `bash -n`, `python3 -m py_compile`, and Ruff all pass. Queue is empty, but
  scratch remains `89.3%`, so E1 should be submitted only with explicit
  approval as the next low-footprint gate.
- Older local artifact
  `outputs/eval/diagnostics/mamba_ssm_logits_parity_probe_2026-05-02/report.json`
  already showed large HF-vs-official logit deltas for the public Mamba base,
  but the tightened E1 rerun is still needed for current prompts/top-k details
  before using implementation parity as advisor-facing evidence.
- 2026-05-22 10:52 SAST: submitted E1 as job `856253` after quota-first check
  showed scratch `89.3%` and an empty queue. Job completed successfully in
  `00:00:22`; artifact pulled to
  `outputs/eval/diagnostics/e1_mamba_hf_official_logits_parity_20260521/report.json`.

Results:

- E1 fails close implementation parity under the current simple HF-to-official
  mapping:
  - parameter counts match (`126427168` vs `126427168`);
  - state load reports no missing or unexpected keys;
  - `all_logits_close_at_1e-4=false`;
  - `all_same_greedy_next_token=false`;
  - minimum top-10 overlap is `4/10`;
  - max absolute logit deltas are `16.0124` for an AfriHG-like headline prompt,
    `9.3086` for a T2X-like triple prompt, and `12.5421` for a NER-tag prompt;
  - greedy next token differs on the AfriHG-like and NER-tag prompts.
- Older 2026-05-02 config-variant evidence already found that matching obvious
  non-weight knobs did not close the gap, so this is not explained by the
  basic checked config flags.

Decision:

- Complete. Carry-forward tag: `Negative` for implementation parity, but
  important caveat for final claims.
- Current downstream evidence remains valid for the HF-Mamba path actually used
  in SALLM experiments, but E1 blocks a broad claim that the failures are
  inherent to every official Mamba2 implementation.
- Continue B3/C3 as practical HF-Mamba rescue gates, and phrase advisor-facing
  architecture claims as "HF pure-Mamba2 implementation/config path remains
  weak or fragile" unless a later official-path parity/rescue experiment closes
  this gap.

### [x] E2. Mamba Generation Parity Smoke Test

Question: do cache/fallback/batch settings change generated outputs?

Hypothesis:

- If output changes across safe settings, final evaluation must pin one safe
  setting and document why.

Planned design:

- Compare short greedy generation under:
  - cache on/off where possible;
  - fast path/fallback where possible;
  - batch size 1 vs small batch.

Metrics:

- Exact output match.
- Qualitative differences.
- Runtime/memory.

Success criteria:

- Outputs are stable enough that evaluation settings are not confounding
  quality.

Observations:

- 2026-05-21: added `scripts/run_mamba_generation_parity_smoke.py`.
  - Representative Mamba cases:
    - current T2X checkpoint-selected/continued checkpoint;
    - current AfriHG Xho checkpoint-656.
  - Greedy deterministic generation on validation samples.
  - Variants:
    - batch size 1, cache off;
    - batch size 1, cache on;
    - batch size 4, cache off;
    - batch size 4, cache on.
  - Compares exact generated text against batch1/cache-off baseline and records
    any variant errors.
- Added `scripts/submit_e2_mamba_generation_parity_smoke_2026_05_21.sh`.
  - One L40S GPU.
  - `max_samples=8`, `max_new_tokens=64`.
- Local validation:
  - `uv run ruff check scripts/run_mamba_generation_parity_smoke.py` passed;
  - `uv run python scripts/run_mamba_generation_parity_smoke.py --help`
    passed;
  - `bash -n scripts/submit_e2_mamba_generation_parity_smoke_2026_05_21.sh`
    passed.
- Submitted as job `855312` after quota/top-consumer check:
  - scratch `86.9%`;
  - top consumers checkpoints `54G`, results `15G`, logs `91M`;
  - immediate log tail showed Mamba CUDA kernels available and the intended
    command running.
- Job `855312` completed successfully in `00:00:41`; artifacts were pulled to
  `outputs/eval/diagnostics/e2_mamba_generation_parity_smoke_20260521/`.

Results:

- No variant crashed; all cache/batch variants produced outputs.
- T2X, 8 validation examples:
  - batch1/cache-off baseline: `8/8` self-match;
  - batch1/cache-on: `7/8` exact match to baseline;
  - batch4/cache-off: `7/8` exact match to baseline;
  - batch4/cache-on: `7/8` exact match to baseline.
  - First difference: `I-Binignit ivela kwingingqi ye-visayas.` vs
    `I-Binignit ivela kwi-visayas.`
- AfriHG Xho, 8 validation examples:
  - batch1/cache-off baseline: `8/8` self-match;
  - batch1/cache-on: `6/8` exact match to baseline;
  - batch4/cache-off: `7/8` exact match to baseline;
  - batch4/cache-on: `6/8` exact match to baseline.
  - Differences include semantically visible changes such as
    `Ubhubhane uNothemba uncedeza` vs
    `Ubhubhane uNothemba uncede uAunt Laura`.

Decision:

- Complete. E2 does not show a crash or unusable implementation path, but it
  does show Mamba generation is not perfectly invariant to cache/batch
  settings.
- This is an implementation-hygiene finding: final Mamba generation results
  must pin and document cache/batch settings. Prefer a conservative
  batch1/cache-off or explicitly documented harness setting for final reruns.
- Carry-forward tag: `Negative` as a rescue, but important final-evaluation
  hygiene. Do not interpret small Mamba generation metric differences without
  checking whether evaluation settings changed.

---

## Phase 3: Base Recipe Decisions After D1 Completes

### D1 Completion Branch

Use this branch when `854988`/`854989` changes state.

1. Start with quota and scratch pressure:
   - run `/scratch/slurm/bin/purequota` first;
   - if scratch remains above `85%`, inspect top consumers;
   - if scratch approaches or exceeds `90%`, stop new submissions and ask for
     cleanup approval rather than adding more checkpoint pressure.
2. If `854988` fails:
   - pull and record the relevant log tail;
   - do not run or resubmit `854989` unless the failure is clearly a repairable
     implementation issue;
   - keep B3/C3/E1 as the next low-footprint gates only after scratch is safe.
3. If `854988` completes but `854989` is still pending/running:
   - pull only lightweight summaries/trainer states first;
   - wait for the clean generation-loss audit before declaring D1 positive or
     negative unless checkpoint-15000/20000 is already decisive.
4. If `854989` completes:
   - pull `fresh_pretrain_summary.json`, checkpoint trainer states, and clean
     generation-loss diagnostics locally;
   - summarize locally, then update this checklist, the daily note, and
     `sallm_progress.md` if the high-level story changes.
5. Decision branches:
   - **D1 positive:** if held-out eval and clean generation-loss both improve
     enough to be credible, mark D1 positive and run D3 first as a small
     downstream transfer probe from the best new base candidate. Hold D2 until
     D3 shows whether base improvement transfers to task metrics.
   - **D1 ambiguous:** if pretraining eval improves but clean generation-loss
     does not, treat base quality as not proven. Run E1 implementation parity
     and the B3/C3 formulation gates before spending on a D2 base HPO matrix.
   - **D1 negative:** if it does not improve over current/previous Mamba base
     evidence, mark D1 negative and prioritize E1 plus B3/C3, then design D2
     only if scratch and compute are acceptable.
   - 2026-05-22: D1 is negative by this rule. The best fresh-wide clean-loss
     PPLs remain far worse than current Mamba base and LLaMA base, so D3 is
     skipped and E1/B3/C3 become the next gates once scratch/user approval is
     safe.
6. Next non-D1 gate priority after D1 closes and scratch is safe:
   - E1 first if Mamba implementation parity remains a serious interpretability
     confounder;
   - B3 next for T2X source preservation;
   - C3 next for AfriHG focus prompting;
   - D3 before D2 only when D1 yields a plausible new base candidate.

### [x] D2. Pure-Mamba Base HPO Matrix

Question: if the wide base gate is not enough, what small HPO matrix can find a
better pure-Mamba base recipe?

Hypothesis:

- The current base issue may be recipe/shape, not an inherent pure-Mamba limit.

Planned design:

- Only run after D1 result is known.
- Candidate axes:
  - shape: current `expand=4/state64`, wide `expand=2/state128`, maybe one
    intermediate matched-parameter shape;
  - LR: `2e-4`, `4e-4`, maybe `6e-4` if stable;
  - warmup: `1000`, maybe `2000`;
  - horizon: screen at `5k` or `10k`, continue only promising arms.

Metrics:

- Eval-loss trajectory.
- Clean generation-loss audit.
- GPU hours.
- Scratch footprint.

Success criteria:

- At least one arm beats the current Mamba base trajectory or materially
  improves clean downstream generation loss.

Observations:

- Not started. Do not design or submit D2 until D1 is classified as positive,
  ambiguous, or negative with clean generation-loss evidence.
- If D1 is positive, run D3 first before a broader D2 matrix. If D1 is
  ambiguous or negative, use E1/B3/C3 to reduce implementation/formulation
  uncertainty before spending on D2.
- 2026-05-22 11:41 SAST: D1 is negative and E1/B3/C3/C4 have now closed, so D2
  design is justified. Scratch is `89.4%`, so no D2 job was submitted.
- Added one-arm submitter:
  `scripts/submit_d2_mamba_base_hpo_screen_2026_05_22.sh`.
- Local checks passed:
  - `bash -n scripts/submit_d2_mamba_base_hpo_screen_2026_05_22.sh`;
  - usage output when no arm is supplied;
  - script is executable.
- Candidate arms:
  - `current_lr2e4_wu2000_10k`: current expand4/state64 shape, lower LR,
    longer warmup;
  - `wide_lr2e4_wu2000_10k`: wide expand2/state128 shape, lower LR, longer
    warmup;
  - `wide_lr1e4_wu1000_10k`: wide expand2/state128 shape, conservative LR.
- Each arm is serial: 20-step canary -> 10k train screen -> clean generation
  loss audit on T2X Xho, AfriHG Xho, and AfriHG Zul validation prompts.
- Scratch cleanup candidates were identified but not deleted:
  - `checkpoints/base_hpo/mamba_fresh_base_probe_20260519`: `3.4G`;
  - `checkpoints/base_hpo/mamba_fresh_base_wide_e2_s128_torcheval3_20k_20260521`:
    `3.1G`;
  - `checkpoints/base_hpo/mamba_fresh_base_lr4e4_20k_20260520`: `3.1G`;
  - `checkpoints/base_hpo/mamba_fresh_base_wide_e2_s128_torcheval3_20k_20260521_canary`:
    `977M`.
- 2026-05-22 11:50 SAST: queue is empty and scratch remains `89.4%`. Cleanup
  audit confirms the three large base-HPO scientific runs have local summaries,
  clean-loss rows, and vault decisions already recorded. The canary is
  canary-only. Deleting those four exact checkpoint dirs would free roughly
  `10.6G`, but deletion still requires explicit user approval.
- 2026-05-22 11:58 SAST: reduced D2 scratch footprint before any submission.
  Added `--skip-final-save` to `scripts/run_mamba_fresh_pretrain_streaming.py`
  and patched D2 to use it for canary/train jobs. D2 clean-loss audit now uses
  only `checkpoint-5000` and `checkpoint-10000` plus current Mamba/LLaMA base
  controls, avoiding an extra duplicated `final_model` copy. Validation passed:
  `bash -n` on D2 submitter, `py_compile`, `ruff` on the Python runner, and
  `uv run python ... --help` shows `--skip-final-save`.
- 2026-05-22 12:04 SAST: synced the lower-footprint D2 runner/submitter to HEX.
  Remote `bash -n` and `py_compile` pass, and remote files contain
  `--skip-final-save`. A first rsync flattened files into the remote repo root;
  corrected with targeted syncs into `scripts/`. No deletion and no submission.
- 2026-05-22 12:15 SAST: added `--dry-run` support to the D2 submitter and
  synced it to HEX. Remote dry run for `current_lr2e4_wu2000_10k` prints the
  exact planned canary/train/clean-loss paths and submits no jobs. Added
  `sallm_memory/mamba_root_cause_rescue_evidence_matrix.md` as the current
  advisor-ready evidence/status matrix.
- 2026-05-22 12:26 SAST: user approved cleanup. Deleted only the four audited
  old base-HPO checkpoint dirs, freeing enough quota for D2. After quota
  refresh, `/scratch` dropped from `89.4%` to about `79%`.
- Repaired the D2 scratch guard to parse `/scratch/slurm/bin/purequota` instead
  of filesystem `df`; forced-threshold test now refuses before submission.
- During guard testing, the earlier bad guard briefly submitted `856315` ->
  `856317`; all were cancelled immediately, and no D2 partial output dirs were
  found.
- Submitted first D2 arm `current_lr2e4_wu2000_10k` as `856320` -> `856322`.
  Created heartbeat `sallm-d2-mamba-base-hpo-screen-watch`.
- Latest light HEX status: `/scratch` `79.6%`; `856320` completed `0:0`;
  `856321` is running on `srvrocgpu012`; `856322` is dependency-pending.
- 2026-05-22 13:51 SAST: `checkpoint-5000/trainer_state.json` exists.
  Trainer metrics are poor: train loss `6.3609`, eval loss `6.5037`,
  eval PPL about `667.6`. This is worse than D1 wide checkpoint-5000 eval loss
  (`5.6133`), so the first D2 arm has a negative early signal. Wait for
  checkpoint-10000 and clean-loss audit before final classification.
- 2026-05-22 15:51 SAST: `856321` completed `0:0` and `856322` clean-loss audit
  is running. Checkpoint-10000 remains weak: train loss `6.2177`, eval loss
  `6.4936`, eval PPL about `660.9`. It improves only slightly from checkpoint
  5000 and remains far too high.
- 2026-05-22 16:23 SAST: `856322` completed `0:0`; artifacts pulled and
  summarized locally in
  `outputs/eval/diagnostics/d2_mamba_base_hpo_screen_current_lr2e4_wu2000_10k_20260522/base_gate_summary.md`.
- Clean generation-loss confirms the arm is negative. Checkpoint-10000 PPLs:
  T2X Xho `11925.1`, AfriHG Xho `26758.1`, AfriHG Zul `28854.2`. Current
  Mamba base PPLs on the same audits are `347.8`, `438.5`, and `605.9`;
  LLaMA base PPLs are `64.0`, `205.6`, and `267.5`.
- 2026-05-24 00:04 SAST: submitted second D2 arm
  `wide_lr2e4_wu2000_10k`, because the current-shape LR/warmup rescue failed
  and the next defensible base screen should test shape/architecture recipe.
  Local and remote syntax/py_compile/dry-run checks passed. Scratch was
  `81.0%` at submission.
- Job chain for the wide arm:
  - `861386`: canary;
  - `861387`: 10k train, dependency `afterok:861386`;
  - `861388`: clean generation-loss audit, dependency `afterok:861387`.
- 2026-05-24 04:06 SAST: D2 wide arm completed. Jobs `861386` -> `861388`
  all completed `0:0`; scratch was `83.2%`, so no cleanup was needed.
  Artifacts were pulled and summarized locally under
  `outputs/eval/diagnostics/d2_mamba_base_hpo_screen_wide_lr2e4_wu2000_10k_20260522/`.
  Summary:
  `base_gate_summary.md`.

Results:

- 2026-05-22: D1 closed negative. Do not use D1 as a base candidate.
- D2 first arm `current_lr2e4_wu2000_10k` is negative.
- D2 second arm `wide_lr2e4_wu2000_10k` is negative:
  - trainer eval improved from checkpoint-5000 `6.5252` to checkpoint-10000
    `6.4619`;
  - clean generation loss still strongly rejects both checkpoints;
  - clean-loss PPLs for checkpoint-5000 are T2X Xho `9983.6`, AfriHG Xho
    `21768.7`, AfriHG Zul `20378.4`;
  - clean-loss PPLs for checkpoint-10000 are T2X Xho `11783.7`, AfriHG Xho
    `25461.3`, AfriHG Zul `21620.8`;
  - current Mamba base remains far lower at `347.8`, `438.5`, `605.9`, and
    LLaMA base lower still at `64.0`, `205.6`, `267.5`.

Decision:

- D2 first arm and D2 second arm are both closed negative.
- Use a smaller, discriminative one-arm-at-a-time screen rather than blindly
  extending D1.
- Lower LR/longer warmup on the current expand4/state64 shape is not enough.
- Do not submit D3 from this base.
- The wide shape/architecture recipe plus lower LR/warmup did not produce a
  credible base candidate.
- Do not submit D3 from either D2 arm.
- Recommendation: stop pure-Mamba base HPO here and report the base-rescue path
  as negative. The remaining conservative wide arm (`wide_lr1e4_wu1000_10k`) is
  optional only if an exhaustive ablation table is worth another run; the
  completed D2 arms are still orders of magnitude worse than current Mamba base
  and LLaMA base on clean generation loss, so the expected scientific value is
  low.

### [x] D3. Downstream Probe From Best New Base Candidate

Question: if D1 or D2 produces a better base, does downstream fine-tuning close
the actual task gaps?

Hypothesis:

- Better base likelihood should translate into better T2X/AfriHG and possibly
  better POS/NER label scoring.

Planned design:

- Fine-tune only the best new base candidate on small representative gates:
  - T2X Xho;
  - AfriHG Xho/Zul;
  - POS/NER label-scoring formulation if A1 succeeds.
- Keep run count small until we see downstream transfer.

Metrics:

- T2X chrF plus entity/value preservation.
- AfriHG chrF plus length/hook preservation.
- POS token accuracy.
- NER F1/non-O recall.

Success criteria:

- Improvements appear on at least one generation task and do not regress
  output-shape diagnostics.

Observations:

- 2026-05-22: prepared but did not submit
  `scripts/submit_d3_mamba_downstream_transfer_probe_2026_05_22.sh`.
  The script is executable and passes `bash -n`.
- The script is gated on `D3_BASE_CANDIDATE_PATH`, so it cannot accidentally
  reuse the old public Mamba base. It should only be run after D1/D2 selects a
  credible base candidate and scratch is safe. It is currently local-only, so
  rsync the repo to HEX before submitting it.
- Planned D3 jobs are serial L40S one-GPU fine-tune/eval pairs for T2X Xho,
  AfriHG Xho, and AfriHG Zul on `validation` by default. AfriHG uses the
  current Mamba-only beam5/length-penalty rescue settings; T2X pins batch size
  to 1 because E2 showed Mamba generation batch/cache sensitivity.
- Zul uses a `+eval.eval_model.peft_adapter=...` override because its eval YAML
  does not currently define `peft_adapter`; Xho/T2X use the normal
  `eval.eval_model.peft_adapter=...` override because those YAMLs already
  define it.
- 2026-05-22 00:32 SAST: confirmed C3 and E1 submit scripts are executable;
  B3 and D3 were already executable. All four prepared submit scripts have
  passed `bash -n` locally.
- 2026-05-22 00:34 SAST: checked the prepared submitters for GPU/partition
  hygiene. B3, C3, and E1 explicitly submit to `--partition=l40s` with
  `--gres=gpu:l40s:1`; D3 submits through `scripts/launch_finetune.sh` and
  `scripts/launch_evaluation.sh`, whose SBATCH headers also reserve
  `--partition=l40s` and `--gres=gpu:l40s:1`. This satisfies the current
  one-GPU/L40S constraint for the prepared post-D1 gates.
- 2026-05-22 01:29 SAST: re-audited D3 overrides against the actual eval YAMLs.
  `run_mamba_afrihg_zul.yaml` does not define `eval_model.peft_adapter`, while
  T2X and AfriHG Xho do, so the Zul `+eval.eval_model.peft_adapter=...`
  override is intentional and should remain. `bash -n` still passes after the
  audit; no code change was kept.
- 2026-05-22 01:32 SAST: confirmed the post-D1 readiness audit did not leave a
  local code diff in the D3 submitter.
- 2026-05-22 01:34 SAST: refreshed the active
  `sallm-mamba-wide-torch-eval-gate-watch` heartbeat so it now reflects the
  current D1 mixed-signal state, checkpoint-15000 metrics, scratch `88.3%`,
  and the no-submit-until-D1-closes/scratch-safe constraint.

Results:

- No D3 result. D1 and both completed D2 arms did not produce a credible base
  candidate, so the downstream probe was intentionally not submitted.

Decision:

- Do not submit D3 after D1 or the completed D2 arms. No credible D1/D2 base
  candidate exists.
- Keep the submitter gated on `D3_BASE_CANDIDATE_PATH`; it is a prepared
  contingency, not part of the current completed rescue checklist.
- Base-model follow-up decision: stop the current pure-Mamba base rescue branch
  unless the user explicitly wants one final exhaustive conservative-wide
  ablation.

---

## [x] D4. POS/NER Full-Data Atomic Tag-Sequence Rescue

Question: do POS/NER recover if we keep the pure decoder-only Mamba base fixed,
but give the atomic tag-sequence formulation full task data instead of the
128-example canary?

Hypothesis:

- The earlier POS/NER failures are not only base-model quality failures. Some
  of the weakness may be task-adaptation/formulation limited, especially
  because the atomic NER canary found a narrow useful signal while normal
  free-generation and constrained output-shape-only routes failed.

Design:

- Keep architecture fixed: `anrilombard/sallm-mamba-125m`, Mamba2 LoRA,
  decoder-only generation.
- Keep atomic POS/NER label tokens and modules-to-save from the prior canaries.
- Remove the 128-example train/val caps and train on the full Xho task data.
- Evaluate on the existing capped train/validation atomic lm-eval task packs.
- Run only the highest-information recipes:
  - POS high-signal: `lr3e-4`, `80` epochs.
  - POS steadier midpoint: `lr2e-4`, `120` epochs.
  - NER best canary window: `lr1e-4`, `60` epochs.

Metrics:

- POS token accuracy, strict token accuracy, exact length, nonempty rate, and
  overgeneration rate.
- NER token accuracy, strict token accuracy, exact length, nonempty rate,
  non-`O` recall, all-`O` rate, and BIO/entity F1.
- Compare against the prior atomic canary baselines rather than the generic
  free-generation baselines.

Success criteria:

- POS: validation token/strict accuracy improves materially over prior atomic
  canaries without severe overgeneration or empty output.
- NER: validation token accuracy/entity F1/non-`O` recall improves over the
  `lr1e-4/e60` canary without increasing all-`O` collapse.
- If train improves but validation does not, classify as ambiguous
  memorization/formulation evidence, not a rescue.

Observations:

- 2026-05-24 04:23 SAST: added and synced
  `scripts/submit_d4_mamba_pos_ner_fulldata_atomic_rescue_2026_05_24.sh`.
  Local `bash -n` passed.
- 2026-05-24 04:23 SAST: started HEX submission pass with
  `/scratch/slurm/bin/purequota`; `/scratch` was `83.2%`, so no cleanup was
  needed.
- 2026-05-24 04:23 SAST: submitted L40S one-GPU train/eval pairs:
  - `861509` -> `861510`: POS full-data atomic `lr3e-4/e80`.
  - `861511` -> `861512`: POS full-data atomic `lr2e-4/e120`.
  - `861513` -> `861514`: NER full-data atomic `lr1e-4/e60`.
- 2026-05-24 04:24 SAST: queue check showed all three train jobs running and
  the three eval jobs dependency-pending.
- 2026-05-24 04:25 SAST: watcher
  `sallm-d4-pos-ner-fulldata-rescue-watch` created.
- 2026-05-24 09:41 SAST: early health check still has `/scratch` at `83.2%`.
  Jobs `861509`, `861511`, and `861513` are running; eval jobs `861510`,
  `861512`, and `861514` remain dependency-pending. Lightweight log tails show
  the train jobs are stepping and producing teacher-forced learning signal, but
  no validation decision is available yet.
- 2026-05-24 09:42 SAST: repeated quota-first check still shows `/scratch`
  `83.2%`. All train jobs are still running and all eval jobs remain
  dependency-pending. POS logs have reached roughly epoch `5-6`, and NER logs
  roughly epoch `3`; no final artifacts are ready to pull.
- 2026-05-24 09:43 SAST: quota-first check shows `/scratch` at `84.7%`, below
  the `85%` cleanup threshold, so no cleanup was performed. Train jobs
  `861509`, `861511`, and `861513` are still running; eval jobs remain
  dependency-pending. POS logs have reached roughly epoch `7-8`, and NER logs
  roughly epoch `4`; still no validation artifacts.
- 2026-05-24 09:45 SAST: quota-first check crossed the cleanup threshold at
  `/scratch` `85.1%`. Inspected top consumers before cleanup. Deleted only
  completed, summarized, negative D2 base-HPO checkpoint directories:
  `d2_mamba_base_hpo_screen_wide_lr2e4_wu2000_10k_20260522`,
  `d2_mamba_base_hpo_screen_current_lr2e4_wu2000_10k_20260522`,
  `d2_mamba_base_hpo_screen_wide_lr2e4_wu2000_10k_20260522_canary`, and
  `d2_mamba_base_hpo_screen_current_lr2e4_wu2000_10k_20260522_canary`.
  `checkpoints/base_hpo` dropped from `4.3G` to `0`; total checkpoints dropped
  from `52G` to `47G`. Immediate `purequota` still reported `85.1%`, likely
  due to refresh lag. Active D4 artifacts were not touched.
- 2026-05-24 09:46 SAST: quota refresh confirmed `/scratch` down to `80.5%`.
  D4 train jobs are still running, eval jobs still dependency-pending, and each
  active D4 arm has reached at least `checkpoint-1500`.
- 2026-05-24 09:47 SAST: quota-first check still shows `/scratch` `80.5%`.
  Train jobs `861509`, `861511`, and `861513` are still running; eval jobs
  `861510`, `861512`, and `861514` remain dependency-pending. No final eval
  artifacts are ready to pull.
- 2026-05-24 09:50 SAST: quota remains `80.5%`; no cleanup needed.
  Train jobs are still running and light log tails show healthy train-side
  learning: POS arms around epoch `22`, NER around epoch `11`. Eval jobs remain
  dependency-pending, so no D4 classification yet.
- 2026-05-24 09:51 SAST: quota-first check still shows `/scratch` `80.5%`.
  Train jobs `861509`, `861511`, and `861513` remain running; eval jobs
  `861510`, `861512`, and `861514` remain dependency-pending. No artifacts are
  ready to pull.
- 2026-05-24 09:51 SAST: repeated quota-first check still shows `/scratch`
  `80.5%`; train jobs remain running and eval jobs remain dependency-pending.
  No pullable D4 eval artifacts yet.
- 2026-05-24 09:52 SAST: quota-first check still shows `/scratch` `80.5%`;
  train jobs remain running and eval jobs remain dependency-pending. No
  pullable D4 eval artifacts yet.
- 2026-05-24 09:53 SAST: quota-first check still shows `/scratch` `80.5%`;
  D4 remains in the train phase with eval jobs dependency-pending.
- 2026-05-24 09:53 SAST: repeated quota-first check still shows `/scratch`
  `80.5%`; train jobs remain running and eval jobs remain dependency-pending.
- 2026-05-24 09:54 SAST: quota-first check shows `/scratch` `81.0%`, still
  below cleanup threshold. D4 remains in train phase with eval jobs
  dependency-pending.
- 2026-05-24 10:36 SAST: quota-first check showed `/scratch` `81.7%`, so no
  cleanup was needed. Train jobs `861509`, `861511`, and `861513` completed
  successfully. Original eval jobs `861510`, `861512`, and `861514` failed
  with exit `1:0` after generation while writing lm-eval `results.json`:
  `TypeError: Object of type function is not JSON serializable`. Patched
  `_to_serializable` in `src/main/sallm/evaluation/lm_eval_runner.py`, validated
  with `py_compile` and a local sanitizer smoke test, added the eval-only
  resubmission script, synced the fix to HEX, and queued rerun eval jobs
  `861516`, `861517`, and `861518`. Updated and pulled the D4 manifest; updated
  the heartbeat to watch the rerun jobs.
- 2026-05-24 10:40 SAST: rerun eval jobs `861516`, `861517`, and `861518`
  completed `0:0`; artifacts and lightweight trainer states were pulled.
  Local tag-sequence summaries were generated under
  `outputs/eval/diagnostics/d4_mamba_pos_ner_fulldata_atomic_rescue_20260524/`.

Results:

- Negative.
- POS `lr3e-4/e80` validation: token accuracy `0.0970`, strict `0.0958`,
  exact length `0.0333`, nonempty `0.6933`, overgeneration `0.0108`.
- POS `lr2e-4/e120` validation: token accuracy `0.1960`, strict `0.1423`,
  exact length `0.0400`, nonempty `0.9933`, overgeneration `1.1486`.
- NER `lr1e-4/e60` validation: token accuracy `0.4252`, strict `0.3839`,
  exact length `0.0352`, nonempty `0.9609`, non-`O` recall `0.1772`,
  all-`O` rate `0.9375`, BIO entity F1 `0.1769`.

Decision:

- D4 does not rescue POS/NER. Full-data atomic adaptation does not beat the
  prior atomic POS baselines and does not materially improve the NER non-`O` /
  entity signal; all-`O` collapse remains severe. Treat current pure-Mamba
  POS/NER rescue attempts as exhausted unless a substantially different base or
  architecture is introduced.

---

## D5 - Mamba-2 Hybrid Base Screen

Status: `[x]` complete, `Negative`.

- Run: `d5_mamba2_hybrid_base_screen_hybrid_lr4e4_wu2000_10k_20260524`
  (`861602` canary -> `861603` train -> `861604` clean-loss audit).
- Model: `hybrid_126m_waleffe8attn`, `126357846` params, Waleffe-style
  sparse-attention Mamba-2 hybrid with 2 attention layers out of 24
  (8.3% attention).
- Training: LR `4e-4`, warmup `2000`, `10000` steps, best trainer checkpoint
  `checkpoint-10000`.
- Pretraining eval improved from `5.7042` at `checkpoint-5000` to `5.6123` at
  `checkpoint-10000`, better than recent pure-Mamba fresh HPO arms but still
  not decisive.
- Clean generation-loss rejects both checkpoints:
  - checkpoint-5000 PPLs: T2X Xho `11664.9717`, AfriHG Xho `14834.6155`,
    AfriHG Zul `15053.4398`;
  - checkpoint-10000 PPLs: T2X Xho `12364.6594`, AfriHG Xho `14020.4567`,
    AfriHG Zul `14634.7756`;
  - current Mamba base remains far lower at `347.7666`, `438.5023`,
    `605.8546`;
  - LLaMA base remains lower still at `64.0379`, `205.6206`, `267.5144`.
- Decision: do not use this hybrid base for D3 or final downstream runs. The
  hybrid improves the fresh-training loss shape relative to pure Mamba, but it
  does not solve the downstream base-likelihood problem.
- Carry-forward tag: `Negative`. Keep as architecture-follow-up evidence, not
  as a usable base recipe. Caveat: this is negative for the exact tested
  implementation and 10k-step budget; audit positional attention and block
  structure before treating it as a broad hybrid result.

---

## Running Log

Use this section to add brief chronological notes as experiments run.

- 2026-05-21 20:30 SAST: checklist created from proposed experiment design.
- 2026-05-22 10:19 SAST: D1 repaired clean-loss audit job `855987` is still
  running on `srvrocgpu012`; scratch is `89.3%`; `loss_rows.jsonl` is now
  non-empty (`520774` bytes), so the torch-forward audit repair appears to have
  passed the earlier Mamba2 stride crash and reached row writing. No
  `summary.json` yet; D1 remains undecided.
- 2026-05-22 10:43 SAST: D1 repaired clean-loss audit job `855987` completed
  `0:0`; artifacts were pulled and summarized locally. D1 is negative: best
  fresh-wide PPLs are still thousands to tens of thousands on T2X/AfriHG,
  versus current Mamba base in the hundreds and LLaMA base lower still. Next
  gates should be E1/B3/C3 after scratch/user approval; skip D3 for D1.
- 2026-05-22 10:48 SAST: revalidated E1/B3/C3 locally and checked HEX. Queue
  is empty, scratch remains `89.3%`, and no job was submitted. Recommend E1 as
  the next approved low-footprint gate, then B3/C3 after E1 or scratch cleanup.
- 2026-05-22 10:52 SAST: E1 submitted as job `856253`, completed `0:0`, and
  report pulled. E1 fails HF-vs-official logits parity under the current simple
  mapping despite matching parameter counts and clean state load. Proceed with
  B3/C3 as HF-Mamba rescue gates, but keep final architecture claims cautious.
- 2026-05-22 11:04 SAST: B3 completed after two structured-config repairs.
  Final job `856269` completed `0:0`; artifacts pulled. B3 is negative as a
  rescue: source-checklist prompt does not meaningfully improve Mamba and
  repetition controls reduce quality. T2X source preservation remains a real
  Mamba gap relative to LLaMA.
- 2026-05-22 11:08 SAST: C3 submitted. First chain failed on Hydra append
  syntax for `decoding.max_batch_size`; repaired and resubmitted as
  `856290` -> `856293`. Heartbeat active.
- 2026-05-24 15:20 SAST: after D5 closed negative, pivoted the next mainline
  architecture check to strict-125M HF xLSTM. The `hidden=736`, 12-layer
  xLSTM shape is now viable after installing `xlstm`/`mlstm-kernels`, with
  `126,901,952` params and passing forward/backward/save-load/tiny-overfit
  probes. The first non-streaming canary exposed scratch/cache materialization,
  not OOM; a streaming harness fixed that. Canary jobs `861745` and `861746`
  completed on one L40S without OOM, and the 10k xLSTM base screen is now
  `861747` with clean-loss audit `861748`.
