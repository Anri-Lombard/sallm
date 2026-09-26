# Pure-GDN adapter execution — 2026-08-07

- Implemented validation-only runner
  `scripts/run_pure_gdn_validation_trial.sh` for the preregistered NER, POS,
  SIB, Intent, T2X, AfriHG, and General three-point LR screens. It fixes the
  A100-40GB Slurm contract, canonical pure-GDN artifact, architecture-complete
  LoRA targets, validation metrics, output roots, early stopping, seed, and Hub
  disablement. It contains no held-out evaluation submission path.
- General uses `SALLM_DISABLE_TASK_METRICS=1` so selection runs loss-only and
  does not spend time on scientifically inapplicable mixed free-generation
  metrics. The shared factory default remains unchanged for every other arm.
- Shared `launch_finetune.sh` now recognizes GDN selected through a Hydra
  architecture override and performs its explicit fast-kernel check. It also
  correctly normalizes Hydra `++` add-or-override arguments required to reuse
  established task configs without copying seven YAML files.
- All 21 family/trial mappings dry-ran locally. NER and token-balanced General
  Hydra compositions passed with the intended pure-GDN identity, targets,
  metrics, validation prompt expansion, and Hub-disabled state. Bash syntax and
  `git diff --check` passed. Focused regression test passed (`1 passed`); three
  unrelated pre-existing iterable-dataset tests cannot execute in the macOS
  sandbox because PyTorch `torch_shm_manager` is denied shared-memory access.
- Fresh HEX pre-submit check: `/scratch` `100/300 GB` (`33.5%`), no owned jobs,
  and no completed pure-GDN adapter-screen artifact. No A100-80GB or L40S work
  is active or schedulable under the owner.
- No remote file changed and no job was submitted. Targeted HEX sync was denied
  at the external-write approval boundary. Explicit user approval is required
  before syncing these runner files to `hex:~/masters/sallm/`; only after that
  may the single SIB LR-0 A100-40GB validation canary be submitted.

- After explicit approval, the four validated files were synced to their
  relative paths under `hex:~/masters/sallm/`; their remote SHA-256 values
  matched local. Pre-submit HEX state was empty across owned GPU jobs with no
  SIB LR-0 adapter artifact; scratch was `100/300 GB` (`33.5%`).
- First canary `1189728` failed `1:0` after `00:00:50` at Hydra composition:
  `training.gradient_checkpointing` is absent from `llama_sib_all` and
  requires add-or-override syntax. It failed before model/data loading or any
  validation metric, so the preregistered identical retry rule applies.
- The minimal implementation correction changed that override to
  `++training.gradient_checkpointing=false`. Bash syntax, `git diff --check`,
  and a real SIB Hydra composition all passed locally. The corrected runner
  SHA-256 is
  `2333bd1bb798211650fdcadea3a5c8db08ae898fb3dbcd0e8cbcae89d24fbd88`.
- Corrected canary `1189730` is the sole owned job on one A100-40GB
  `gpu:ampere`. It passed the fast GatedDeltaNet kernel gate, loaded canonical
  `GatedDeltaNetForCausalLM` in BF16, attached exactly the ten frozen LoRA
  targets, exposed `4,649,088` trainable parameters, built SIB train/validation
  sets of `4,206/2,970` examples with all five validation prompts, and began
  training. W&B run `86xxffgr` reports finite step-40/50/60/90 loss
  `9.6734/9.0982/8.4799/6.1871`; step-90 gradient norm is finite at `25.6114`.
  Hub enable/push flags are all false; no held-out
  or test path and no traceback/OOM/NCCL marker is present. No additional
  validation trial was released during the canary gate. At the latest pass it
  was at step 96 after `00:06:57`, HEX scratch was `100/300 GB` (`33.6%`),
  and it was the only owned Slurm job. Kombuys remained read-only and idle:
  RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`.

- SIB LR-0 canary gate passed at epoch 1. Job `1189730` wrote
  `checkpoint-526`; `trainer_state.json` records finite validation loss
  `13.205979719065656`, finite frozen selection metric
  `eval_classification/all_f1=0.10182469859889215`, and that checkpoint as the
  current best. All six language partitions completed before training resumed
  into epoch 2. No held-out/test metric was touched.
- After duplicate and artifact checks, the two remaining SIB grid points were
  submitted individually into the open A100-40GB slots: LR-1 job `1189838`
  (`8e-5`, W&B `v2b6kqxe`) and LR-2 job `1189839` (`1.5e-4`, W&B `web3ll2q`).
  Both passed canonical pure-GDN/BF16/fast-kernel/ten-target startup, built the
  same `4,206/2,970` validation-only datasets, and reached finite step-20 loss
  and gradient norm: `9.8656/42.3403` and `9.6596/70.3637`, respectively.
  Together with `1189730`, these are exactly three owned A100-40GB jobs; no
  A100-80GB or L40S work is active. Scratch remains `100/300 GB` (`33.6%`).
  No sheet update is due during validation selection.

- First comparable SIB validation checkpoint is now available for all three
  grid points. Frozen macro-F1 is exactly tied at
  `0.10182469859889215` for LR-0/LR-1/LR-2 at epoch 1. Their epoch-1
  validation losses are `13.205979719065656`, `13.479149897411617`, and
  `13.106245889888468`, respectively, but loss is not the SIB selection
  metric and does not change the preregistered tie rule.
- LR-0 epoch 2 also completed with unchanged macro-F1
  `0.10182469859889215`; validation loss improved to `13.0279887087016`.
  This is one non-improving metric epoch under frozen patience-2, so the job
  correctly continues into epoch 3. At the latest pass LR-0/LR-1/LR-2 were at
  steps about `1320/740/737`, all with finite loss and gradients and no fault
  or policy marker. Scratch rose only to `101/300 GB` (`33.7%`). Do not select
  yet; wait for all three terminal states and apply macro-F1, then lower LR,
  then earlier checkpoint. No sheet or held-out action is due.

- SIB LR-0 job `1189730` completed cleanly `0:0` in `02:06:28` after the
  frozen patience rule stopped it at epoch 3. Macro-F1 was exactly
  `0.10182469859889215` at epochs 1, 2, and 3; validation loss was
  `13.205979719065656/13.0279887087016/13.381533268886784`. The restored best
  checkpoint is epoch-1 `checkpoint-526`. Final adapter/model and config
  SHA-256 are
  `c275ffa45b6ba53b3cb96f487bc3a4e81f3d49b13daca5a36bf631bd568b22f3`
  and `5dfa39aa4f48e79b3ad7a00f9f392703e246e461506b999d7c5156b0090ee961`;
  config verifies rank 16, alpha 32, dropout 0.05, all ten targets, and the
  canonical pure-GDN base path.
- After final-artifact and duplicate checks, the open slot was filled with the
  next preregistered missing validation trial: NER LR-0 job `1190665` (`3e-5`,
  W&B `1nscfvh3`). It passed A100-40GB, fast-GDN, canonical checkpoint, BF16,
  ten-target, Hub-disabled, and validation-only startup; datasets are
  `4,323/10,760` after complete validation prompt expansion. Step-10 loss
  `10.0914` and gradient norm `43.2608` are finite. SIB LR-1/LR-2 continue, so
  exactly three A100-40GB jobs remain active. Scratch is `101/300 GB` (`33.8%`).
  Do not promote SIB or touch held-out tests until the other two SIB runs end.

- SIB LR-1 `1189838` and LR-2 `1189839` completed cleanly `0:0` in
  `02:05:42` and `02:05:32`. Both retained macro-F1
  `0.10182469859889215` at epochs 1--3. Their validation-loss histories are
  LR-1 `13.4791498974/13.5307184804/13.4180159341` and LR-2
  `13.1062458899/13.0404007523/13.0272262008`; loss is not the selection
  metric. Final adapter SHA-256 values are
  `5d05e84567248296feed51ccbf7d0932b4719abf503630b8a92930c36547f59c`
  and `32ed34024ce52d597494576bf5307c6329f0d56f8146212b940bb56851f4ebbb`.
  Configs verify the canonical base and exact ten targets.
- The SIB family is now validation-frozen. All three LRs and all three epochs
  tied exactly on macro-F1, so the preregistered lower-LR then earlier-checkpoint
  rule selects LR `3e-5`, job `1189730`, epoch-1 `checkpoint-526`. No SIB
  held-out evaluation is allowed until every family winner is frozen.
- Released slots now run NER LR-1 `1191239` (`8e-5`, W&B `dwpm9nao`) and LR-2
  `1191240` (`1.5e-4`, W&B `9iuk2z87`) beside LR-0 `1190665`. Both new jobs
  passed the canonical pure-GDN/BF16/ten-target/fast-kernel, `4,323/10,760`
  validation-only dataset, Hub-disable, and finite-startup gates. Step-10
  loss/gradient norm is `10.0909/42.9461` and `10.0901/42.8108`.
- NER LR-0 completed epoch 1 at `checkpoint-541`: validation loss
  `12.874351620585502`, frozen span-F1 `0.0`, callback runtime `450.22 s`.
  It resumed epoch 2 cleanly. Exactly three A100-40GB jobs are active; HEX
  scratch is `101/300 GB` (`33.9%`). No held-out or sheet action is due.

- At 11:23 SAST, NER LR-0 `1190665` completed epoch 2 at
  `checkpoint-1082` with validation loss `13.48668861814591`; trainer state
  retains best metric `0.0` and epoch-1 `checkpoint-541`, so the frozen span-F1
  has not improved. NER LR-1 `1191239` and LR-2 `1191240` completed epoch 1 at
  `checkpoint-541` with validation losses `13.596421570051115` and
  `13.719133727346655`; both trainer states record best metric `0.0`. All three
  resumed training with finite loss and gradient norms, no traceback/OOM/NCCL
  marker, canonical pure-GDN/BF16/ten-target configuration, and no held-out or
  Hub path. They are the only owned jobs, all on A100-40GB `gpu:ampere`; HEX
  scratch is `102/300 GB` (`34.1%`). Estimated terminal window under the frozen
  patience rule is about 12:00 SAST for LR-0 and 12:45--13:00 for LR-1/LR-2.
  Kombuys is read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`). No sheet update or official-test action is due.

- NER LR-0 `1190665` completed cleanly `0:0` in `01:55:27`. Its epoch-3
  validation loss was `13.620804411593866`; frozen span-F1 remained `0.0`
  across epochs 1--3, so patience stopped the run and restored epoch-1
  `checkpoint-541`. The final validation-loss history is
  `12.874351620585502/13.48668861814591/13.620804411593866`. Final adapter and
  config SHA-256 values are
  `075bcea23fc0fd517c1ab0067d570f555aa1f016ad67087e2e00a8f574ea84cb`
  and `47635d24822a96bb6a327ffe9687d9cb0fe7a8120f9cfc8a1e0f5e513fe02768`;
  config verifies the canonical base and the frozen LoRA contract.
- NER LR-1 `1191239` and LR-2 `1191240` completed epoch 2 at
  `checkpoint-1082` with validation loss `13.534134671236059` and
  `13.491445094679367`. Both trainer states retain best metric `0.0` and
  epoch-1 `checkpoint-541`, then resumed epoch 3 cleanly.
- After verifying the NER LR-0 final artifact, no POS LR-0 completion, and no
  active duplicate, the open A100-40GB slot was filled by preregistered POS
  LR-0 job `1191712` (`3e-5`, W&B `366ujt2g`). It passed the fast-GDN gate,
  loaded canonical `GatedDeltaNetForCausalLM` in BF16, attached the exact ten
  LoRA targets with `4,649,088` trainable parameters, kept Hub paths disabled,
  built validation-only POS train/eval datasets of `2,259/2,250` after
  five-prompt validation expansion, and reached training step 27 without a
  fault marker. Exactly three A100-40GB jobs are active; no A100-80GB or L40S
  job, held-out evaluation, or sheet update overlaps. Scratch remains
  `102/300 GB` (`34.1%`).

- NER LR-1 `1191239` and LR-2 `1191240` completed cleanly `0:0` in
  `01:54:59` and `01:54:19`. Frozen span-F1 stayed `0.0` for all three epochs;
  their validation-loss histories are LR-1
  `13.596421570051115/13.534134671236059/13.591446909851301` and LR-2
  `13.719133727346655/13.491445094679367/13.734490807969332`. Final adapter
  SHA-256 values are
  `e3f02b9489226da5eaee1cd59e790aa161168923e22a7a6ffc58cc46412b9dda`
  and `671c1392fcfc0b71f7ae7bc57c9005b846a3beccfaf9b96ece67e2e10b65d69d`;
  final config SHA-256 values are
  `d851908530946f24a828e6ac1b414b0634105a56f0b19a87a521055ad140ec8f`
  and `4180cb95b4c9af59366442c6ede795678f354a008d45d74b3e23c36664c9f46a`.
  Configs verify the canonical base and exact frozen ten-target LoRA contract.
- NER is now validation-frozen. Because every LR and epoch tied at span-F1
  `0.0`, the preregistered lower-LR and earlier-checkpoint rules select LR
  `3e-5`, job `1190665`, epoch-1 `checkpoint-541`. Do not run its held-out
  evaluation until every task-family winner is frozen.
- POS LR-0 `1191712` completed epoch 1 at `checkpoint-283` with validation
  loss `11.073915798611111` and frozen token accuracy `0.0`, then resumed.
  After final-artifact and active-duplicate checks, the two released slots were
  filled individually by POS LR-1 `1192240` (`8e-5`, W&B `ojzj40j0`) and LR-2
  `1192241` (`1.5e-4`, W&B `n1z4mr8r`). Both passed fast-GDN, canonical
  checkpoint, BF16, exact ten-target LoRA with `4,649,088` trainable parameters,
  Hub-disabled, `2,259/2,250` validation-only dataset startup and reached step
  11 without a fault marker. Together with `1191712`, exactly three
  A100-40GB jobs are active; no A100-80GB or L40S work overlaps. Scratch is
  `102/300 GB` (`34.3%`). No held-out or sheet action is due.

- POS LR-0 `1191712` completed cleanly `0:0` in `01:15:53`. Frozen token
  accuracy was `0.0` at all three epochs, so patience stopped the run and
  restored epoch-1 `checkpoint-283`. Validation losses were
  `11.073915798611111/12.532222222222222/12.237223090277778`. Final adapter
  and config SHA-256 values are
  `1817e8e2014b63bf24d96f499f417965bb137533f1fe5bfa87507e5071142d25`
  and `1045e80c1369142d0a29567f6bdba47018b7296fbb0ac87190c56f1cdf42dca1`;
  config verifies the canonical base and exact frozen ten-target LoRA contract.
- After verifying the POS LR-0 final artifact, absence of an Intent LR-0 final
  adapter, and absence of an active duplicate, the released slot was filled by
  preregistered Intent LR-0 job `1192257` (`3e-5`, W&B `t7smhrtn`). It passed
  fast-GDN, canonical checkpoint, BF16 and exact ten-target LoRA startup with
  `4,649,088` trainable parameters, kept Hub paths disabled, built
  validation-only Intent datasets of `7,045/4,155`, and entered training
  without a fault marker. Together with POS LR-1 `1192240` and POS LR-2
  `1192241`, exactly three A100-40GB jobs are active; no A100-80GB or L40S work
  overlaps. Scratch is `103/300 GB` (`34.4%`). Kombuys is read-only and idle
  (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). No held-out
  or sheet action is due.

- POS LR-1 `1192240` and LR-2 `1192241` completed cleanly `0:0` in
  `00:56:16` and `00:56:09`. Frozen token accuracy remained `0.0` at every
  epoch; their validation-loss histories are LR-1
  `12.562584201388889/12.222162326388888/12.427377604166667` and LR-2
  `12.234731770833333/12.293402777777779/12.681581597222221`. Both restored
  epoch-1 `checkpoint-283`. LR-1 final adapter/config SHA-256 values are
  `6bfbf68ae775224bd43dd2d95a22e05f14b7297f91a1d09520e54b0219825381` and
  `6b6baf81281f8c3318623922abfc26299a0f1aa2100bca3e4d38fb389369f797`;
  LR-2 values are
  `d33ad0418adedf50eef8a31d24ce757ae89b718fbaf74ce45c2d60d20d1395db` and
  `4235eff737d961ec82595226622c1f734921118a77594c36171250b21fe4f19c`.
  Configs verify the canonical pure-GDN base and exact ten-target LoRA contract.
- POS is now validation-frozen. Every LR and epoch tied at token accuracy
  `0.0`, so the preregistered lower-LR then earlier-checkpoint rules select LR
  `3e-5`, job `1191712`, epoch-1 `checkpoint-283`. Do not run POS held-out
  evaluation before the global family-winner freeze gate.
- After rechecking that Intent LR-1/LR-2 had neither a final artifact nor an
  active duplicate, the two open A100-40GB slots were filled by Intent LR-1
  `1192267` (`8e-5`, W&B `r9qqor27`) and LR-2 `1192268` (`1.5e-4`, W&B
  `ug1czuc8`). Both loaded canonical `GatedDeltaNetForCausalLM` in BF16,
  attached `4,649,088` trainable parameters across the exact ten LoRA targets,
  built the same validation-only `7,045/4,155` datasets, and produced finite
  latest startup loss/gradient-norm records of `7.4798/24.5739` and
  `6.4725/10.7797` without a fault marker. With LR-0 `1192257`, exactly three owned
  jobs are active on `srvrocgpu010`, all A100-40GB `gpu:ampere`; no
  A100-80GB/L40S work overlaps. Scratch is `103/300 GB` (`34.5%`). Kombuys
  remains read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`). The sheet and all held-out adapter tests remain untouched.

- At the 14:23 SAST pass, Intent LR-0 `1192257` completed epoch-1 training at
  step `881` and produced finite validation loss `13.088938755076715`. Its
  frozen full-validation macro-F1 callback was still traversing all four
  languages and five prompts, so no selection metric or checkpoint decision
  was yet available. LR-1 `1192267` and LR-2 `1192268` were near step `500`,
  with finite losses/gradients and no traceback, OOM, NCCL, or runtime fault.
  All three jobs remained on `srvrocgpu010` A100-40GB `gpu:ampere`; scratch was
  `103/300 GB` (`34.5%`). No slot was released, no new job was submitted, and
  the held-out matrix and canonical sheet remained untouched.

- By the 14:53 SAST pass, Intent LR-0 `1192257` had completed its full
  epoch-1 callback and saved `checkpoint-881`: validation loss
  `13.088938755076715`, frozen validation macro-F1
  `0.0012702529230128718`. It resumed epoch 2 and reached about step `1275`
  with finite training state. Intent LR-1 `1192267` and LR-2 `1192268`
  completed epoch-1 trainer evaluation with finite losses
  `12.905990053399519` and `13.072481502895608`; their four-language,
  five-prompt selection callbacks were still running, so neither had a
  checkpoint metric eligible for LR comparison. All three remained healthy on
  `srvrocgpu010` A100-40GB `gpu:ampere`; scratch was `103/300 GB` (`34.6%`).
  No slot opened and no held-out, sheet, or new-job action was taken.

- At the 15:23 SAST pass, all three Intent trials had complete epoch-1
  checkpoints and frozen selection metrics. LR-0 `1192257` remained the
  provisional leader with macro-F1 `0.0012702529230128718`; LR-1 `1192267`
  and LR-2 `1192268` tied at `0.0010740856332265788`. All best checkpoints
  were epoch-1 `checkpoint-881`. LR-0 completed epoch-2 trainer evaluation
  with finite validation loss `13.292073743983153` and entered its second
  full prompt/language callback; LR-1/LR-2 were healthy around steps
  `1232/1198` in epoch 2. All three remained A100-40GB `gpu:ampere` jobs on
  `srvrocgpu010`; HEX scratch was `103/300 GB` (`34.6%`). No trial or family
  was frozen before terminal state, and no new, held-out, or sheet action was
  taken.

- At the 15:53 SAST pass, Intent LR-0 `1192257` had completed epoch 2 and
  saved `checkpoint-1762`. Epoch-2 validation loss was
  `13.292073743983153`; the trainer retained epoch-1 macro-F1
  `0.0012702529230128718` and `checkpoint-881`, so epoch 2 did not improve
  the frozen selection metric. LR-0 resumed epoch 3 and reached about step
  `1984`. LR-1 `1192267` and LR-2 `1192268` had reached their epoch-2
  validation phase with no fault marker and still retained epoch-1
  `checkpoint-881`. All three jobs remained on `srvrocgpu010` A100-40GB
  `gpu:ampere`; HEX scratch was `104/300 GB` (`34.7%`). No slot, held-out,
  sheet, or submission action was available or taken.

- At the 16:27 SAST pass, LR-1 `1192267` and LR-2 `1192268` had completed
  epoch 2 and improved their frozen validation macro-F1 to
  `0.001265884217324858` and `0.0013538364263929801`, respectively, at
  `checkpoint-1762`. LR-2 is therefore the provisional Intent leader, ahead
  of LR-0's retained `0.0012702529230128718`, but the family remains unfrozen
  until all three jobs terminate. LR-0 was inside its epoch-3 full validation
  callback; LR-1/LR-2 were in epoch-3 training near global steps `1997/1968`.
  All three remained healthy and were the only owned jobs, all on
  `srvrocgpu010` A100-40GB `gpu:ampere`; no A100-80GB/L40S job overlapped.
  HEX quota was `/home` `3/10 GB` (`32.5%`) and `/scratch` `104/300 GB`
  (`34.7%`). Kombuys remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX
  3080 Ti `1 MiB/0%`, scratch `61%`). No slot opened, so no T2X trial was
  submitted; held-out evaluation and the canonical sheet remained untouched.

- Intent LR-0 `1192257` completed cleanly `0:0` at 16:48:55 SAST after
  `03:24:52`. Epoch-3 validation loss was `13.254398644329122`; its frozen
  macro-F1 did not beat epoch-1 `0.0012702529230128718`, so the restored best
  remains `checkpoint-881`. Final adapter/config SHA-256 values are
  `e541616d33084967d52cf12deeea4c49be8f19e8285e5ed3eb9a515061f30b11`
  and `a2ce56b4c9c4f51ee838d3c7bb1a489aaf763c6026f757f356b2ebaeaa5dd1f3`.
  After verifying that the T2X LR-0 artifact directory did not exist and no
  active T2X duplicate was present, the released slot was filled by the next
  preregistered trial: T2X LR-0 job `1192813` (`3e-5`). Slurm accepted it with
  the frozen A100-40GB `gpu:ampere` contract; it remains pending for `Priority`
  with no scheduler start estimate, so startup health is not yet assessable.
  Intent LR-1/LR-2 `1192267/1192268` remain healthy in epoch 3 near global
  steps `2553/2520` and should reach their terminal callbacks in roughly
  20--35 minutes. They are the only running owned GPUs; no A100-80GB/L40S work
  overlaps. HEX quota remains `/home` `3/10 GB` (`32.5%`) and `/scratch`
  `104/300 GB` (`34.7%`). Kombuys remains read-only and idle (RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). No held-out evaluation
  or canonical-sheet update occurred.

- At the 17:24 SAST pass, Intent LR-1 `1192267` had completed epoch 3 at
  `checkpoint-2643` and improved the frozen validation macro-F1 to
  `0.002555133509628254`, making it the new provisional family leader.
  Epoch-3 validation loss was `12.903085749473526`; loss is not the Intent
  selection metric. LR-2 `1192268` was still completing its epoch-3 callback
  and retained epoch-2 macro-F1 `0.0013538364263929801` at
  `checkpoint-1762`, so Intent remains unfrozen until both terminal states are
  confirmed. T2X LR-0 `1192813` remains pending for `Priority` with no start
  estimate, and together with the two running Intent jobs accounts for the
  three-job limit. All requested GPU resources are A100-40GB `gpu:ampere`;
  there is no A100-80GB/L40S overlap. HEX quota remains `/home` `3/10 GB`
  (`32.5%`) and `/scratch` `104/300 GB` (`34.7%`). Held-out evaluation and the
  canonical sheet remain untouched. Kombuys remains read-only and idle (RTX
  5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`).

- Intent LR-2 `1192268` completed cleanly `0:0` at 17:25:19 SAST after
  `03:28:35`. Its epoch-3 validation loss was `12.963834282867028`, but the
  frozen best selection metric remains epoch-2 macro-F1
  `0.0013538364263929801` at `checkpoint-1762`. Final adapter/config SHA-256
  values are `4d87e8a8d2c411807108c2a8ec0c9fba2556737b378c08671a8fcbb7f375792b`
  and `b97c8fb363e6210080cb04592c239aee96b5958b7ed663ffff853520f51a84a6`.
  LR-1 `1192267` remains healthy and correctly continues epoch 4 because its
  epoch-3 macro-F1 improvement reset the preregistered patience counter; it was
  near global step `3220`. Intent therefore remains unfrozen. After verifying
  no T2X LR-1 artifact or active duplicate, the released slot was filled by
  preregistered T2X LR-1 job `1193166` (`8e-5`). It is pending for `Priority`;
  T2X LR-0 `1192813` is pending for `Resources` with Slurm's provisional
  start estimate `2026-08-08 00:25:19 SAST`. Together with running `1192267`,
  these are exactly the three owned jobs, all A100-40GB `gpu:ampere`; no
  A100-80GB/L40S work overlaps. HEX quota remains `/home` `3/10 GB` (`32.5%`)
  and `/scratch` `104/300 GB` (`34.7%`). Kombuys remains read-only and idle
  (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). No held-out
  evaluation or canonical-sheet update occurred.

- At the 18:24 SAST pass, Intent LR-1 `1192267` remained healthy at the end of
  epoch-4 training (`3524/8810`) and had entered validation; its frozen best
  was still epoch-3 macro-F1 `0.002555133509628254` at `checkpoint-2643`.
  Because epoch 3 improved, continued execution is expected under the fixed
  patience rule and is not a hang. No fault marker was present. T2X LR-0
  `1192813` remained pending for `Resources` with provisional start
  `2026-08-08 00:25:19 SAST`; T2X LR-1 `1193166` remained pending for
  `Priority` without an estimate. These are exactly the three owned jobs, all
  A100-40GB `gpu:ampere`; no A100-80GB/L40S work overlaps. HEX quota remained
  `/home` `3/10 GB` (`32.5%`) and `/scratch` `104/300 GB` (`34.7%`). Kombuys
  remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`). No slot, held-out evaluation, or canonical-sheet action was
  available.

- At the 18:54 SAST pass, Intent LR-1 `1192267` had completed epoch 4 at
  `checkpoint-3524`: validation loss `12.862821407754211`, with frozen best
  macro-F1 still `0.002555133509628254` at epoch-3 `checkpoint-2643`.
  Epoch 4 therefore consumed one of the two fixed non-improvement patience
  epochs, and the healthy job entered epoch 5 near global step `3901`; absent
  a new improvement, terminal ETA is roughly 45--60 minutes. No fault marker
  was present. T2X LR-0 `1192813` remained pending for `Resources` with
  provisional start `2026-08-08 00:25:19 SAST`; T2X LR-1 `1193166` remained
  pending for `Priority` without an estimate. These are exactly the three
  owned jobs, all A100-40GB `gpu:ampere`; no A100-80GB/L40S work overlaps.
  HEX quota was `/home` `3/10 GB` (`32.5%`) and `/scratch` `104/300 GB`
  (`34.8%`). Kombuys remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX
  3080 Ti `1 MiB/0%`, scratch `61%`). No held-out evaluation or canonical-sheet
  action occurred.

- At the 19:24 SAST pass, Intent LR-1 `1192267` was healthy inside its epoch-5
  full validation callback after completing training step `4405`; English and
  Sesotho partitions had completed and no epoch-5 checkpoint metric or fault
  marker was yet available. If the callback does not improve the frozen best
  macro-F1 `0.002555133509628254`, patience should terminate the job in roughly
  10--20 minutes and restore `checkpoint-2643`. T2X LR-0 `1192813` remained
  pending for `Resources` with provisional start `2026-08-08 00:25:19 SAST`;
  T2X LR-1 `1193166` remained pending for `Priority`. These are the three-job
  limit, all A100-40GB `gpu:ampere`, with no A100-80GB/L40S overlap. HEX quota
  was `/home` `3/10 GB` (`32.5%`) and `/scratch` `104/300 GB` (`34.8%`).
  Kombuys remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti
  `1 MiB/0%`, scratch `61%`). No held-out evaluation or canonical-sheet action
  occurred.

- Intent LR-1 `1192267` completed cleanly `0:0` at 19:44:30 SAST after
  `05:47:47`. Epoch 5 did not beat its frozen epoch-3 validation macro-F1
  `0.002555133509628254`, so the restored best is `checkpoint-2643`. Final
  adapter/config SHA-256 values are
  `755c2e778162f6ee2880ae33b25fc32d05354bb87b3ba1932ee05830e77400b7`
  and `db9b46e1620e982d69e91a9b673d5c8a97d6474a177304ce1ab5daab088fbccc`.
  Comparing all three terminal validation-only trials freezes the Intent
  family to LR `8e-5`, job `1192267`, epoch-3 `checkpoint-2643`; no held-out
  metric entered this selection. This makes five total frozen family winners
  (News, SIB, NER, POS, Intent), with T2X, AfriHG, and General still open.
- T2X LR-0 `1192813` started immediately on `srvrocgpu010` A100-40GB at
  19:44:30 SAST (W&B `esmggy0y`). It passed canonical pure-GDN
  `GatedDeltaNetForCausalLM`, BF16, exact architecture-complete LoRA
  (`4,649,088` trainable parameters), validation-only dataset (`3,859/460`
  train/eval), and Hub-disabled startup gates, then reached about step `170`
  without a fault marker. After verifying no T2X LR-2 artifact or active
  duplicate, the third preregistered grid point was submitted as job `1193880`
  (`1.5e-4`); it is pending for `Priority`. T2X LR-1 `1193166` is pending for
  `Resources` with provisional start `2026-08-08 00:25:19 SAST`. These are
  exactly the three owned jobs, all A100-40GB `gpu:ampere`, with no
  A100-80GB/L40S overlap. HEX quota was `/home` `3/10 GB` (`32.5%`) and
  `/scratch` `104/300 GB` (`34.8%`). Kombuys remained read-only and idle (RTX
  5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). No held-out
  evaluation or canonical-sheet update occurred.

- At the 20:25 SAST pass, T2X LR-0 `1192813` was healthy near step
  `676/1932`. Its epoch-1 `checkpoint-483` records validation loss
  `13.24983547044837` and frozen validation chrF `0.0`; the trainer therefore
  retains `checkpoint-483` as the current best. T2X LR-1 `1193166` started at
  20:08:15 SAST and passed the same pure-GDN, BF16, exact ten-target LoRA,
  validation-only `3,859/460` dataset, and Hub-disabled gates; it was healthy
  near step `341/1932` with no checkpoint yet. T2X LR-2 `1193880` remained
  pending for `Resources`, with Slurm's conservative start estimate
  `2026-08-08 19:44:30 SAST`. These are exactly the three owned jobs, all
  A100-40GB `gpu:ampere`, with no A100-80GB/L40S overlap and no fault markers.
  HEX quota was `/home` `3/10 GB` (`32.5%`) and `/scratch` `104/300 GB`
  (`34.8%`). Kombuys remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX
  3080 Ti `1 MiB/0%`, scratch `61%`). No slot, held-out evaluation, or
  canonical-sheet action was available.

- At the 20:53 SAST pass, T2X LR-0 `1192813` had completed epoch 2 at
  `checkpoint-966`: validation loss `13.744999363111413`, frozen validation
  chrF still `0.0`, and retained best `checkpoint-483`. It resumed cleanly and
  was near step `1111/1932`. T2X LR-1 `1193166` completed epoch 1 at
  `checkpoint-483` with validation loss `13.892279848845108` and chrF `0.0`,
  then resumed to about step `784/1932`. Both had zero fault markers. T2X LR-2
  `1193880` remained pending for `Priority`; Slurm no longer exposed a start
  estimate. These remain exactly three owned A100-40GB `gpu:ampere` jobs with
  no A100-80GB/L40S overlap. HEX quota was `/home` `3/10 GB` (`32.5%`) and
  `/scratch` `104/300 GB` (`34.9%`). Kombuys remained read-only and idle (RTX
  5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). No held-out test,
  sheet update, or new submission occurred.

- T2X LR-0 `1192813` completed cleanly `0:0` at 21:16:20 SAST in
  `01:31:50`. Validation chrF remained `0.0` through epoch 3, so the restored
  best is epoch-1 `checkpoint-483`; validation losses were
  `13.24983547044837/13.744999363111413/13.86932956861413`. Final adapter and
  config SHA-256 values are
  `4b1dc2fcd3656809057ada44cb31b7eae66cbd229edac21760e560f0585fff96` and
  `b0507be218a10ee0f1edae7a58bc41fa900387c2e2ef891aed1fa9fcc45332bd`.
  The config verifies the canonical base, rank 16, alpha 32, dropout 0.05, and
  exact ten-target LoRA contract.
- At 21:23 SAST, T2X LR-1 `1193166` was healthy near step `1304/1932` with
  chrF `0.0` through epoch 2, retained `checkpoint-483`, and validation losses
  `13.892279848845108/13.933596934442935`. T2X LR-2 `1193880` remained
  resource-pending with provisional start `2026-08-08 04:16:34 SAST`.
  Following exact-script rsync, dry-run verification, absence of a final
  artifact, and absence of an active duplicate, the released slot was filled
  by preregistered AfriHG LR-0 job `1194878` (`3e-5`); it is priority-pending,
  so startup health is not yet assessable. These are exactly three schedulable
  A100-40GB `gpu:ampere` jobs with no A100-80GB/L40S overlap. HEX quota was
  `/home` `3/10 GB` (`32.5%`) and `/scratch` `104/300 GB` (`35.0%`). Kombuys
  remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`). No held-out test or sheet update occurred.

- T2X LR-1 `1193166` completed cleanly `0:0` at 21:38:11 SAST in
  `01:29:56`. Validation chrF remained `0.0` through epoch 3, so the restored
  best is epoch-1 `checkpoint-483`; validation losses were
  `13.892279848845108/13.933596934442935/13.909505562160327`. Final adapter
  and config SHA-256 values are
  `066a1ac354bedd81e98b2431cc008e852805e07a5f43526355081719e7ea35ba` and
  `8db71038176b69cf0d6dc0e54ce8e2b0f1d89a6458b3c02cd1102f4ae387540a`.
  The config verifies the canonical base and exact frozen ten-target LoRA
  contract.
- T2X LR-2 `1193880` started immediately at 21:38:11 SAST on A100-40GB. By
  21:53 it had passed fast-GDN, canonical `GatedDeltaNetForCausalLM`, BF16,
  exact ten-target LoRA with `4,649,088` trainable parameters, Hub-disabled,
  and validation-only `3,859/460` dataset gates, and was healthy near step
  `286/1932` with zero fault markers. AfriHG LR-0 `1194878` remained
  resource-pending with provisional start `2026-08-08 04:16:34 SAST`.
  Following exact-script dry-run/rsync and final-artifact, duplicate, and
  three-job-limit checks, the remaining slot was filled by preregistered
  AfriHG LR-1 job `1194979` (`8e-5`); it is priority-pending. These are exactly
  three schedulable A100-40GB `gpu:ampere` jobs with no A100-80GB/L40S
  overlap. HEX quota was `/home` `3/10 GB` (`32.5%`) and `/scratch`
  `104/300 GB` (`35.0%`). Kombuys remained read-only and idle (RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). No held-out test or
  sheet update occurred.

- At the 22:23 SAST pass, T2X LR-2 `1193880` had completed epoch 1 at
  `checkpoint-483`: validation loss `14.137025518002718`, validation chrF
  `0.0`, and retained best `checkpoint-483`. It resumed cleanly and was near
  step `770/1932` with zero fault markers. AfriHG LR-0 `1194878` and LR-1
  `1194979` were both priority-pending without scheduler estimates. These are
  exactly three schedulable A100-40GB `gpu:ampere` jobs with no A100-80GB/L40S
  overlap. HEX quota was `/home` `3/10 GB` (`32.5%`) and `/scratch`
  `105/300 GB` (`35.0%`). Kombuys remained read-only and idle (RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). No slot, held-out test,
  sheet update, or new submission was available.

- At the 22:53 SAST pass, T2X LR-2 `1193880` had completed epoch 2 at
  `checkpoint-966`: validation loss `13.793190599524456`, validation chrF
  still `0.0`, and retained best `checkpoint-483`. It resumed cleanly and was
  near step `1251/1932` with zero fault markers; absent a metric improvement,
  the fixed patience rule should terminate it after epoch 3 in roughly 15--20
  minutes. AfriHG LR-0 `1194878` and LR-1 `1194979` remained priority-pending
  without scheduler estimates. These are exactly three schedulable A100-40GB
  `gpu:ampere` jobs with no A100-80GB/L40S overlap. HEX quota was `/home`
  `3/10 GB` (`32.5%`) and `/scratch` `105/300 GB` (`35.0%`). Kombuys remained
  read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch
  `61%`). No slot, held-out test, sheet update, or new submission was available.

- T2X LR-2 `1193880` completed cleanly `0:0` at 23:09:11 SAST in
  `01:31:00`. Validation chrF remained `0.0` through epoch 3, so the restored
  best is epoch-1 `checkpoint-483`; validation losses were
  `14.137025518002718/13.793190599524456/13.867788298233696`. Final adapter
  and config SHA-256 values are
  `b6a5e5b132a0f53adc62efc26c482bf7f81e4c4bbbd00ec22e9a07ca4a0bb6fb` and
  `a0b71d9707d0ea425fe62b5edb52c73ff4edc199ea778bebaae6a2049c67bab9`.
  The config verifies the canonical base and exact frozen ten-target LoRA
  contract. All three T2X LRs and epochs tied at validation chrF `0.0`, so the
  preregistered lower-LR and earlier-checkpoint rules freeze T2X to LR `3e-5`,
  job `1192813`, epoch-1 `checkpoint-483`. This is the sixth frozen family
  winner; held-out evaluation remains blocked.
- Following exact-script dry-run/rsync plus final-artifact, duplicate, and
  three-job-limit checks, AfriHG LR-2 job `1195667` (`1.5e-4`) was submitted.
  At 23:23 SAST, LR-0 `1194878` was resource-pending with provisional start
  `2026-08-08 04:16:34 SAST`; LR-1 `1194979` and LR-2 `1195667` were
  priority-pending without estimates. These are exactly three schedulable
  A100-40GB `gpu:ampere` jobs; no owned A100-80GB/L40S work overlaps. HEX quota
  was `/home` `3/10 GB` (`32.5%`) and `/scratch` `105/300 GB` (`35.1%`).
  Kombuys remained read-only; RTX 5090 was idle (`10 MiB/0%`), while RTX 3080
  Ti was externally active (`3,524 MiB/84%`). Scratch remained `61%`. No
  held-out test or sheet update occurred.

- At the 23:53 SAST pass, AfriHG LR-0 `1194878` remained resource-pending
  with provisional start `2026-08-08 04:16:34 SAST`; LR-1 `1194979` and LR-2
  `1195667` remained priority-pending without estimates. The blocker was
  explicit A100-40GB capacity: all four `gpu:ampere` devices on
  `srvrocgpu010` were allocated to other users' jobs. No owned GPU job was
  running, and no other user's job was modified. HEX quota remained `/home`
  `3/10 GB` (`32.5%`) and `/scratch` `105/300 GB` (`35.1%`). Kombuys was again
  read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch
  `61%`). No held-out test, sheet update, or submission occurred.
