# Pure-GDN adapter execution — 2026-08-08

- At 00:23:23 SAST, AfriHG LR-0 `1194878` (`3e-5`) and LR-1 `1194979`
  (`8e-5`) started together on `srvrocgpu010` A100-40GB `gpu:ampere`.
  Both passed the fast-GDN gate, loaded the canonical pure
  `GatedDeltaNetForCausalLM` base in BF16, attached the exact ten-target LoRA
  with `4,649,088` trainable parameters, kept Hub paths disabled, and built
  the validation-only AfriHG train/eval datasets of `24,649/3,082`. At the
  00:25 pass both had entered training and reached step `8/15410` at about
  `3.16 s/step` without a fault marker. Raw training time is therefore roughly
  13.5 hours for all five epochs, or about 8.1 hours to the earliest patience
  stop after epoch 3, before full task-native validation callback overhead.
  AfriHG LR-2 `1195667` (`1.5e-4`) remained
  resource-pending with Slurm's conservative start estimate
  `2026-08-09 00:23:23 SAST`. These are exactly three schedulable A100-40GB
  jobs with no owned A100-80GB/L40S overlap. HEX quota was `/home` `3/10 GB`
  (`32.5%`) and `/scratch` `105/300 GB` (`35.1%`). Kombuys remained read-only
  and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). No
  held-out test or canonical-sheet update occurred.

## Next gate

- Monitor both running trials through finite training, task-native validation
  chrF, checkpoints, artifacts, and terminal state. Do not freeze AfriHG until
  all three frozen LR trials are terminal. General remains next; all held-out
  adapter evaluation stays blocked behind the eight-family winner freeze.

- At the 00:53 SAST pass, AfriHG LR-0 `1194878` and LR-1 `1194979` were
  healthy near steps `553/15410` and `548/15410`, respectively, at about
  `3.1 s/step`, with zero fault markers. Raw full-grid ETA remained roughly
  12.8 hours from this pass; the earliest patience-stop window was roughly
  7.7 hours plus task-native validation callback time. LR-2 `1195667` remained
  resource-pending with conservative start `2026-08-09 00:23:23 SAST`.
  Exactly two owned A100-40GB jobs were running and the third was schedulable;
  no A100-80GB/L40S work overlapped. HEX quota was `/home` `3/10 GB` (`32.5%`)
  and `/scratch` `106/300 GB` (`35.4%`). Kombuys remained read-only and idle
  (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). No held-out
  test, sheet update, or submission occurred.

- At the 02:23 SAST pass, AfriHG LR-0 `1194878` and LR-1 `1194979` were
  healthy near steps `2329/15410` and `2275/15410`, respectively, with zero
  fault markers. Latest finite loss/gradient records were
  `0.8004/0.13055996596813202` and `0.7865/0.10756953805685043`; these remain
  health signals only. Epoch-1 training was roughly 40--45 raw minutes away,
  followed by the first full task-native validation callback. Earliest
  epoch-3 patience window was roughly 5.8 raw training hours away plus
  validation callbacks. LR-2 `1195667` remained resource-pending with
  conservative start `2026-08-09 00:23:23 SAST`. HEX quota remained `/home`
  `3/10 GB` (`32.5%`) and `/scratch` `106/300 GB` (`35.4%`). Kombuys remained
  read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch
  `61%`). No held-out test, sheet update, or submission occurred.

- At the 01:53 SAST pass, AfriHG LR-0 `1194878` and LR-1 `1194979` were
  healthy near steps `1737/15410` and `1701/15410`, respectively, with zero
  fault markers. Latest finite loss/gradient records were
  `0.8110/0.1755472868680954` and `0.7910/0.07980149984359741`; these remain
  health signals only. Epoch-1 training was about 1.1 raw hours away, followed
  by the first full task-native validation callback. Earliest epoch-3 patience
  window was about 6.4 raw training hours away plus validation callbacks; full
  five-epoch raw training had about 11.6 hours remaining. LR-2 `1195667`
  remained resource-pending with conservative start
  `2026-08-09 00:23:23 SAST`. HEX quota remained `/home` `3/10 GB` (`32.5%`)
  and `/scratch` `106/300 GB` (`35.4%`). Kombuys remained read-only and idle
  (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). No held-out
  test, sheet update, or submission occurred.

- At the 02:53 SAST pass, AfriHG LR-0 `1194878` and LR-1 `1194979` were
  healthy near steps `2925/15410` and `2848/15410`, respectively, with zero
  fault markers and finite latest loss/gradient records of
  `0.7934/0.11439689248800278` and `0.7848/0.0658421739935875`. LR-0 and LR-1
  were roughly 8 and 12 raw minutes from finishing epoch-1 training, followed
  by their first full task-native validation callbacks. LR-2 `1195667`
  remained resource-pending with conservative start
  `2026-08-09 00:23:23 SAST`. HEX quota remained `/home` `3/10 GB` (`32.5%`)
  and `/scratch` `106/300 GB` (`35.4%`). Kombuys remained read-only and idle
  (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). No held-out
  test, sheet update, or submission occurred.

- At the 01:23 SAST pass, AfriHG LR-0 `1194878` and LR-1 `1194979` were
  healthy near steps `1145/15410` and `1120/15410`, respectively, with stable
  throughput near `3.0--3.1 s/step` and zero fault markers. Latest finite
  training records were loss/gradient norm `1.3379/1.6356323957443237` for
  LR-0 and `0.8083/0.23651307821273804` for LR-1; these are health signals,
  not selection metrics. Earliest epoch-3 patience window was about 6.8 raw
  training hours away plus three task-native validation callbacks; full
  five-epoch raw training had about 12 hours remaining. LR-2 `1195667`
  remained resource-pending with conservative start
  `2026-08-09 00:23:23 SAST`. HEX quota remained `/home` `3/10 GB` (`32.5%`)
  and `/scratch` `106/300 GB` (`35.4%`). Kombuys remained read-only and idle
  (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). No held-out
  test, sheet update, or submission occurred.

- At the 03:26 SAST pass, AfriHG LR-0 `1194878` and LR-1 `1194979` remained
  healthy on `srvrocgpu010` A100-40GB after reaching epoch-1 step `3082/15410`.
  Their epoch-1 trainer validation losses were finite at
  `0.7932677865028381/0.7838549017906189` with runtimes
  `187.5596/187.7053` seconds. These losses are health signals only: the
  preregistered selector is mean Xhosa/Zulu validation chrF. Both task-native
  callbacks were still inside their automatic generation batch-size-64 probe,
  so no `eval_all_chrf`, checkpoint, or best-model artifact existed yet and no
  winner could be inferred. No traceback, OOM, NCCL, NaN, or Inf marker was
  present. LR-2 `1195667` remained resource-pending with Slurm's conservative
  start estimate `2026-08-09 00:23:23 SAST`. HEX quota was `/home` `3/10 GB`
  (`32.5%`) and `/scratch` `106/300 GB` (`35.4%`). These are exactly three
  schedulable A100-40GB `gpu:ampere` jobs with no owned A100-80GB/L40S overlap.
  Kombuys remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti
  `1 MiB/0%`, scratch `61%`). Callback throughput—and therefore a defensible
  terminal ETA—remains unknown until the first real generation progress or
  chrF artifact appears. No held-out test, sheet update, or submission
  occurred.

- At the 06:54 SAST pass, LR-0/LR-1 `1194878/1194979` remained healthy in
  epoch 2 at steps `4462/4401`, with latest finite loss/gradient
  `0.7869/0.11798744648694992` and `0.7818/0.0507061704993248`. Their
  verified epoch-1 `checkpoint-3082` artifacts remained the only/current
  bests. LR-2 `1195667` reached step `2430/15410`; latest finite loss/gradient
  was `0.7853/0.295060396194458`, leaving about 33 raw training minutes to its
  epoch-1 trainer validation before task-native callback overhead. No fault
  marker or new checkpoint appeared. Conditional ETAs remain about
  `10:00 SAST` for LR-2's first checkpoint, `11:00 SAST` for LR-0/LR-1 epoch
  2, and `16:10/20:30 SAST` for tied-metric LR-0/1 and LR-2 termination.
  HEX quota remained `/home` `3/10 GB` (`32.5%`) and `/scratch` `107/300 GB`
  (`35.7%`). Exactly three A100-40GB jobs are active, with no A100-80GB/L40S
  overlap or General slot. Kombuys remained read-only and idle (RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). No held-out test,
  Sheet update, or submission occurred.

- At the 07:24 SAST pass, LR-0/LR-1 `1194878/1194979` remained healthy in
  epoch 2 at steps `5032/4972`, with latest finite loss/gradient
  `0.7888/0.12010953575372696` and `0.7797/0.12031915038824081`; their
  full-validation chrF `0.0` epoch-1 checkpoints remained current bests.
  LR-2 `1195667` reached step `3018/15410` with latest finite loss/gradient
  `0.7813/0.05587604269385338`, roughly three raw training minutes from the
  epoch-1 boundary. No new checkpoint or fault marker existed yet.
  Conditional ETAs remain near `10:00 SAST` for LR-2's first checkpoint,
  `11:00 SAST` for LR-0/LR-1 epoch 2, and `16:10/20:30 SAST` for tied-metric
  LR-0/1 and LR-2 termination. HEX quota remained `/home` `3/10 GB` (`32.5%`)
  and `/scratch` `107/300 GB` (`35.7%`). Exactly three A100-40GB jobs are
  active, with no A100-80GB/L40S overlap or General slot. Kombuys remained
  read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch
  `61%`). No held-out test, Sheet update, or submission occurred.

- At the 03:54 SAST pass, jobs `1194878/1194979` remained running cleanly
  after `03:30:34` on `srvrocgpu010`; `1195667` remained resource-pending with
  estimated start `2026-08-09 00:23:23 SAST`. Neither running trial had yet
  produced a checkpoint, trainer state, validation chrF, or other selection
  artifact. LR-0's task-native callback emitted batch-level truncation events
  at `03:10:45`, `03:29:54`, and `03:50:44`; LR-1 emitted corresponding
  events at `03:15:05` and `03:34:06`. The observed roughly 19--21-minute
  cadence after the automatic batch-size-64 probes makes the 24-hour wall-time
  a material execution risk. Extrapolating about 49 batches for `3,082`
  validation examples gives a provisional epoch-1 callback completion window
  around `19:00--20:00 SAST`, with wide uncertainty because callback progress
  does not expose an explicit batch counter. That would leave insufficient
  wall time for all patience-governed epochs. Do not cancel or modify the
  frozen runs from this estimate: wait for the first task-native chrF and
  checkpoint, then treat any actual wall-time termination as infrastructure
  failure and resume identically from preserved state if supported. No fault
  markers were present. HEX quota remained `/home` `3/10 GB` (`32.5%`) and
  `/scratch` `106/300 GB` (`35.4%`); only A100-40GB `gpu:ampere` work was
  owned. Kombuys remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080
  Ti `1 MiB/0%`, scratch `61%`). No held-out test, Sheet update, or submission
  occurred.

- At the 07:54 SAST pass, LR-2 `1195667` had completed epoch-1 training step
  `3082/15410` and finite trainer validation loss `0.7802404761314392` in
  `190.2706` seconds, then entered the full task-native AfriHG callback. Its
  automatic generation probe selected batch size 64 at `07:35:55`; no chrF or
  checkpoint existed yet and trainer loss is not the selection metric.
  LR-0/LR-1 `1194878/1194979` remained healthy in epoch 2 at steps
  `5609/5553`, about 29--32 raw training minutes from their epoch-2 trainer
  validation. No fault markers existed. Conditional ETAs remain near
  `10:05 SAST` for LR-2's first chrF checkpoint, `11:00 SAST` for LR-0/LR-1
  epoch 2, and `16:10/20:30 SAST` for tied-metric LR-0/1 and LR-2 terminal
  states. HEX quota remained `/home` `3/10 GB` (`32.5%`) and `/scratch`
  `107/300 GB` (`35.7%`). Exactly three A100-40GB jobs are active, with no
  A100-80GB/L40S overlap or General slot. Kombuys remained read-only and idle
  (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). No held-out
  test, Sheet update, or submission occurred.

- At the 04:24 SAST pass, `1194878/1194979` remained healthy and running after
  `04:00:28`; `1195667` remained resource-pending for the same conservative
  `2026-08-09 00:23:23 SAST` start. LR-0 emitted further callback events at
  `04:18:08`; LR-1 emitted them at `03:54:48` and `04:21:59`. Thus both runs
  show nearly identical post-probe intervals that grew from roughly 19--21 to
  about 27 minutes. Still no chrF, checkpoint, trainer state, final adapter,
  or fault marker exists. The earlier `19:00--20:00 SAST` callback estimate
  is no longer conservative: using the observed range for the remaining
  approximate `3,082/64` batches moves the wide epoch-1 completion window to
  about `21:00 SAST` through the `00:23 SAST` wall-time boundary. This is an
  extrapolation from log events, not an explicit progress counter. Preserve
  both jobs unchanged until valid artifact or terminal evidence exists. HEX
  quota remained `/home` `3/10 GB` (`32.5%`) and `/scratch` `106/300 GB`
  (`35.4%`); only A100-40GB `gpu:ampere` work was owned. Kombuys remained
  read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch
  `61%`). No held-out test, Sheet update, or submission occurred.

- At 04:49:05 SAST Slurm started AfriHG LR-2 `1195667` (`1.5e-4`) on
  `srvrocgpu010`, so the complete frozen three-LR grid is now concurrently
  active on exactly three A100-40GB `gpu:ampere` GPUs. By 04:54 it had loaded
  the canonical local pure `GatedDeltaNetForCausalLM` checkpoint in BF16,
  attached `4,649,088` trainable LoRA parameters consistent with the frozen
  exact ten-target contract, kept `push_to_hub=false`, built the same
  validation-only `24,649/3,082` train/eval datasets, and reached step
  `60/15410` near `3.09 s/step` without a fault marker. LR-0/LR-1
  `1194878/1194979` remained healthy in epoch-1 task-native callbacks; their
  latest events were `04:43:10/04:46:48`, still with no chrF or checkpoint.
  The wide `21:00--00:23 SAST` epoch-1 callback estimate remains provisional.
  HEX quota was `/home` `3/10 GB` (`32.5%`) and `/scratch` `106/300 GB`
  (`35.6%`). No owned A100-80GB/L40S work overlaps, and there is no free
  schedulable slot for General. Kombuys remained read-only and idle (RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). No held-out test,
  Sheet update, or new submission occurred.

- At the 05:24 SAST pass, all three AfriHG trials remained running on
  `srvrocgpu010` A100-40GB. LR-2 `1195667` reached step `646/15410` near
  `2.99 s/step`; its latest finite scalar was loss `1.1426`, gradient norm
  `1.4747387170791626`, and learning rate `0.0001292656587473002`, with no
  fault marker. At that rate epoch-1 training is roughly due around
  `07:25 SAST`, before its own expensive task-native callback. LR-0/LR-1
  `1194878/1194979` remained in their epoch-1 callbacks with no chrF,
  checkpoint, or final artifact. Their logs had been quiet since
  `04:43:10/04:46:48`, a roughly 41/37-minute interval at inspection. This
  strengthens the wall-time risk but does not establish a hang: Slurm still
  reported both running, and no traceback/OOM/NCCL/non-finite marker existed.
  Treat the callback ETA as at or beyond the `00:23 SAST` wall-time boundary
  until explicit progress or an artifact says otherwise. HEX quota remained
  `/home` `3/10 GB` (`32.5%`) and `/scratch` `106/300 GB` (`35.6%`). Exactly
  three A100-40GB jobs are active with no A100-80GB/L40S overlap. Kombuys
  remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`). No held-out test, Sheet update, or submission occurred.

- At the 09:24 SAST pass, all three AfriHG trials remained `RUNNING` on
  `srvrocgpu010`: LR-0/LR-1 `1194878/1194979` continued epoch-2 task-native
  callbacks with fresh real progress at `09:12:09/09:14:46`, and LR-2
  `1195667` continued its epoch-1 callback with fresh real progress at
  `09:08:05`. No new chrF/checkpoint existed beyond LR-0/LR-1 epoch-1
  `checkpoint-3082`; the apparent single fault-pattern match per log was only
  the benign configuration field `logging_nan_inf_filter=True`, with no
  traceback, OOM, NCCL, or non-finite runtime event. Exactly three owned
  A100-40GB `gpu:ampere` jobs remained active, no A100-80GB/L40S overlap, and
  no General slot was free. First-checkpoint ETAs remain about `10:05 SAST`
  for LR-2 and `11:00 SAST` for LR-0/LR-1; tied-metric terminal ETAs remain
  near `20:30/16:10 SAST`. HEX quota remained `/home` `3/10 GB` (`32.5%`)
  and `/scratch` `107/300 GB` (`35.7%`). Kombuys remained read-only and idle
  (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`; only the
  pre-existing `tailscale-kombuys` tmux session). Base progress remains
  `16/16`, adapter-family winners `6/8`; no held-out test, Sheet update,
  Hugging Face action, or submission occurred.

- At the 08:55 SAST pass, LR-0/LR-1 `1194878/1194979` had both completed
  finite epoch-2 trainer validation with losses
  `0.7849799990653992/0.7786943912506104`; these are health signals, not the
  frozen selector. Their full task-native callbacks selected generation batch
  size 64 at `08:31:53/08:34:46` and logged real progress at
  `08:51:12/08:53:56`. LR-2 `1195667` remained in its epoch-1 callback and
  logged new real progress at `08:43:07`; it still had no chrF/checkpoint.
  All three jobs remained `RUNNING` and fault-free on `srvrocgpu010`, exactly
  three A100-40GB `gpu:ampere` allocations with no owned A100-80GB/L40S
  overlap and no free General slot. Conditional first-checkpoint ETAs remain
  near `10:05 SAST` for LR-2 and `11:00 SAST` for LR-0/LR-1; tied-metric
  terminal ETAs remain near `20:30/16:10 SAST`. HEX quota remained `/home`
  `3/10 GB` (`32.5%`) and `/scratch` `107/300 GB` (`35.7%`). Kombuys remained
  read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch
  `61%`; only the pre-existing `tailscale-kombuys` tmux session). Frozen base
  progress remains `16/16`, adapter-family winners `6/8`, and no held-out
  test, Sheet update, Hugging Face action, or submission occurred.

- Epoch-1 selection artifacts became valid at `05:41:44/05:44:51 SAST` for
  LR-0 `1194878` and LR-1 `1194979`. Both `checkpoint-3082/trainer_state.json`
  files record `best_metric=0.0`, `epoch=1.0`, `global_step=3082`, and their
  own checkpoint as `best_model_checkpoint`; this is the preregistered full
  validation mean chrF. The 64-record `debug_generation_examples` files are
  not a metric cap: `training/factory.py` constructs
  `GenerationMetricsCallback(max_samples_per_lang=None)`, while the separate
  callback debug collector defaults to 64 examples. Thus the selection
  metric used all `3,082` validation examples and satisfies the frozen
  protocol. LR-0 adapter/config/trainer-state SHA-256 values are
  `a03f1ef17e228fd69be66ae7af2d5b281d4b2fdced9dea1cd77519f825c5179e`,
  `723872cc560f7fda5aafe2560bd70a0f24a456fc8ef183a57991149e1ac3eb9c`,
  and `0a0347d7440fb34105f04514abea6e8d7946195f8d6f6bf83eb5d9b4d26dba31`.
  LR-1 values are
  `d692a07a85b1ec8e98b0b5043b7097349039220de9eb0fde81b081bfb8129de4`,
  `35f3db3aa551d4754122139992d3bf6d9878e0f90f8df98c0788cf96a8d0a3d3`,
  and `fdd4aad47f250708daf0f16c0a79ef0d3af684fc9da70dd49307680e358b900b`.
  Both jobs resumed epoch-2 training and were near steps `3324/3262` by 05:53
  without fault markers. Their measured epoch-1 callback durations were about
  `2h35m`, so if chrF stays tied at `0.0`, epoch-2 checkpoints are expected
  near `11:00 SAST` and patience-stop epoch-3 completion near `16:10 SAST`.
  LR-2 `1195667` was healthy at step `1244/15410` with latest finite
  loss/gradient `0.7953/0.1947757601737976`; its first checkpoint is expected
  near `10:00 SAST` and tied-metric epoch-3 completion near `20:30 SAST`.
  These are conditional ETAs; validation improvement would extend training
  and restore wall-time risk. HEX quota was `/home` `3/10 GB` (`32.5%`) and
  `/scratch` `107/300 GB` (`35.7%`). Exactly three A100-40GB jobs remain
  active with no A100-80GB/L40S overlap. Kombuys remained read-only and idle
  (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`). No held-out
  test, Sheet update, or submission occurred.

- At the 06:24 SAST pass, all three AfriHG trials remained healthy on
  `srvrocgpu010` A100-40GB. LR-0/LR-1 `1194878/1194979` advanced normally
  through epoch 2 to steps `3885/3824`; latest finite loss/gradient values were
  `0.7908/0.20520105957984924` and `0.7834/0.07749360054731369`. Their verified
  epoch-1 chrF `0.0` checkpoints remain current bests. LR-2 `1195667` reached
  step `1836/15410` with latest finite loss/gradient
  `0.7861/0.1272258758544922`. No new checkpoint or fault marker appeared.
  Conditional tied-metric ETAs remain about `11:00 SAST` for LR-0/LR-1
  epoch-2 checkpoints, `16:10 SAST` for their terminal epoch 3, and
  `10:00/20:30 SAST` for LR-2's first/terminal checkpoints. HEX quota remained
  `/home` `3/10 GB` (`32.5%`) and `/scratch` `107/300 GB` (`35.7%`). Exactly
  three A100-40GB jobs are active, with no A100-80GB/L40S overlap or General
  slot. Kombuys remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080
  Ti `1 MiB/0%`, scratch `61%`). No held-out test, Sheet update, or submission
  occurred.

- At the 08:24 SAST pass, LR-0 `1194878` had crossed epoch-2 step `6164` and
  entered its evaluation/completion sequence; LR-1 `1194979` was at step
  `6127/15410`, roughly two raw training minutes from the same boundary. Both
  remained fault-free and their epoch-1 full-validation chrF `0.0`
  checkpoints remained current bests pending new task-native metrics. LR-2
  `1195667` remained healthy in its epoch-1 full-validation callback, with
  progress events at `07:55:01` and `08:15:47` after the batch-size-64 probe;
  no chrF/checkpoint existed yet. First-checkpoint ETAs remain about
  `10:05 SAST` for LR-2 and `11:00 SAST` for LR-0/LR-1, with tied-metric
  terminal ETAs near `20:30/16:10 SAST`. HEX quota remained `/home` `3/10 GB`
  (`32.5%`) and `/scratch` `107/300 GB` (`35.7%`). Exactly three A100-40GB
  jobs are active, with no A100-80GB/L40S overlap or General slot. Kombuys
  remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`). No held-out test, Sheet update, or submission occurred.

- At the 09:54 SAST pass, LR-0/LR-1 `1194878/1194979` remained healthy in
  epoch-2 task-native callbacks with fresh real progress at
  `09:39:45/09:42:08`. LR-2 `1195667` remained `RUNNING` in its epoch-1
  callback; its most recent event was still `09:08:05`, a roughly 46-minute
  quiet interval but not a proven hang because Slurm remained active and no
  traceback, OOM, NCCL, or new selection artifact appeared. No checkpoint
  existed beyond LR-0/LR-1 epoch-1 `checkpoint-3082`. Conditional checkpoint
  ETAs are now roughly `10:05--10:30 SAST` for LR-2 and `11:00 SAST` for
  LR-0/LR-1; tied-metric terminal ETAs remain near `20:30/16:10 SAST`.
  Exactly three A100-40GB `gpu:ampere` jobs remained active with no owned
  A100-80GB/L40S overlap and no free General slot. HEX quota remained `/home`
  `3/10 GB` (`32.5%`) and `/scratch` `107/300 GB` (`35.7%`). Kombuys remained
  read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch
  `61%`; only the pre-existing `tailscale-kombuys` tmux session). Base
  progress remains `16/16`, adapter-family winners `6/8`; no held-out test,
  Sheet update, Hugging Face action, or submission occurred.

- LR-2 `1195667` produced a valid epoch-1 selection artifact at
  `10:06:32 SAST`: `checkpoint-3082/trainer_state.json` records
  `best_metric=0.0`, `best_model_checkpoint=.../lr_2/checkpoint-3082`,
  `global_step=3082`, and `epoch=1.0`. Because the frozen best-model metric is
  `eval_all_chrf`, this is full-validation mean chrF `0.0`; the accompanying
  trainer loss `0.7802404761314392` remains health-only. SHA-256 values for
  `adapter_model.bin`, `adapter_config.json`, and `trainer_state.json` are
  `6c8a2711ece2cfddb0d87140093c5b1a905c3214efd86d21f5d1f91eca135a33`,
  `6f9169df6bdd3f6b1ef5cdbe2043f823c12bb60fbe0096e60bddabbdb79a03c7`,
  and `1b9dd239137717320d2ba07a8a3f643edde79137b4a394c7018b1a2f214a5c8a`.
  LR-2 resumed epoch 2 cleanly and reached step `3421/15410` near
  `3.06 s/step` by 10:23. LR-0/LR-1 `1194878/1194979` remained in epoch-2
  callbacks with fresh events at `10:04:54/10:07:06`; their epoch-1 chrF
  `0.0` checkpoints remained current bests. All three jobs remained healthy
  on exactly three A100-40GB `gpu:ampere` allocations with no owned
  A100-80GB/L40S overlap and no free General slot. Conditional terminal ETAs
  remain near `16:10 SAST` for LR-0/LR-1 and `20:30 SAST` for LR-2. HEX quota
  remained `/home` `3/10 GB` (`32.5%`) and `/scratch` `107/300 GB` (`35.7%`).
  Kombuys remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti
  `1 MiB/0%`, scratch `61%`; only pre-existing `tailscale-kombuys` tmux).
  Base remains `16/16`, adapter-family winners `6/8`; no held-out test, Sheet
  update, Hugging Face action, or General submission occurred.

- At the 10:54 SAST pass, all three AfriHG jobs remained `RUNNING` on exactly
  three A100-40GB GPUs. LR-0/LR-1 `1194878/1194979` remained inside epoch-2
  task-native callbacks; their last events were `10:04:54/10:07:06`, no new
  `checkpoint-6164` existed, and no fault marker appeared. LR-2 `1195667`
  advanced normally through epoch 2 to step `4005/15410` near `3.03 s/step`;
  its verified epoch-1 chrF `0.0` `checkpoint-3082` remained current best.
  Conditional LR-0/LR-1 epoch-2 checkpoint ETA is now about
  `11:00--11:20 SAST`; tied-metric terminal ETAs remain near `16:10 SAST`
  for LR-0/LR-1 and `20:30 SAST` for LR-2. There was no owned A100-80GB/L40S
  overlap and no free General slot. HEX quota remained `/home` `3/10 GB`
  (`32.5%`) and `/scratch` `107/300 GB` (`35.7%`). Kombuys remained read-only
  and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`;
  only pre-existing `tailscale-kombuys` tmux). Base remains `16/16`, family
  winners `6/8`; no held-out test, Sheet update, publication, or submission
  occurred.

- LR-0/LR-1 `1194878/1194979` produced valid epoch-2 `checkpoint-6164`
  artifacts at `11:03:47/11:05:29 SAST`. Both trainer states record
  `global_step=6164`, `epoch=2.0`, and `best_metric=0.0` with the epoch-1
  `checkpoint-3082` still the best model, so full-validation mean chrF again
  tied at `0.0`. Epoch-2 trainer losses improved to
  `0.7849799990653992/0.7786943912506104`, but remain health-only. LR-0
  checkpoint-6164 adapter/config/trainer-state SHA-256 values are
  `caf8c71d008b3fa7ac85b2ad8fe5d5638359ec73629f8904e388f8afa132a13c`,
  `723872cc560f7fda5aafe2560bd70a0f24a456fc8ef183a57991149e1ac3eb9c`,
  and `36ab7b012e6392014bd9107b2636f2abe8a0d9ced01b975818888d968f988134`.
  LR-1 values are
  `4024cdaf1ad78a567eec241324d80aebde73dcd94e9b4c0341187e0d5f5bdb71`,
  `35f3db3aa551d4754122139992d3bf6d9878e0f90f8df98c0788cf96a8d0a3d3`,
  and `0584c24a6f58c1dd76dd1e49eb08680bfd0baf776e3d484a27c3474fef6e790b`.
  At 11:46 SAST LR-0/LR-1 had resumed epoch 3 at steps `6971/6936`; LR-2
  `1195667` advanced through epoch 2 to step `5015`. All three remained
  healthy on exactly three A100-40GB `gpu:ampere` allocations with no owned
  A100-80GB/L40S overlap and no free General slot. Conditional terminal ETAs
  remain near `16:20 SAST` LR-0/LR-1 and `20:40 SAST` LR-2. HEX quota was
  `/home` `3/10 GB` (`32.5%`) and `/scratch` `107/300 GB` (`35.8%`). Kombuys
  remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`; only pre-existing `tailscale-kombuys` tmux). Base remains
  `16/16`, family winners `6/8`; no held-out test, Sheet update, Hugging Face
  action, or General submission occurred.

- At the 12:24 SAST pass, LR-0/LR-1 `1194878/1194979` advanced normally
  through epoch 3 to steps `7712/7676` near `3.11 s/step`; LR-2 `1195667`
  reached step `5752/15410` in epoch 2 near `3.13 s/step`. All three jobs
  remained `RUNNING` and fault-free on `srvrocgpu010`, exactly three owned
  A100-40GB `gpu:ampere` allocations with no A100-80GB/L40S overlap and no
  free General slot. LR-2 was about 22 raw training minutes from epoch-2
  validation; conditional terminal ETAs remain about `16:20 SAST` for
  LR-0/LR-1 and `20:40 SAST` for LR-2. HEX quota remained `/home` `3/10 GB`
  (`32.5%`) and `/scratch` `107/300 GB` (`35.8%`). Kombuys remained read-only
  and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`;
  only pre-existing `tailscale-kombuys` tmux). Base remains `16/16`, family
  winners `6/8`; no held-out test, Sheet update, Hugging Face action, or
  submission occurred.

- At the 13:00 SAST pass, LR-2 `1195667` had reached epoch-2 step `6164`,
  completed finite trainer validation loss `0.7753579020500183` in
  `191.5017` seconds, and entered its full task-native callback; automatic
  generation selected batch size 64 at `12:55:02`. The trainer loss is a
  health signal only, and no `checkpoint-6164` or new chrF existed yet.
  LR-0/LR-1 `1194878/1194979` advanced normally through epoch 3 to steps
  `8406/8367` near `3.1 s/step`, about 43--46 raw minutes from their epoch-3
  validation boundary. All three jobs remained `RUNNING` and fault-free on
  exactly three A100-40GB `gpu:ampere` allocations with no owned
  A100-80GB/L40S overlap and no free General slot. Conditional terminal ETAs
  remain near `16:25 SAST` for LR-0/LR-1 and `20:40 SAST` for LR-2. HEX quota
  remained `/home` `3/10 GB` (`32.5%`) and `/scratch` `107/300 GB` (`35.8%`).
  Kombuys remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti
  `1 MiB/0%`, scratch `61%`; only pre-existing `tailscale-kombuys` tmux).
  Base remains `16/16`, family winners `6/8`; no held-out test, Sheet update,
  Hugging Face action, or submission occurred.

- At the 14:19 SAST pass, LR-0/LR-1 `1194878/1194979` had crossed epoch-3
  step `9246`, completed finite trainer validation losses
  `0.7823924422264099/0.7764996290206909` in `190.5165/188.9957` seconds,
  and entered terminal full task-native callbacks. Automatic generation chose
  batch size 64 at `13:53:21/13:55:16`. These losses are health-only; no
  epoch-3 chrF/checkpoint or clean terminal state existed yet. LR-2 `1195667`
  continued its epoch-2 callback with fresh progress at `14:02:21`, still
  without `checkpoint-6164`. All three jobs remained `RUNNING` and fault-free
  on exactly three A100-40GB `gpu:ampere` allocations with no owned
  A100-80GB/L40S overlap and no General slot. Conditional terminal ETAs remain
  near `16:25 SAST` for LR-0/LR-1 and `20:40 SAST` for LR-2. HEX quota
  remained `/home` `3/10 GB` (`32.5%`) and `/scratch` `107/300 GB` (`35.8%`).
  Kombuys remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti
  `1 MiB/0%`, scratch `61%`; only pre-existing `tailscale-kombuys` tmux).
  Base remains `16/16`, family winners `6/8`; no held-out test, Sheet update,
  Hugging Face action, or submission occurred.

- At the 13:30 SAST pass, LR-0/LR-1 `1194878/1194979` remained healthy at
  epoch-3 steps `8978/8940` near `3.1 s/step`, about 14--16 raw minutes from
  their final training boundary before task-native validation. LR-2
  `1195667` remained `RUNNING` in its epoch-2 callback with a fresh progress
  event at `13:14:11`; no `checkpoint-6164`, new chrF, or fault marker existed
  yet. Exactly three A100-40GB `gpu:ampere` jobs remained active with no owned
  A100-80GB/L40S overlap and no General slot. Conditional terminal ETAs remain
  near `16:25 SAST` for LR-0/LR-1 and `20:40 SAST` for LR-2. HEX quota
  remained `/home` `3/10 GB` (`32.5%`) and `/scratch` `107/300 GB` (`35.8%`).
  Kombuys remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti
  `1 MiB/0%`, scratch `61%`; only pre-existing `tailscale-kombuys` tmux).
  Base remains `16/16`, family winners `6/8`; no held-out test, Sheet update,
  Hugging Face action, or submission occurred.

- At the 14:32 SAST pass, all three AfriHG trials remained `RUNNING` and
  fault-free on `srvrocgpu010`: LR-0 `1194878` and LR-1 `1194979` continued
  their terminal epoch-3 task-native callbacks with fresh progress at
  `14:12:39/14:14:24`, while LR-2 `1195667` continued its epoch-2 callback
  with fresh progress at `14:27:12`. No epoch-3 LR-0/LR-1 chrF/checkpoint or
  LR-2 `checkpoint-6164` existed yet, so AfriHG remains unfrozen and no
  General slot is open. Conditional terminal ETAs remain near `16:25 SAST`
  for LR-0/LR-1 and `20:40 SAST` for LR-2. HEX quota was `/home` `3/10 GB`
  (`32.5%`) and `/scratch` `107/300 GB` (`35.8%`). Exactly three owned
  A100-40GB `gpu:ampere` jobs were active, with no A100-80GB/L40S overlap.
  Kombuys remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti
  `1 MiB/0%`, scratch `61%`; only pre-existing `tailscale-kombuys` tmux).
  Base remains `16/16`, family winners `6/8`; no held-out test, Sheet update,
  Hugging Face action, or submission occurred.

- At the 15:01 SAST pass, LR-0/LR-1 `1194878/1194979` remained healthy in
  their terminal task-native callbacks with fresh progress at
  `14:33:38/14:35:14`. LR-2 `1195667` remained healthy in its epoch-2
  callback with its latest progress at `14:27:12`. No epoch-3 LR-0/LR-1
  chrF/checkpoint, LR-2 `checkpoint-6164`, final adapter, terminal state, or
  runtime fault marker existed. All three jobs continued on exactly three
  A100-40GB `gpu:ampere` allocations on `srvrocgpu010`, with no owned
  A100-80GB/L40S overlap and no General slot. Conditional terminal ETAs remain
  near `16:25 SAST` for LR-0/LR-1 and `20:40 SAST` for LR-2. HEX quota was
  `/home` `3/10 GB` (`32.5%`) and `/scratch` `107/300 GB` (`35.8%`). Kombuys
  remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`; only pre-existing `tailscale-kombuys` tmux). Base remains
  `16/16`, family winners `6/8`; no held-out test, Sheet update, Hugging Face
  action, or submission occurred.

- By the 16:02 SAST pass, LR-2 `1195667` had produced its valid epoch-2
  `checkpoint-6164` at `15:25:39`. Its trainer state records
  `best_metric=0.0`, `global_step=6164`, and the epoch-1 `checkpoint-3082`
  remains the best model, so epoch-2 full-validation mean chrF again tied at
  `0.0`. SHA-256 values for checkpoint-6164 adapter/config/trainer state are
  `ee89e2af74f8d3076bdfb7dc85d9c15aece1198cda8b8a84510c12baab2e98d6`,
  `6f9169df6bdd3f6b1ef5cdbe2043f823c12bb60fbe0096e60bddabbdb79a03c7`,
  and `6d4f7fba6fabf0130d54d4b0fafe1cff21ed86547ec8db1bf10a6ce1eac01c47`.
  LR-2 resumed epoch 3 cleanly and reached step `6581/15410`. LR-0/LR-1
  `1194878/1194979` remained healthy in their terminal callbacks with fresh
  progress at `15:26:22/15:27:35`; neither had an epoch-3 checkpoint, final
  adapter, clean exit, or fault marker yet. Exactly three A100-40GB
  `gpu:ampere` jobs remained active on `srvrocgpu010`, with no owned
  A100-80GB/L40S overlap and no General slot. Conditional terminal ETAs remain
  near `16:25 SAST` for LR-0/LR-1 and `20:40 SAST` for LR-2. HEX quota was
  `/home` `3/10 GB` (`32.5%`) and `/scratch` `107/300 GB` (`35.8%`). Kombuys
  remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`; only pre-existing `tailscale-kombuys` tmux). Base remains
  `16/16`, family winners `6/8`; no held-out test, Sheet update, Hugging Face
  action, or General submission occurred.

- At 16:07 SAST, all three AfriHG jobs remained `RUNNING`. Slurm accounting
  showed active batch CPU time tracking elapsed time for LR-0/LR-1
  `1194878/1194979`, but their logs had not advanced since
  `15:26:27/15:27:35`; neither job had a checkpoint-9246, final adapter,
  terminal state, or fault marker. This approximately 40-minute quiet period is
  longer than recent callback event intervals but is not evidence of failure;
  preserve both frozen jobs and wait for artifact or terminal evidence. LR-2
  `1195667` continued epoch-3 training with its log current at `16:06:22`.
  Exactly three A100-40GB `gpu:ampere` allocations remained active on
  `srvrocgpu010`, with no owned A100-80GB/L40S overlap and no General slot.
  Widen LR-0/LR-1's conditional terminal window to approximately
  `16:25--17:00 SAST`; LR-2 remains near `20:40 SAST`. HEX quota remained
  `/home` `3/10 GB` (`32.5%`) and `/scratch` `107/300 GB` (`35.8%`). Kombuys
  remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`; only pre-existing `tailscale-kombuys` tmux). Base remains
  `16/16`, family winners `6/8`; no held-out test, Sheet update, Hugging Face
  action, or General submission occurred.

- AfriHG LR-0 `1194878` and LR-1 `1194979` completed cleanly `0:0` in
  `16:01:56/16:02:44`; the terminal logs report fine-tuning complete and final
  adapters saved at `16:25:13/16:26:00`. Both stopped at epoch-3 step `9246`
  under the frozen patience rule and retained epoch-1 `checkpoint-3082` as
  best; with nonnegative chrF and the verified epoch-1/2 best metric `0.0`, the
  terminal selection metric remained tied at `0.0`. Final adapter/config
  SHA-256 values are
  `dc6e3c1aee872b86bc005d0b375efc55eb03172d095a246789bd99ac8a90db09` /
  `723872cc560f7fda5aafe2560bd70a0f24a456fc8ef183a57991149e1ac3eb9c`
  for LR-0 and
  `5dbb6e3e68443d02215bf55b4aad793aa7683eca6e28b1e8701d14df3ba181a2` /
  `35f3db3aa551d4754122139992d3bf6d9878e0f90f8df98c0788cf96a8d0a3d3`
  for LR-1. AfriHG remains unfrozen until LR-2 `1195667` terminates.

- After verifying two released slots, no General artifact/active duplicate,
  and matching local/HEX hashes for the preregistered runner, launcher,
  loss-only callback gate, and token-balanced config, General LR-0 `1204261`
  (`3e-5`) and LR-1 `1204262` (`8e-5`) were submitted individually on
  A100-40GB. LR-0 started at `16:37:00` on `srvrocgpu010`; it passed the pure
  `GatedDeltaNetForCausalLM`, BF16, fast-GDN, exact ten-target LoRA
  (`4,649,088` trainable parameters), validation-loss selection,
  task-metric-disable, and Hub-disable gates. It built the frozen token-balanced
  train/eval sets of `43,637/12,209` examples and was tokenizing at 16:40 with
  no fault marker; W&B run is `4dk0wnt7`. It entered training at `16:39:59`
  over `13,640` planned steps and reached step 2 with finite execution; the
  initial `7--8 s/step` is too early for a reliable terminal ETA and creates a
  provisional 24-hour wall-time risk that does not justify changing the frozen
  recipe. LR-1 remained resource-pending as job `1204262`; Slurm's conservative
  start estimate was `2026-08-09 04:49:05 SAST`, so it did not start or consume
  a GPU. Active owned GPU work was
  exactly AfriHG LR-2 `1195667` plus General LR-0 `1204261`, both A100-40GB
  `gpu:ampere`; no A100-80GB/L40S overlap. HEX quota was `/home` `3/10 GB`
  (`32.5%`) and `/scratch` `107/300 GB` (`35.9%`). Kombuys remained read-only
  and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`; only
  pre-existing `tailscale-kombuys` tmux). Base remains `16/16`, family winners
  `6/8`; no held-out test, Sheet update, or Hugging Face action occurred.

- At 17:06 SAST, AfriHG LR-2 `1195667` remained healthy at epoch-3 step
  `8119/15410` near `3.1 s/step`, with finite recent loss/gradient records and
  no fault marker. Its raw epoch-3 training boundary is approximately
  `18:05 SAST`, followed by the terminal full-validation callback; conditional
  terminal ETA remains near `20:40 SAST`. General LR-0 `1204261` was healthy
  at step `248/13640` near `6.16 s/step`, with current logs and no fault
  marker. This puts its raw epoch-1 training boundary near `21:20 SAST` before
  full validation; a terminal ETA is not defensible until the first validation
  runtime and patience trajectory are observed, and five full epochs would
  create 24-hour wall-time risk. General LR-1 `1204262` remained
  resource-pending with conservative scheduler start
  `2026-08-09 04:49:05 SAST`. Active owned work was exactly AfriHG LR-2 plus
  General LR-0 on two A100-40GB `gpu:ampere` allocations; no A100-80GB/L40S
  overlap. HEX quota remained `/home` `3/10 GB` (`32.5%`) and `/scratch`
  `107/300 GB` (`35.9%`). Kombuys remained read-only and idle (RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`; only pre-existing
  `tailscale-kombuys` tmux). Base remains `16/16`, family winners `6/8`; no
  held-out test, Sheet update, Hugging Face action, or new submission occurred.

- At 17:36 SAST, AfriHG LR-2 `1195667` remained healthy at epoch-3 step
  `8699/15410` near `3.05 s/step`, about 28 raw training minutes from the
  terminal validation boundary. Conditional completion remains near
  `20:40 SAST` after the full task-native callback. General LR-0 `1204261`
  remained healthy at step `534/13640` near `6.24 s/step`, putting its raw
  epoch-1 boundary near `21:24 SAST` before validation. General LR-1 `1204262`
  remained resource-pending with conservative scheduler start
  `2026-08-09 04:49:05 SAST`. Logs were current and contained no fault marker.
  Exactly two A100-40GB `gpu:ampere` jobs were active on `srvrocgpu010`; no
  A100-80GB/L40S overlap. HEX quota remained `/home` `3/10 GB` (`32.5%`) and
  `/scratch` `107/300 GB` (`35.9%`). Kombuys remained read-only and idle (RTX
  5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`; only pre-existing
  `tailscale-kombuys` tmux). Base remains `16/16`, family winners `6/8`; no
  held-out test, Sheet update, Hugging Face action, or submission occurred.

- At 18:07 SAST, AfriHG LR-2 `1195667` reached epoch-3 step `9246` at
  `18:02:20` and entered trainer validation. The log remained current with
  finite loss/gradient and no fault marker; the epoch-3 trainer result and
  terminal task-native callback had not yet appeared. Conditional completion
  remains near `20:40--20:45 SAST`. General LR-0 `1204261` remained healthy
  near step `819/13640`, preserving its raw epoch-1 boundary near
  `21:20--21:25 SAST`. General LR-1 `1204262` started at `18:03:48` on
  `srvrocgpu010`, passed the pure `GatedDeltaNetForCausalLM`, BF16, fast-GDN,
  exact ten-target LoRA (`4,649,088` trainable parameters), validation-loss,
  token-balanced `43,637/12,209` train/eval, task-metric-disable, and Hub-disable
  gates, then entered training at `18:06:24` and reached step 2 without a fault
  marker. Exactly three A100-40GB `gpu:ampere` jobs were now active; no
  A100-80GB/L40S overlap and no capacity for General LR-2. HEX quota remained
  `/home` `3/10 GB` (`32.5%`) and `/scratch` `107/300 GB` (`35.9%`). Kombuys
  remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`; only pre-existing `tailscale-kombuys` tmux). Base remains
  `16/16`, family winners `6/8`; no held-out test, Sheet update, Hugging Face
  action, or new submission occurred.

- AfriHG LR-2 `1195667` completed cleanly `0:0` at `20:43:43 SAST` after
  epoch 3. Its terminal trainer validation loss was the health-only
  `0.7731777429580688`; task-native chrF remained `0.0`, and
  `checkpoint-3082` remained the restored best with `best_metric=0.0`.
  Fine-tuning completed and the final adapter was saved at `20:43:37`. The
  final adapter/config SHA-256 values are
  `dc4cbbf124121aafb44ddddca775d5f16bbb878ea4fd700f9f4e15be3088689c` /
  `6f9169df6bdd3f6b1ef5cdbe2043f823c12bb60fbe0096e60bddabbdb79a03c7`.
  Since all three AfriHG LRs tied at chrF `0.0`, the preregistered lower-LR
  tie-break freezes AfriHG to LR `3e-5`, job `1194878`, epoch-1
  `checkpoint-3082`. This raises frozen family winners to `7/8`.

- By the 23:32 SAST reconciliation, General LR-0 `1204261` and LR-1
  `1204262` had both produced valid epoch-1 `checkpoint-2728` selection
  artifacts with validation losses `13.603466245839952` and
  `13.732546259124712`, respectively. LR-0 adapter/config/trainer-state
  SHA-256 values are
  `5c1c08009f5c83bba61a40c998b5e6733f754068dd46168fca81ba1f5bcd1b5d` /
  `f2d56f5fed78c09f0d4e6a5e96a374cc46fc45c91c579aacc8c222ce75af419a` /
  `b5dad80258441a283823304241b94d16e604c46c1e1a6d9db62e2bab473a9bed`;
  LR-1 values are
  `ac4fbca21f6dd2b0df0072faed6d6553813de6e5903ca5bcc50a9091623fd4b9` /
  `b8a895382a87dc48f5a5ef06c2aab19f33b6875ad50e453b3252487ebe268a6b` /
  `94a12d58edd30c3ca54d0185f8e0375071783625592c07a31f5b6b6a141013cb`.
  Both jobs remained healthy in epoch 2 near steps `3891/3070`, with no fault
  marker. After confirming the released slot, no LR-2 artifact or active
  duplicate, exact A100-40GB-only state, and syncing the frozen runner,
  General LR-2 was submitted individually as job `1207524`. It started on
  `srvrocgpu010` at `23:29:48`, passed the fast-GDN, canonical pure
  `GatedDeltaNetForCausalLM`, BF16, exact ten-target LoRA (`4,649,088`
  trainable parameters), validation-loss, Hub-disabled, and token-balanced
  `43,637/12,209` dataset gates. It entered training and reached step
  `3/13640` near `6.70 s/step` without a fault marker. Exactly three owned
  A100-40GB `gpu:ampere` jobs were active, with no A100-80GB/L40S overlap.
  Conditional next validation windows are about `02:15`, `03:45`, and
  `04:25 SAST` for LR-0/LR-1/LR-2; terminal timing remains patience-dependent
  and the 24-hour
  limit is a material risk if all five epochs are needed. HEX quota was
  `/home` `3/10 GB` (`32.6%`) and `/scratch` `108/300 GB` (`36.0%`). Kombuys
  remained read-only and idle (RTX 5090 `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`,
  scratch `61%`; only pre-existing `tailscale-kombuys` tmux). No held-out
  adapter test, Sheet write, or Hugging Face action occurred.

- At 23:56 SAST, General LR-0/LR-1/LR-2 jobs
  `1204261/1204262/1207524` were healthy at steps
  `4125/3306/222` of `13640`, with current finite training records and zero
  fault markers. LR-0/LR-1 retained their verified epoch-1 validation-loss
  checkpoints; LR-2 had no validation artifact yet. Updated conditional next
  validation windows are approximately `02:15--02:20`, `03:40--03:45`, and
  `04:10--04:20 SAST`, respectively. Exactly three owned A100-40GB
  `gpu:ampere` jobs were active on `srvrocgpu010`, with no A100-80GB/L40S
  overlap. HEX quota remained `/home` `3/10 GB` (`32.6%`) and `/scratch`
  `108/300 GB` (`36.0%`). Kombuys remained read-only and idle (RTX 5090
  `10 MiB/0%`, RTX 3080 Ti `1 MiB/0%`, scratch `61%`; only pre-existing
  `tailscale-kombuys` tmux). Base remains `16/16`, frozen family winners
  `7/8`; no held-out adapter test, Sheet write, Hugging Face action, or new
  submission occurred.
