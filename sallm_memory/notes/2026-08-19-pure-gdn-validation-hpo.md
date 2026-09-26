# Pure-GDN validation-only HPO — 2026-08-19

## AfriHG a2 advances toward checkpoint 9246 — 22:21 SAST

- A2 job `1246940` remains healthy and fault-free at `8669/15410` after
  `9:54:50` on `srvrocgpu010` A100-40GB. At about `3.1 s/step`, checkpoint
  boundary 9246 remains near `22:50--23:00 SAST`; terminal completion remains
  tentatively `07:30--08:00 SAST` on 20 August after the remaining full
  validations and frozen exact callbacks.
- A2 retains checkpoint 6164 at validation-only mean chrF
  `23.449880263733377`. No new selector artifact exists and no AfriHG winner
  is frozen. Operationally jobs `1246938/1246939` are complete and `1246940`
  is running; scientifically AfriHG remains `2/3` terminal-valid and the
  global freeze remains `0/8`.
- A2 is the only owned job and requests exactly one A100-40GB
  `gpu:ampere`; no A100-80GB/L40S work overlaps. HEX quota is home `88.6%`,
  scratch `40.2%`; Kombuys remains read-only with RTX 5090 untouched,
  held-out adapter evaluation is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

## AfriHG a1 completes terminal reconciliation — 21:50 SAST

- A1 job `1246939` completed `0:0` at `21:38:34 SAST` after saving its
  terminal exact artifact and final adapter. The terminal artifact has exactly
  `128` rows at global step 15410, split `64/64` Xho/Zul, with Xho/Zul chrF
  `23.35450117522189/25.122629092549037` and terminal mean
  `24.238565133885464`. The retained validation-only best therefore remains
  checkpoint 12328 at mean chrF `24.316085107021614`; the lower terminal mean
  does not replace it.
- The terminal audit found no exact-empty, whitespace-only, normalized-empty,
  prediction-empty, or debug-empty output; Xho/Zul had `64/63` unique
  predictions. All 128 prompts start with `[BOS]`, end with
  `[EOS]<|assistant|>`, and have zero EOS tokens after the assistant marker.
  The retained checkpoint BIN and final adapter safetensors contain the same
  424 keys and all `71,762,560` tensor values compare exactly, with no missing,
  extra, or mismatched tensor.
- SHA-256 values are terminal artifact
  `1a6963e6a1308ad2c77fc0df50b58f8a66035b64d011d0df2139e9bbb7c9b295`,
  retained trainer state
  `37e19be9ba81365273f6d5b4f7be521cd335d63825e392ad5c329cc59bcd16ab`,
  byte-identical retained/final adapter config
  `de73c7e95eaff81e208c70b8c8bb5fef35c7069103567f4ed8327bff8a91f428`,
  retained weights
  `317dcf88b4a92a8852159186184f837205b88643acd2ccdea8c0fe7879734876`,
  final weights
  `f43e5ed0f199026e13421ed2b3f3d610a46d2e5c18e2a534f4c48c2cf0982ee8`,
  execution manifest
  `32e43a2bd448aadf34002bcc9078768a0db618278439b960561e3c3e3382fd72`,
  and trial record
  `c4225ab462a7881238e552e93e965fdfa27e667c88bda5ac6084c9ded770d34a`.
- A1 is scientifically terminal-valid, advancing AfriHG from `1/3` to `2/3`
  terminal-valid. A2 job `1246940` remains healthy near `8087/15410` on
  `srvrocgpu010`, retaining checkpoint 6164 at mean chrF
  `23.449880263733377`; its next exact boundary is 9246 around
  `22:50--23:00 SAST`, with tentative terminal completion around
  `07:30--08:00 SAST` on 20 August. No winner is frozen until a2 completes
  terminal reconciliation.
- A2 is the only owned job and uses A100-40GB `gpu:ampere`; no
  A100-80GB/L40S work overlaps. HEX quota is home `88.6%`, scratch `40.2%`;
  Kombuys remains read-only with RTX 5090 untouched, held-out adapter
  evaluation is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

## AfriHG a1 terminal callback reaches second language — 21:20 SAST

- A1 job `1246939` remained healthy in its frozen terminal exact callback on
  `srvrocgpu010` A100-40GB. Its second language phase established automatic
  generation batch size 64 at `21:04:36 SAST`; the terminal step-15410 JSONL
  was still absent at `21:20:47`, with no targeted fault marker.
- Prior callback timing keeps the exact 128-row terminal artifact around
  `21:35--21:50 SAST`. A1 remains non-terminal-valid and checkpoint 12328 at
  mean chrF `24.316085107021614` remains the eligible validation-only best
  until the terminal artifact and retained/final adapter are reconciled.
- A2 job `1246940` remained healthy near `7515/15410`, retaining audited
  checkpoint 6164 at mean chrF `23.449880263733377`. Operationally AfriHG is
  one complete, one in terminal callback, and one running; scientifically it
  remains `1/3` terminal-valid with no winner frozen.
- Jobs `1246939/1246940` are the only owned work and both use A100-40GB
  `gpu:ampere`; no A100-80GB/L40S work overlaps. HEX quota is home `88.6%`,
  scratch `40.1%`; Kombuys remains read-only with RTX 5090 untouched,
  held-out adapter evaluation is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

## AfriHG a1 enters terminal exact callback — 20:51 SAST

- A1 job `1246939` reached exactly `15410/15410` at `20:32 SAST`, then
  completed terminal full declared validation over all `3,082` rows at
  health-only loss `2.2272494524659314` in `138.7424 s`. This loss is not
  selector evidence; checkpoint 12328 remains the eligible validation-only
  best at mean chrF `24.316085107021614` until the terminal exact artifact is
  complete and reconciled.
- The frozen terminal exact callback established automatic generation batch
  size 64 for its first segment at `20:36:31 SAST`. At `20:51:55`, the
  step-15410 JSONL was still absent and no targeted fault marker existed.
  Prior callback timing puts the terminal artifact around
  `21:35--21:50 SAST`; a1 remains non-terminal-valid until the artifact,
  retained checkpoint, and final adapter are reconciled.
- A2 job `1246940` remained healthy near `6933/15410`, retaining audited
  checkpoint 6164 at mean chrF `23.449880263733377`. Operationally AfriHG is
  one complete, one in terminal callback, and one running; scientifically it
  remains `1/3` terminal-valid with no winner frozen.
- Jobs `1246939/1246940` are the only owned work and both use A100-40GB
  `gpu:ampere`; no A100-80GB/L40S work overlaps. HEX quota is home `88.6%`,
  scratch `40.1%`; Kombuys remains read-only with RTX 5090 untouched,
  held-out adapter evaluation is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

## AfriHG a2 step 6164 becomes retained validation best — 20:20 SAST

- A2 job `1246940` completed its frozen step-6164 exact callback at
  `20:10:12 SAST` and resumed healthy training near `6362/15410` on
  `srvrocgpu010` A100-40GB. The artifact has exactly `128` rows, `64` Xho
  plus `64` Zul, all at global step 6164, with no exact-empty,
  whitespace-only, normalized-empty, or debug-empty prediction and `64/64`
  unique predictions per language.
- Xho/Zul validation chrF is `22.762274809600555/24.137485717866202`; mean
  chrF `23.449880263733377` exactly matches `trainer_state.json`
  `best_metric`, `best_global_step=6164`, and retained `checkpoint-6164`.
  This improves a2 checkpoint 3082 mean `22.763135043827063`. Artifact,
  trainer-state, and adapter-config SHA-256 values are
  `7d4ec097ce2ec64cfbafe079c93e0276f66cb6d0612381103314f8748413d08e`,
  `e9e9425e5448a6d4edc8d14c6721d870b830977ca1d21c6aec30a14070823091`,
  and `04ca6881ae6b2d040d1315a33882080a84673433be9030752efc01742d0f8985`.
  All 128 prompts start with `[BOS]`, end with
  `[EOS]<|assistant|>`, and have no EOS after the assistant marker.
- This is valid within-run validation evidence only: a2 is not terminal-valid
  and no AfriHG winner is selected. A1 job `1246939` remains healthy near
  `15192/15410`, retaining audited checkpoint 12328 at mean chrF
  `24.316085107021614`; its final step boundary is about 11 minutes away,
  after which full validation and the final exact callback still must finish.
- Operationally AfriHG is one complete plus two running; scientifically it
  remains `1/3` terminal-valid. Jobs `1246939/1246940` are the only owned
  work, targeted fault scans are empty, and both use A100-40GB `gpu:ampere`
  with no A100-80GB/L40S overlap. HEX quota is home `88.6%`, scratch
  `40.1%`; Kombuys remains read-only with RTX 5090 untouched, held-out
  adapter evaluation is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

## AfriHG a2 enters step 6164 exact callback — 19:20 SAST

- A2 job `1246940` reached exactly `6164/15410`, completed full declared
  validation over all `3,082` rows at health-only loss
  `2.188938227510545` in `138.7629 s`, and established automatic generation
  batch size 64 for the first segment of its frozen exact callback at
  `19:02:49 SAST`. The loss is not selector evidence; retained checkpoint
  3082 at mean chrF `22.763135043827063` remains eligible until the exact
  step-6164 artifact is complete and reconciled. Prior callback timing puts
  the complete 128-row artifact around `20:05--20:20 SAST`.
- A1 job `1246939` remains healthy near `14047/15410`, retaining audited
  checkpoint 12328 at mean chrF `24.316085107021614`. At about
  `3.13 s/step`, its final step-15410 boundary remains near `20:32 SAST`,
  followed by full validation and its final frozen exact callback.
- Operationally AfriHG is one complete plus two running; scientifically it
  remains `1/3` terminal-valid and no winner is frozen. Jobs
  `1246939/1246940` are the only owned work, targeted fault scans are empty,
  and both use A100-40GB `gpu:ampere` with no A100-80GB/L40S overlap. HEX
  quota is home `88.6%`, scratch `40.1%`; Kombuys remains read-only with RTX
  5090 untouched, held-out adapter evaluation is `0`, Sheet E/F/G remain
  blank, and General/Monolingual/publication remain blocked.

## AfriHG a1 step 12328 becomes retained validation best — 18:21 SAST

- A1 job `1246939` completed its frozen step-12328 exact callback at
  `17:50:07 SAST` and resumed healthy training to about `12923/15410` on
  `srvrocgpu010` A100-40GB. The artifact has exactly `128` rows, `64` Xho
  plus `64` Zul, all at global step 12328, with no exact-empty,
  whitespace-only, normalized-empty, or debug-empty prediction. Xho has 64
  unique predictions and Zul 63.
- Xho/Zul validation chrF is `23.372141107936447/25.260029106106785`; mean
  chrF `24.316085107021614` exactly matches `trainer_state.json`
  `best_metric`, `best_global_step=12328`, and retained `checkpoint-12328`.
  This narrowly improves the previous a1 retained checkpoint 9246 mean
  `24.23494200032799`. Artifact, trainer-state, and adapter-config SHA-256
  values are
  `b097141d7926648a5b87b3a75769c8b342699ebc0ac7af243fa12bfb2f12ea5c`,
  `37e19be9ba81365273f6d5b4f7be521cd335d63825e392ad5c329cc59bcd16ab`,
  and `de73c7e95eaff81e208c70b8c8bb5fef35c7069103567f4ed8327bff8a91f428`.
  All 128 prompts start with `[BOS]`, end with
  `[EOS]<|assistant|>`, and have no EOS after the assistant marker.
- This is valid within-run validation evidence only: a1 is not terminal-valid
  and no AfriHG winner is selected. At `18:21 SAST`, a2 job `1246940` is
  also RUNNING cleanly near `5456/15410`; a1/a2 targeted fault scans are
  empty. Operationally AfriHG is one complete plus two running;
  scientifically it remains `1/3` terminal-valid. A1 is tentatively due at
  its final step-15410 boundary around `20:32 SAST`, then needs its last
  exact callback; a2's next step-6164 boundary is around `19:00 SAST`.
- These are the only owned jobs and use A100-40GB `gpu:ampere`; no
  A100-80GB/L40S work overlaps. HEX quota is home `88.6%`, scratch `40.1%`;
  Kombuys remains read-only with RTX 5090 untouched, held-out adapter
  evaluation is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

## AfriHG a2 first exact checkpoint reconciles; a1 enters step 12328 callback — 16:46 SAST

- A2 job `1246940` completed its frozen step-3082 exact callback at
  `16:17:01 SAST` and resumed clean training past `3658/15410` on
  `srvrocgpu010` A100-40GB. The artifact has exactly `128` rows, `64` Xho
  plus `64` Zul, all at global step 3082, with no exact-empty or
  whitespace-only raw output, no normalized-empty prediction, and `64/64`
  unique predictions per language.
- Xho/Zul validation chrF is `21.862952647458833/23.663317440195296`; mean
  chrF `22.763135043827063` exactly matches `trainer_state.json`
  `best_metric`, `best_global_step=3082`, and retained `checkpoint-3082`.
  Artifact, trainer-state, and adapter-config SHA-256 values are
  `00adcb61c3432a3c448db240e3da7ed5657e2e962d538625a35f9bc15c9ef761`,
  `14c1b5f4f4b866b454072a2cdf70def39b5692f632598b7a261ef448aaee0182`,
  and `04ca6881ae6b2d040d1315a33882080a84673433be9030752efc01742d0f8985`.
  All 128 prompts start with `[BOS]`, end with `[EOS]<|assistant|>`, and have
  no EOS after the assistant marker. This is valid within-run validation
  evidence only: a2 is not terminal-valid and no AfriHG winner is selected.
- A1 job `1246939` reached `12328/15410`, completed full declared validation
  over all `3,082` rows at health-only loss `2.2190199647144713` in
  `140.1820 s`, and established automatic generation batch size 64 for its
  frozen exact callback at `16:46:37 SAST`. The loss is not selector evidence;
  retained checkpoint 9246 remains the eligible a1 best until the exact
  artifact reconciles. Prior callback timing puts that artifact around
  `17:45--18:00 SAST`.
- Operationally AfriHG is one complete plus two running; scientifically it
  remains `1/3` terminal-valid with no winner frozen. Jobs `1246939/1246940`
  are the only owned work and use A100-40GB `gpu:ampere`; targeted fault scans
  are empty and no A100-80GB/L40S work overlaps. HEX quota is home `88.6%`,
  scratch `40.1%`; Kombuys remains read-only with RTX 5090 untouched,
  held-out adapter evaluation is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

## AfriHG a2 exact callback continues — 16:16 SAST

- A2 job `1246940` remains RUNNING cleanly in its frozen step-3082 exact
  callback on `srvrocgpu010` A100-40GB. No completed artifact or targeted
  fault marker exists; its checkpoint root still contains only the execution
  manifest and frozen trial record. The observed callback duration shifts the
  complete 128-row artifact ETA slightly to `16:20--16:35 SAST`.
- A1 job `1246939` is healthy near `11824/15410`, retaining audited
  checkpoint 9246 at mean chrF `24.23494200032799`. It is about 504 steps,
  or 26 minutes at roughly `3.13 s/step`, from its step-12328 boundary near
  `16:42 SAST`.
- Operationally AfriHG is one complete plus two running; scientifically it is
  `1/3` terminal-valid and no winner is frozen. HEX quota is home `88.6%`,
  scratch `40.1%`; all owned jobs are A100-40GB, Kombuys is read-only with
  RTX 5090 untouched, held-out is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

## AfriHG a2 callback reaches second language — 15:46 SAST

- A2 job `1246940` remains healthy in its frozen step-3082 exact callback on
  `srvrocgpu010` A100-40GB. The second language phase established automatic
  generation batch size 64 at `15:36:04 SAST`; no completed artifact or
  targeted fault marker exists yet. The checkpoint root still contains only
  its execution manifest and frozen trial record.
- Prior callback timing keeps the complete 128-row artifact ETA around
  `16:10--16:25 SAST`. Until exact reconciliation, a2 has no selector
  evidence or eligible retained checkpoint.
- A1 job `1246939` is healthy near `11254/15410`, retaining audited
  checkpoint 9246 at mean chrF `24.23494200032799`. At the current roughly
  `3.15 s/step`, its step-12328 boundary is due around `16:42 SAST`, with the
  exact artifact tentatively around `17:45--18:00`.
- Operationally AfriHG is one complete plus two running; scientifically it is
  `1/3` terminal-valid and no winner is frozen. HEX quota is home `88.6%`,
  scratch `40.1%`; all owned jobs are A100-40GB, Kombuys is read-only with
  RTX 5090 untouched, held-out is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

## AfriHG a2 first exact callback active — 15:16 SAST

- A2 job `1246940` reached exactly `3082/15410` cleanly on
  `srvrocgpu010` A100-40GB. Full declared validation covered all `3,082`
  rows at health-only loss `2.233357958635829` in `140.3220 s`; the frozen
  exact callback then established automatic generation batch size 64 at
  `15:09:52 SAST`.
- No completed step-3082 exact artifact or targeted fault marker exists yet.
  Prior callback timing keeps the complete 128-row artifact ETA around
  `16:10--16:25 SAST`. The health-only loss is not selector evidence, and a2
  has no eligible validation-only checkpoint until exact reconciliation.
- A1 job `1246939` remains healthy near `10682/15410`, retaining its audited
  checkpoint 9246 at mean chrF `24.23494200032799`.
- Operationally AfriHG is one complete plus two running; scientifically it is
  `1/3` terminal-valid and no winner is frozen. HEX quota is home `88.6%`,
  scratch `40.1%`; all owned jobs are A100-40GB, Kombuys is read-only with
  RTX 5090 untouched, held-out is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

## AfriHG a2 nears first validation boundary — 14:46 SAST

- Jobs `1246939` and `1246940` remain RUNNING on `srvrocgpu010`
  A100-40GB with no targeted fault marker. A1 is near `10124/15410` after
  cleanly resuming from its audited step-9246 callback; its retained
  validation-only best remains checkpoint 9246 at mean chrF
  `24.23494200032799`.
- A2 is near `2704/15410` at about `3.04 s/step`, leaving `378` training
  steps, or about 19 minutes, to its first full-validation boundary at step
  3082. The boundary remains due around `15:05 SAST`; the frozen exact
  128-row AfriHG artifact remains tentatively due around `16:10--16:25` if
  callback timing remains stable. No selector evidence exists for a2 yet.
- Operationally AfriHG is one complete plus two running; scientifically it is
  `1/3` terminal-valid and no winner is frozen. HEX quota is home `88.6%`,
  scratch `40.1%`; all owned jobs are A100-40GB, Kombuys is read-only with
  RTX 5090 untouched, held-out is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

## AfriHG a1 step-9246 exact checkpoint reconciles — 14:17 SAST

- A1 job `1246939` completed its frozen step-9246 exact callback at
  `13:59:45 SAST` and resumed training on `srvrocgpu010` A100-40GB. The
  artifact has exactly `128` rows, `64` Xho plus `64` Zul, all at global step
  9246, with no exact-empty or whitespace-only raw output, no normalized-empty
  prediction, and `64/64` unique predictions per language.
- Xho/Zul validation chrF is `23.517448141947057/24.952435858708927`; exact
  mean `24.23494200032799` matches `trainer_state.json` `best_metric`,
  `best_global_step=9246`, and retained `checkpoint-9246`, superseding a1
  checkpoint 6164 within the same run. Artifact and trainer-state SHA-256
  values are
  `26aec28ad225009aa6295ea7059e8c365cf70c1e49a1f1ed5eb0f02adfbee9ff`
  and `92ce92d557b3c9a644cef34445329080f529a135c26eb60f49082de46caf609d`.
- All 128 prompts start with `[BOS]`, end with `[EOS]<|assistant|>`, and have
  no EOS after the assistant marker. This is scientifically valid within-run
  validation evidence only: a1 is not terminal-valid and no AfriHG winner is
  selected.
- A2 job `1246940` remains healthy near `2107/15410`. Its first step-3082
  boundary remains due around `15:05 SAST`, with the exact artifact around
  `16:10--16:25` if callback timing remains stable.
- Operationally AfriHG is one complete plus two running; scientifically it is
  `1/3` terminal-valid. HEX quota is home `88.6%`, scratch `40.1%`; all jobs
  are A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

## AfriHG a1 step-9246 callback reaches second language — 13:46 SAST

- A1 job `1246939` remains healthy in its frozen step-9246 exact callback on
  `srvrocgpu010` A100-40GB. Automatic generation batch size 64 was
  established for the second language phase at `13:26:44 SAST`; no targeted
  fault marker appears.
- The step-9246 JSONL remains absent. Prior callback timing keeps the
  complete-artifact ETA around `14:00--14:15 SAST`; a1 remains non-terminal
  and checkpoint 6164 is still its eligible validation-only best.
- A2 job `1246940` is healthy near `1507/15410`. Its first step-3082 boundary
  remains due around `15:05 SAST`, with the exact artifact around
  `16:10--16:25` if callback timing remains stable.
- Operationally AfriHG is one complete plus two running; scientifically it is
  `1/3` terminal-valid. HEX quota is home `88.6%`, scratch `40.1%`; all jobs
  are A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

## AfriHG a1 step-9246 exact callback active — 13:16 SAST

- A1 job `1246939` reached `9246/15410` cleanly on `srvrocgpu010`
  A100-40GB. Full declared validation covered all `3,082` rows at health-only
  loss `2.211229248839025` in `139.0600 s`; the frozen exact callback then
  established automatic generation batch size 64 at `12:59:09 SAST`.
- No step-9246 JSONL or targeted fault marker exists yet. Prior callback
  timing gives a complete-artifact ETA around `14:00--14:15 SAST`; the
  health-only loss is not selector evidence, and checkpoint 6164 remains
  a1's eligible validation-only best.
- A2 job `1246940` is healthy near `910/15410` on the second A100-40GB. Its
  first step-3082 boundary is tentatively due around `15:05 SAST`, with the
  exact artifact around `16:10--16:25` if callback timing remains stable.
- Operationally AfriHG is one complete plus two running; scientifically it is
  `1/3` terminal-valid. HEX quota is home `88.6%`, scratch `40.1%`; all jobs
  are A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

## AfriHG a0 terminal-valid; a2 starts — 12:48 SAST

- A0 job `1246938` completed `0:0` at `12:26:26 SAST` after `19:08:18`.
  Its terminal step-15410 artifact covers exactly `128` rows, `64` Xho plus
  `64` Zul, with no exact-empty or whitespace-only raw output, no normalized-
  empty prediction, and `64/64` unique predictions per language. All prompts
  start with `[BOS]`, end with `[EOS]<|assistant|>`, and have no EOS after the
  assistant marker.
- Terminal Xho/Zul chrF is `21.82177467761243/23.335156227554975`; exact
  mean `22.578465452583703` is below the retained validation-only best
  `22.843278831141802`, so the frozen selector correctly keeps checkpoint
  12328. Terminal-artifact, retained-state, and trial SHA-256 values are
  `a8b785757a497e8780c5d0b5f609e42ffa2dbcc63a43ea826165afc7ca483b75`,
  `6ceeefc3bfc763016687c9966467fdbc15c8b28ce3fe32b510b9a6ae2651dbab`,
  and `01d022dc5d714dd4844fee33b504b0b2af13cdff26600cfcad0c5ed8f3a76f62`.
- A local read-only comparison proved exact equality for all `424/424`
  tensors (`71,762,560` values) between retained checkpoint 12328 and
  `final_adapter`, with zero missing, extra, or mismatched keys. Retained and
  final weight SHA-256 values are
  `0a9f21a998c9c4dcbc54e0db9f9f0b5f85c69309f076928809cd6da1408d8350`
  and `ab7331f5d0c88fe102e2d1e8621ed171056959f20588875393b533d7ef66c728`;
  both configs are byte-identical at
  `0ca737e9db8182cb61e882fb827f1d9917015e7743c168b5156f1cbe208f3aff`.
  A0 is therefore scientifically terminal-valid, advancing AfriHG to `1/3`;
  no family winner is frozen.
- A2 job `1246940` started automatically at `12:26:26 SAST` on the freed
  `srvrocgpu010` A100-40GB and is healthy past step 355. Its execution-
  manifest and trial SHA-256 values are
  `03ac17c961a2f9ca96644f11b8bd3058f70b6f37fa57b777cb6c4d4b910e5d3d`
  and `a85ed5f5e3e4d15f8e3ffb00157fafc0236cd1363d88797ad764ab505c07ab11`;
  the frozen a2 recipe is LR `1.5e-4`, rank/alpha `16/32`, dropout `0.05`,
  warmup `0.03`, validation-only selection. A1 job `1246939` remains healthy
  near `9073/15410`, retaining validated checkpoint 6164; its step-9246 exact
  artifact is tentatively due around `14:00--14:15`.
- Operationally AfriHG is one complete plus two running; scientifically it is
  `1/3` terminal-valid. HEX quota is home `88.6%`, scratch `40.1%`; all jobs
  are A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

## AfriHG a0 terminal callback reaches second language — 12:16 SAST

- A0 job `1246938` remains healthy in its frozen terminal exact callback on
  `srvrocgpu010` A100-40GB. Automatic generation batch size 64 was
  established for the second language phase at `11:50:42 SAST`; no targeted
  fault marker appears.
- The step-15410 JSONL remains absent. Prior callback timing narrows the
  terminal-artifact ETA to about `12:20--12:35 SAST`; a0 remains non-terminal-
  valid until that artifact and retained checkpoint are reconciled.
- A1 job `1246939` is healthy near `8501/15410`, retaining validated
  checkpoint 6164 at mean chrF `23.199728020008237`; its step-9246 boundary
  is tentatively due around `12:55 SAST`. A2 job `1246940` remains
  Resources-pending with estimate `14:10:13`.
- AfriHG is operationally two running plus one pending but scientifically
  `0/3` terminal-valid. HEX quota is home `88.6%`, scratch `39.8%`; all jobs
  are A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

## AfriHG a0 terminal exact callback active — 11:46 SAST

- A0 job `1246938` completed terminal full declared validation over all
  `3,082` rows at health-only loss `2.25601756735176` in `140.8837 s`, then
  entered the frozen terminal exact callback on `srvrocgpu010` A100-40GB.
  Automatic generation batch size 64 was established at `11:21:02 SAST`;
  no targeted fault marker appears.
- The step-15410 JSONL remains absent. Prior callback timing keeps the
  terminal-artifact ETA around `12:20--12:35 SAST`; the health-only loss is
  not selector evidence, and a0 is not terminal-valid until the exact
  artifact and retained checkpoint are reconciled.
- A1 job `1246939` is healthy near `7924/15410`, retaining validated
  checkpoint 6164 at mean chrF `23.199728020008237`. A2 job `1246940`
  remains Resources-pending with estimate `14:10:13`.
- AfriHG is operationally two running plus one pending but scientifically
  `0/3` terminal-valid. HEX quota is home `88.6%`, scratch `39.8%`; all jobs
  are A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

## AfriHG a0 reaches terminal training boundary — 11:17 SAST

- A0 job `1246938` reached exactly `15410/15410` training steps cleanly on
  `srvrocgpu010` A100-40GB and entered the frozen terminal evaluation path.
  No terminal validation metric, exact JSONL, or targeted fault marker exists
  yet. Allowing the full declared pass and exact callback keeps the terminal-
  artifact ETA around `12:20--12:35 SAST`.
- Until that terminal artifact is audited, checkpoint 12328 and mean chrF
  `22.843278831141802` remain a0's latest eligible within-run selector
  evidence; a0 is not yet terminal-valid.
- A1 job `1246939` remains healthy near `7348/15410`, retaining validated
  checkpoint 6164 at mean chrF `23.199728020008237`. A2 job `1246940`
  remains Resources-pending with estimate `14:10:13`.
- AfriHG is operationally two running plus one pending but scientifically
  `0/3` terminal-valid. HEX quota is home `88.6%`, scratch `39.8%`; all jobs
  are A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

## AfriHG a0 approaches terminal boundary — 10:46 SAST

- A0 job `1246938` remains healthy near `14822/15410` on `srvrocgpu010`
  A100-40GB, with no targeted fault marker. At current throughput, its final
  training boundary is due around `11:16--11:20 SAST`; allowing full declared
  validation and the frozen exact callback gives a terminal-artifact ETA near
  `12:20--12:35`.
- A1 job `1246939` is healthy near `6780/15410`, retaining validated
  checkpoint 6164 at mean chrF `23.199728020008237`. A2 job `1246940`
  remains Resources-pending with estimate `14:10:13`.
- AfriHG is operationally two running plus one pending but scientifically
  `0/3` terminal-valid. HEX quota is home `88.6%`, scratch `39.8%`; all jobs
  are A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

## AfriHG a1 step-6164 exact checkpoint reconciles — 10:16 SAST

- A1 job `1246939` completed its frozen step-6164 exact callback at
  `10:13:09 SAST` and resumed training on `srvrocgpu010` A100-40GB. The
  artifact has exactly `128` rows, `64` Xho plus `64` Zul, all at global step
  6164, with no exact-empty or whitespace-only raw output, no normalized-empty
  prediction, and `64/64` unique predictions per language.
- Xho/Zul validation chrF is `22.77167471216128/23.627781327855192`; the
  exact arithmetic mean is `23.199728020008237`. This exactly matches
  `trainer_state.json` `best_metric`, `best_global_step=6164`, and retained
  `checkpoint-6164`, superseding a1 checkpoint 3082 within the same run.
  Artifact SHA-256 is
  `59af1a4e3022fd45d100fe590545f0a5bb9c70c5f6b1f28b623201cf0cae2a82`;
  trainer-state SHA-256 is
  `f0a135e9c85bf7f0086dbd6ed0b750a2b289561d47cc47dad5055345c9469996`.
- All 128 prompts start with `[BOS]`, end with `[EOS]<|assistant|>`, and have
  no EOS after the assistant marker. This is scientifically valid within-run
  validation evidence only: a1 is not terminal-valid and no AfriHG winner is
  selected.
- A0 job `1246938` remains healthy near `14257/15410`, retaining validated
  checkpoint 12328 at mean chrF `22.843278831141802`; its training boundary
  is about one hour away before terminal validation and exact generation.
  A2 job `1246940` remains Resources-pending with estimate `14:10:13`.
  AfriHG is operationally two running plus one pending but scientifically
  `0/3` terminal-valid. HEX quota is home `88.6%`, scratch `39.8%`; all jobs
  are A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

## AfriHG a1 step-6164 callback reaches second language — 09:46 SAST

- A1 job `1246939` remains healthy in its frozen step-6164 exact callback on
  `srvrocgpu010` A100-40GB. Automatic generation batch size 64 was
  established for the second language phase at `09:38:49 SAST`; no targeted
  fault marker appears.
- The step-6164 JSONL remains absent. Prior callback timing keeps the
  complete-artifact ETA around `10:10--10:25 SAST`; a1 remains non-terminal
  and checkpoint 3082 is still its only eligible selector evidence.
- A0 job `1246938` is healthy near `13668/15410`, retaining validated
  checkpoint 12328 at mean chrF `22.843278831141802`; its training boundary
  is about 90 minutes away before terminal validation and exact generation.
  A2 job `1246940` remains Resources-pending with estimate `14:10:13`.
  AfriHG is operationally two running plus one pending but scientifically
  `0/3` terminal-valid. HEX quota is home `88.6%`, scratch `39.8%`; all jobs
  are A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

## AfriHG a1 step-6164 exact callback active — 09:16 SAST

- A1 job `1246939` reached `6164/15410` cleanly on `srvrocgpu010`
  A100-40GB. Full declared validation covered all `3,082` rows at health-only
  loss `2.217037688595068` in `139.9795 s`; the frozen exact callback then
  established automatic generation batch size 64 at `09:11:53 SAST`.
- No step-6164 JSONL or targeted fault marker exists yet. Based on the prior
  exact-callback duration, its complete artifact is tentatively due around
  `10:15--10:30 SAST`. The health-only loss is not selector evidence;
  checkpoint 3082 and mean chrF `22.106737232450644` remain a1's only
  eligible within-run evidence.
- A0 job `1246938` remains healthy near `13092/15410`, retaining validated
  checkpoint 12328 at mean chrF `22.843278831141802`; its training boundary
  is roughly two hours away before terminal validation and exact generation.
  A2 job `1246940` remains Resources-pending with estimate `14:10:13`.
  AfriHG is operationally two running plus one pending but scientifically
  `0/3` terminal-valid. HEX quota is home `88.6%`, scratch `39.8%`; all jobs
  are A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

## AfriHG a0 step-12328 exact checkpoint reconciles — 08:43 SAST

- A0 job `1246938` completed its frozen step-12328 exact callback and resumed
  training on `srvrocgpu010` A100-40GB. The artifact has exactly `128` rows,
  `64` Xho plus `64` Zul, all at global step 12328, with no exact-empty or
  whitespace-only raw output, no normalized-empty prediction, and `64/64`
  unique predictions per language.
- Xho/Zul validation chrF is `22.22350325246101/23.46305440982259`; the exact
  arithmetic mean is `22.843278831141802`. This exactly matches
  `trainer_state.json` `best_metric`, `best_global_step=12328`, and retained
  `checkpoint-12328`, superseding a0 checkpoint 9246 within the same run.
  Artifact SHA-256 is
  `c11f2a435bbab829d038496c34b7ba131ea1a7f0ab33041ed07c6c41cbfeaba3`;
  trainer-state SHA-256 is
  `6ceeefc3bfc763016687c9966467fdbc15c8b28ce3fe32b510b9a6ae2651dbab`.
- All 128 prompts start with `[BOS]`, end with `[EOS]<|assistant|>`, and have
  no EOS after the assistant marker. This is scientifically valid within-run
  validation evidence only: a0 is not terminal-valid and no AfriHG winner is
  selected.
- At `08:43 SAST`, a0 `1246938` and a1 `1246939` remain RUNNING at elapsed
  `15:24:31` and `6:11:19`; a2 `1246940` remains Resources-pending with
  estimate `14:10:13`. AfriHG is operationally two running plus one pending
  but scientifically `0/3` terminal-valid. HEX quota is home `88.6%`, scratch
  `39.8%`; all jobs are A100-40GB, Kombuys is read-only with RTX 5090
  untouched, held-out is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

## AfriHG a0 enters step-12328 exact callback — 07:43 SAST

- A0 job `1246938` reached `12328/15410` cleanly on `srvrocgpu010`
  A100-40GB. Full declared validation covered all `3,082` rows at health-only
  loss `2.2554911044725423` in `139.1445 s`; the frozen exact callback
  established automatic generation batch size 64 at `07:29:03 SAST`.
- No step-12328 JSONL or fault marker exists yet. Observed callback timing
  keeps the complete-artifact ETA around `08:25--08:40 SAST`; checkpoint
  9246 and mean chrF `22.440017494077004` remain the only eligible within-run
  selector evidence, and a0 remains non-terminal.
- A1 job `1246939` is healthy past `4556/15410`, retaining validated
  checkpoint 3082. A2 job `1246940` remains Resources-pending with estimate
  `14:10:13`. AfriHG is operationally two running plus one pending but
  scientifically `0/3` terminal-valid. HEX quota is home `88.6%`, scratch
  `39.8%`; all jobs are A100-40GB, Kombuys is read-only with RTX 5090
  untouched, held-out is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

## AfriHG a0 approaches step 12328 — 07:13 SAST

- A0 job `1246938` remains healthy on `srvrocgpu010` A100-40GB near
  `12105/15410`, with no targeted fault marker. Its frozen step-12328
  boundary is about `223` training steps away; current throughput projects
  the boundary near `07:25 SAST` and the complete exact artifact around
  `08:25--08:40`.
- A1 job `1246939` is healthy near `3991/15410`, retaining validated
  checkpoint 3082 at mean chrF `22.106737232450644`. A2 job `1246940`
  remains Resources-pending with estimate `14:10:13`.
- AfriHG is operationally two running plus one pending but scientifically
  `0/3` terminal-valid. HEX quota is home `88.6%`, scratch `39.8%`; all jobs
  are A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

## AfriHG a1 first exact checkpoint reconciles — 06:43 SAST

- A1 job `1246939` completed its frozen step-3082 exact callback at
  `06:25:21 SAST` and resumed training past `3420/15410` on `srvrocgpu010`
  A100-40GB. The artifact has exactly `128` rows, `64` Xho plus `64` Zul,
  all at global step 3082, with no exact-empty or whitespace-only raw output,
  no normalized-empty prediction, and `64/64` unique predictions per
  language.
- Xho/Zul validation chrF is
  `21.14544404556832/23.06803041933297`; the exact arithmetic mean is
  `22.106737232450644`. This exactly matches `trainer_state.json`
  `best_metric`, `best_global_step=3082`, and retained `checkpoint-3082`.
  Artifact SHA-256 is
  `05d2969eddf94ad424067a4806d48ddae2e804ab4058b96ff95fd349fc82c0c9`;
  trainer-state SHA-256 is
  `086ef14fa6fcb1f4f5e84b487a98279152059f026a4ca346f82be5cda0956adb`.
- All 128 prompts start with `[BOS]`, end with `[EOS]<|assistant|>`, and have
  no terminal EOS after the assistant marker. This is scientifically valid
  within-run validation evidence only: a1 is not terminal-valid and no
  AfriHG winner is selected.
- A0 job `1246938` remains healthy past `11536/15410`, retaining validated
  checkpoint 9246 at mean chrF `22.440017494077004`. A2 job `1246940`
  remains Resources-pending with estimate `14:10:13`. AfriHG is
  operationally two running plus one pending but scientifically `0/3`
  terminal-valid. Quota is home `88.6%`, scratch `39.8%`; all jobs are
  A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is `0`,
  Sheet E/F/G are blank, and General/Monolingual/publication remain blocked.

## AfriHG a1 exact callback reaches second language — 06:13 SAST

- A1 job `1246939` remains healthy in its frozen step-3082 exact callback on
  `srvrocgpu010` A100-40GB. Automatic batch size 64 was established for the
  second language phase at `05:49:17 SAST`; no targeted fault marker appears.
- The step-3082 JSONL remains absent. Observed callback timing revises the
  complete-artifact ETA slightly to `06:20--06:35 SAST`; a1 remains
  non-terminal and has no eligible selector metric yet.
- A0 job `1246938` is healthy past `10964/15410`, retaining validated
  checkpoint 9246 at mean chrF `22.440017494077004`. A2 job `1246940`
  remains Resources-pending with estimate `14:10:13`. AfriHG is
  operationally two running plus one pending but scientifically `0/3`
  terminal-valid. HEX quota is home `88.6%`, scratch `39.8%`; all jobs are
  A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is `0`,
  Sheet E/F/G are blank, and General/Monolingual/publication remain blocked.

## AfriHG a1 first exact callback active — 05:43 SAST

- A1 job `1246939` completed full declared validation over all `3,082` rows
  at health-only loss `2.26994530679652` in `139.6339 s`, then entered its
  frozen step-3082 exact callback. Automatic generation batch size 64 was
  established at `05:18:01 SAST`; no fault marker appears.
- The step-3082 exact JSONL is still absent. Observed callback timing keeps
  the complete-artifact ETA around `06:15--06:30 SAST`; the health-only loss
  is not selector evidence and a1 is not terminal-valid.
- A0 job `1246938` remains healthy past `10395/15410`, retaining validated
  checkpoint 9246 at mean chrF `22.440017494077004`. A2 job `1246940`
  remains Resources-pending with estimate `14:10:13`. AfriHG is
  operationally two running plus one pending but scientifically `0/3`
  terminal-valid. HEX quota is home `88.6%`, scratch `39.8%`; all jobs are
  A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is `0`,
  Sheet E/F/G are blank, and General/Monolingual/publication remain blocked.

## AfriHG a1 reaches first validation boundary — 05:13 SAST

- A1 job `1246939` reached `3082/15410` cleanly around `05:11 SAST` on
  `srvrocgpu010` A100-40GB and entered its first full declared validation.
  No step-3082 exact JSONL or fault marker exists yet; allowing the health
  pass and frozen 128-prompt callback gives a tentative complete-artifact ETA
  around `06:15--06:30 SAST`.
- A0 job `1246938` remains healthy past `9821/15410`, retaining validated
  checkpoint 9246 at mean chrF `22.440017494077004`. A2 job `1246940`
  remains Resources-pending with estimate `14:10:13`.
- AfriHG is operationally two running plus one pending but scientifically
  `0/3` terminal-valid. HEX quota is home `88.6%`, scratch `39.8%`; all jobs
  are A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is
  `0`, Sheet E/F/G are blank, and General/Monolingual/publication remain
  blocked.

## AfriHG a0 improves at step 9246 — 04:43 SAST

- A0 job `1246938` completed its frozen step-9246 exact callback at
  `04:43:20 SAST` and resumed training past `9266/15410` on `srvrocgpu010`
  A100-40GB. The artifact has exactly `128` rows, `64` Xho plus `64` Zul,
  all at global step 9246, with no exact-empty or whitespace-only raw output,
  no normalized-empty prediction, and `64/64` unique predictions per
  language.
- Xho/Zul validation chrF is
  `22.038503951986627/22.841531036167382`; the exact arithmetic mean is
  `22.440017494077004`. This exactly matches `trainer_state.json`
  `best_metric`, `best_global_step=9246`, and retained `checkpoint-9246`.
  Artifact SHA-256 is
  `39ab1091aec93f691b653e74e70b1bb70edb4bd8d8e7bb5fc539fa0a5984bc55`;
  trainer-state SHA-256 is
  `c2128f4a7545b33b8acd7c01ae292525bd3aedccd150c89fc58e11daabd6abc8`.
- All 128 prompts start with `[BOS]`, end with `[EOS]<|assistant|>`, and have
  no terminal EOS after the assistant marker. This is scientifically valid
  within-run validation evidence only: a0 is not terminal-valid and no
  AfriHG winner is selected. An independent Luna audit agrees on every count,
  hash, prompt-contract invariant, metric, and retained-checkpoint field, with
  no discrepancy and no held-out access.
- A1 job `1246939` remains healthy near `2505/15410`; a2 job `1246940`
  remains Resources-pending with estimate `14:10:13`. AfriHG is
  operationally two running plus one pending but scientifically `0/3`
  terminal-valid. Quota is home `88.6%`, scratch `39.8%`; all jobs are
  A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is `0`,
  Sheet E/F/G are blank, and General/Monolingual/publication remain blocked.

## AfriHG a0 exact callback reaches second language — 04:13 SAST

- A0 job `1246938` remains healthy in its frozen step-9246 exact callback on
  `srvrocgpu010` A100-40GB. Automatic batch size 64 was established for the
  second language phase at `04:07:04 SAST`; no targeted fault marker appears.
- The step-9246 JSONL is still absent. Observed callback timing keeps the
  complete-artifact ETA around `04:35--04:50 SAST`; checkpoint 6164 and mean
  chrF `21.242485538593336` remain the only eligible within-run selector
  evidence, and a0 remains non-terminal.
- A1 job `1246939` is healthy near `1927/15410`; its first step-3082 boundary
  remains near `05:13 SAST`. A2 job `1246940` is Resources-pending with
  estimate `14:10:13`. AfriHG is operationally two running plus one pending
  but scientifically `0/3` terminal-valid. Quota is home `88.6%`, scratch
  `39.8%`; all jobs are A100-40GB, Kombuys is read-only with RTX 5090
  untouched, held-out is `0`, Sheet E/F/G are blank, and
  General/Monolingual/publication remain blocked.

## AfriHG a0 enters step-9246 exact callback — 03:43 SAST

- A0 job `1246938` reached `9246/15410` cleanly on `srvrocgpu010`
  A100-40GB. Full declared validation covered all `3,082` rows at health-only
  loss `2.2620294312236373` in `140.9852 s`; the frozen exact callback began
  and established automatic generation batch size 64 at `03:38:52 SAST`.
- No step-9246 JSONL exists yet. The complete exact artifact remains
  tentatively due around `04:35--04:50 SAST`; until then checkpoint 6164 and
  mean chrF `21.242485538593336` remain the only eligible within-run selector
  evidence. A0 is not terminal-valid.
- A1 job `1246939` remains healthy near `1349/15410`; a2 job `1246940`
  remains Resources-pending with estimate `14:10:13 SAST`. AfriHG is
  operationally two running plus one pending but scientifically `0/3`
  terminal-valid. HEX quota is home `88.6%`, scratch `39.8%`; all jobs are
  A100-40GB, Kombuys is read-only with RTX 5090 untouched, held-out is `0`,
  Sheet E/F/G are blank, and General/Monolingual/publication remain blocked.

## AfriHG a0 approaches step 9246; a1 steady — 03:13 SAST

- A0 job `1246938` remains healthy on `srvrocgpu010` A100-40GB at about
  `8850/15410`, with no targeted fault marker. Its frozen step-9246 boundary
  is about `396` training steps away; recent throughput projects the boundary
  near `03:34 SAST` and the complete exact artifact around `04:35--04:50`.
  Retained checkpoint 6164 and mean chrF `21.242485538593336` remain the only
  eligible within-run selector evidence.
- A1 job `1246939` is healthy near `778/15410` after `00:42:30`; its first
  step-3082 boundary is tentatively near `05:13 SAST`. A2 job `1246940`
  remains Resources-pending with Slurm estimate `14:10:13 SAST`. AfriHG is
  operationally two running plus one pending but scientifically remains
  `0/3` terminal-valid.
- HEX quota is home `88.6%`, scratch `39.8%`. All three jobs request
  A100-40GB `gpu:ampere`; no A100-80GB/L40S work overlaps. Kombuys remains
  read-only with RTX 5090 untouched, held-out remains `0`, Sheet E/F/G remain
  blank, and corrected General, Monolingual, and publication remain blocked.

## AfriHG a1 starts; a0 remains healthy — 02:43 SAST

- AfriHG Stage-A a1 job `1246939` started at `02:31:20 SAST` on a second
  `srvrocgpu010` A100-40GB. It wrote execution-manifest SHA-256
  `32e43a2bd448aadf34002bcc9078768a0db618278439b960561e3c3e3382fd72`,
  verified all `694` source/config files, passed the GatedDeltaNet fast-path
  gate, loaded exactly `24,649/3,082` train/validation rows, and reached about
  `202/15410` near `3.13 s/step` without a targeted fault marker.
- A0 job `1246938` remains healthy near `8267/15410`, retaining validated
  checkpoint 6164. A2 job `1246940` is Resources-pending, with a new
  scheduler estimate of `14:10:13 SAST`. AfriHG is operationally two running
  plus one pending but scientifically remains `0/3` terminal-valid.
- All three jobs request A100-40GB `gpu:ampere`; no A100-80GB/L40S work
  overlaps. HEX quota is home `88.6%`, scratch `39.8%`; Kombuys remains
  read-only with RTX 5090 untouched, held-out is `0`, Sheet E/F/G are blank,
  and corrected General, Monolingual, and publication remain blocked.

## AfriHG a0 improves at step 6164 — 01:13 SAST

- AfriHG Stage-A a0 job `1246938` completed its second frozen exact callback
  at `00:53:56 SAST` and resumed healthy training near `6545/15410` on
  `srvrocgpu010` A100-40GB. The 128-row artifact covers exactly `64` Xho and
  `64` Zul, with zero empty or whitespace-only raw outputs, zero normalized-
  empty predictions, and `64/64` unique predictions per language.
- Xho/Zul validation chrF is
  `20.742026697780208/21.742944379406463`; the exact arithmetic mean is
  `21.242485538593336`. This exactly matches `trainer_state.json`
  `best_metric`, and retained `best_model_checkpoint` is `checkpoint-6164`.
  Artifact SHA-256 is
  `5fc6747ca7cd80f35a51fa22ae996bf63214e75fe04386cd2b4d432fe21a1e9f`.
  Independent Sol and Luna audits agree on every count and metric.
- This is scientifically valid within-run validation evidence only: a0 is
  not terminal-valid and no AfriHG winner is selected. Jobs
  `1246939/1246940` remain Resources/Priority-pending; a1 is projected at
  `03:18:32 SAST` and a2 has no estimate. Quota remains home `88.6%`, scratch
  `39.6%`; all jobs request A100-40GB `gpu:ampere`, Kombuys is read-only with
  RTX 5090 untouched, held-out is `0`, Sheet E/F/G are blank, and corrected
  General, Monolingual, and publication remain blocked.

## AfriHG a0 enters second exact callback — 00:13 SAST

- AfriHG Stage-A a0 job `1246938` reached step `6164/15410` cleanly on
  `srvrocgpu010` A100-40GB. Full declared validation covered all `3,082`
  rows at health-only loss `2.2817668778644142`, with exact Xho/Zul coverage
  `1305/1777`; runtime was `140.4197 s`.
- The frozen 128-prompt exact callback is active. Automatic generation batch
  size 64 was established at `23:50:15 SAST`; no step-6164 artifact or fault
  marker exists yet. Based on the first callback, the exact artifact remains
  tentatively due around `00:50--01:10 SAST`. The retained step-3082 chrF
  artifact remains the only scientific selector evidence, and a0 is not
  terminal-valid.
- Jobs `1246939/1246940` remain Resources/Priority-pending; Slurm projects a1
  at `03:18:32 SAST` and gives a2 no estimate. All three request A100-40GB
  `gpu:ampere`; quota is home `88.6%`, scratch `39.6%`. Kombuys remains
  read-only with RTX 5090 untouched, held-out remains `0`, Sheet E/F/G remain
  blank, and corrected General, Monolingual, and publication remain blocked.
