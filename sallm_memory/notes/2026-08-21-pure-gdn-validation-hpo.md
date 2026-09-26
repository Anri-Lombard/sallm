# Pure-GDN validation HPO — 2026-08-21

## AfriHG b3 reaches second validation boundary — 23:49 SAST

- B3 job `1250970` reached exactly step `6164/15410`, completed all `3,082`
  declared validation rows at health-only loss `2.2234693127433487` in
  `138.1312 s`, and entered its frozen exact-generation path on
  `srvrocgpu010` A100-40GB. No step-6164 exact artifact exists yet, so the
  loss is not selector evidence and checkpoint 3082 remains the audited
  within-run best. Based on prior callback duration, the artifact is
  tentatively expected around `00:55--01:10 SAST` on 22 August.
- B2 `1248577` remains fault-free beyond step `10844`, retaining checkpoint
  9246 at validation-only mean chrF `22.230781529182863`. Exact tensor
  verifier retry `1252840` remains Resources-pending with scheduler estimate
  `05:38:36 SAST` on 22 August.
- Owned work remains exactly three A100-40GB jobs, with no A100-80GB or L40S
  work. Quota is home `88.6%`, scratch `41.2%`. AfriHG remains `3/11`
  scientifically terminal-valid and the global freeze remains `0/8`.
  Kombuys remains read-only with RTX 5090 untouched, held-out access is `0`,
  Sheet E/F/G remain blank, and General/Monolingual/publication remain
  blocked.

## AfriHG b2 improves at checkpoint 9246; verifier retry queued — 22:45 SAST

- B2 job `1248577` completed its frozen step-9246 callback and resumed
  fault-free training beyond step `9736` on `srvrocgpu010` A100-40GB. Its
  exact artifact has `128` rows (`64/64` Xho/Zul), zero empty predictions,
  `64/64` unique predictions per language, and clean `[BOS]` and
  `[EOS]<|assistant|>` prompt boundaries.
- Xho/Zul chrF is `21.729798398243634/22.73176466012209`; registered mean
  chrF is `22.230781529182863`. This exactly matches `trainer_state.json`,
  improves b2's prior best, and moves its validation-only retained checkpoint
  to 9246. Artifact and trainer-state SHA-256 values are
  `14044d771f1955b19a6872e63a4d661ce483581acaffbf87d51c82edf7c919c6`
  and `c5a92ec3970c2bc5643afaaa01da6a4f6726093d08ebd09c3037e8a7dd94a059`.
  This is within-run selector evidence, not terminal-valid evidence.
- Tensor verifier job `1252793` failed immediately `1:0` before reading any
  adapter because shell quoting stripped Python string literals and caused a
  `SyntaxError`. Its failure output SHA-256 is
  `69d68b830a11a7f7e88600487768838a2ffb4e382f02e4b2151eddf0fadbf363`.
  The failure is operational only and produced no scientific result.
- A reproducible standalone retry script was stored at
  `sallm_memory/artifacts/2026-08-21/verify_afrihg_b0_b1.sbatch`, passed
  `bash -n`, and has SHA-256
  `b10ae7326bc1e52a1888689388678a5a571ca1b84aab57fd8a428cefc9a9e056`.
  It records input hashes and requires exact key, shape, dtype, value, and
  tensor-count equality. Retry job `1252840` was submitted once under the
  required envelope and is Resources-pending with scheduler estimate
  `2026-08-22 05:38:36 SAST`.
- B3 `1250970` remains fault-free beyond step `4987`. Owned work is exactly
  three A100-40GB jobs: b2 and b3 running, verifier retry pending. Quota is
  home `88.6%`, scratch `41.2%`. AfriHG remains `3/11` scientifically
  terminal-valid and the global freeze remains `0/8` until tensor equivalence
  and remaining terminal trials pass. Kombuys remains read-only with RTX 5090
  untouched, held-out access is `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.

## AfriHG b1 completes; b0/b1 tensor verifier queued — 21:47 SAST

- B1 job `1248576` completed `0:0` at `21:36:26 SAST` after `19:02:10`.
  Its terminal step-15410 artifact has exactly `128` rows (`64/64` Xho/Zul),
  zero empty predictions, `64/63` unique predictions, and clean `[BOS]` and
  `[EOS]<|assistant|>` prompt boundaries. Terminal Xho/Zul chrF is
  `22.999692656448786/23.8565376695875`, mean `23.42811516301814`; this does
  not improve the validation-only retained checkpoint 12328 at mean
  `23.50555969452345`.
- Terminal artifact, retained trainer-state, retained BIN, final safetensors,
  trial-record, and execution-manifest SHA-256 values are respectively
  `f7c2465d1ca50ff9092476b013ce4fe16b47205c578733e2c39c3296b175bb20`,
  `1807ebff82c0488b2a77f87e2b575316b55be386875c9bf7ae359aae682ac129`,
  `32c3378f8a1d7418e5f277e97274fdc82121f8e97d50c5074d84ae0a87ce36ef`,
  `0228762dff097e00bc7654cb9406bfb6d0a47ee3245eb281cb519132cbd3cb16`,
  `7a46f0b6086de759d446f276447a5029a8a39567536a2d86121bf0096145ec9d`,
  and `81361da656e5a92711d073964b082890116049a2d457f10728b8fa820c5939a0`.
  Retained/final adapter configs are byte-identical; the final config hash is
  `7fd6ce0fc8f73ce5062c19f88e1b0ea4a5efc3d513f0973d7164842e026c42d9`.
- The freed third slot was used once for read-only compute verifier job
  `1252793` (`verify-afrihg-b0-b1`), which will compare every retained BIN
  tensor with each final safetensors artifact using exact key equality and
  `torch.equal`. It is Priority-pending with scheduler estimate
  `2026-08-22 11:51 SAST`. Until it passes, b0 and b1 are operationally
  complete but not scientifically terminal-valid; AfriHG therefore remains
  `3/11` and the global freeze remains `0/8`.
- B2 `1248577` remains healthy in its step-9246 exact callback; no artifact
  exists yet and it is tentatively expected around `22:15--22:30 SAST`. B3
  `1250970` resumed healthy training beyond step 3900. Owned work is exactly
  three jobs, all A100-40GB (`gpu:ampere`): b2 and b3 running, verifier
  pending. Quota is home `88.6%`, scratch `41.2%`; Kombuys remains read-only
  with RTX 5090 untouched, held-out access is `0`, Sheet E/F/G remain blank,
  and General/Monolingual/publication remain blocked.

## AfriHG b3 first exact artifact; b2 reaches third boundary — 21:14 SAST

- B3 job `1250970` completed its frozen step-3082 callback and resumed
  fault-free training beyond step `3245` on `srvrocgpu010` A100-40GB. Its
  exact artifact has `128` rows (`64/64` Xho/Zul), zero empty predictions,
  `64/64` unique predictions per language, and all prompts begin `[BOS]` and
  end `[EOS]<|assistant|>`.
- Xho/Zul chrF is `21.110816397043155/22.68749899428495`; registered mean
  chrF is `21.89915769566405`, exactly matching `trainer_state.json` and
  retaining checkpoint 3082. Artifact and trainer-state SHA-256 values are
  `d326ba97a910df8d9a36c6badccf6393dacee72b4f324d1530f1c9bf984bbbfa`
  and `99b92b87235c9a1af4c056b37d3e261f4a01076be4fefb1ffbada5625bbd74d9`.
  This is within-run selector evidence, not terminal-valid evidence.
- B2 `1248577` reached step `9246/15410` and completed all `3,082` declared
  validation rows at health-only loss `2.2812004114110156`; its frozen exact
  callback has not produced an artifact yet. B1 `1248576` remains in terminal
  exact generation with both language segments active. All three owned jobs
  are running on A100-40GB with zero targeted faults; quota is
  `88.6%/41.1%`. Scientific progress stays AfriHG `3/11` and global freeze
  `0/8` because b0 tensor equivalence and terminal trials remain pending.
  Kombuys remains read-only with RTX 5090 untouched, held-out access is `0`,
  and Sheet E/F/G remain blank.

## AfriHG b1 reaches terminal validation boundary — 20:44 SAST

- B1 job `1248576` reached exactly step `15410/15410`, completed all `3,082`
  declared validation rows at health-only loss `2.2297809261690866` in
  `139.9348 s`, and entered its frozen terminal exact generation callback on
  `srvrocgpu010` A100-40GB. Automatic generation batch size 64 was
  established for segment 1 at `20:32:21 SAST`.
- No terminal exact artifact or final reconciliation exists yet, so the loss
  is not selector evidence and b1 remains non-terminal-valid. Based on prior
  callbacks, terminal output is tentatively expected around
  `21:30--21:50 SAST`; checkpoint 12328 remains the audited within-run best.
- B3 `1250970` remains in its step-3082 exact callback with both language
  segments active; b2 `1248577` is fault-free near step `8723/15410`. All
  three owned jobs are running on A100-40GB; quota is `88.6%/41.1%`.
  Scientific progress stays AfriHG `3/11` and global freeze `0/8` because b0
  tensor equivalence and terminal trials remain pending. Kombuys remains
  read-only with RTX 5090 untouched, held-out access is `0`, and Sheet E/F/G
  remain blank.

## AfriHG b3 reaches first validation boundary — 20:14 SAST

- B3 job `1250970` reached exactly step `3082/15410`, completed all `3,082`
  declared validation rows at health-only loss `2.2798393393088903` in
  `138.9868 s`, and entered its frozen exact generation callback on
  `srvrocgpu010` A100-40GB. Automatic generation batch size 64 was
  established for segment 1 at `19:57:50 SAST`.
- No step-3082 exact artifact exists yet, so the loss is not selector evidence
  and there is no scientific-progress change. Based on prior callbacks, the
  exact artifact is tentatively expected around `20:55--21:10 SAST`.
- B1 `1248576` is fault-free at step `15123/15410`, about 15 minutes from its
  terminal training boundary; b2 `1248577` is fault-free near step
  `8149/15410`. All three owned jobs are running on A100-40GB; quota is
  `88.6%/41.1%`. Scientific progress stays AfriHG `3/11` and global freeze
  `0/8` because b0 tensor equivalence and terminal trials remain pending.
  Kombuys remains read-only with RTX 5090 untouched, held-out access is `0`,
  and Sheet E/F/G remain blank.

## AfriHG b2 step-6164 exact selector artifact — 18:44 SAST

- B2 job `1248577` completed its frozen step-6164 callback and resumed
  fault-free training beyond step `6455` on `srvrocgpu010` A100-40GB. Its
  exact artifact has `128` rows (`64/64` Xho/Zul), zero empty predictions,
  `64/64` unique predictions per language, and all prompts begin `[BOS]` and
  end `[EOS]<|assistant|>`.
- Xho/Zul chrF is `20.71843131877858/21.26318054903395`; mean chrF is
  `20.990805933906266`. This does not improve the registered validation-only
  best `21.120202579737125`, so b2 retains checkpoint 3082. Artifact and
  trainer-state SHA-256 values are
  `f0339961bc79557317bc9d272e4fb15092590f57e4ec6aee5c8db81ed31f10d9`
  and `047ddfda82e48aac5173a219b0fabee8bb1026b7873898bb6031e75636087de3`.
  This is within-run selector evidence, not terminal-valid evidence.
- B1 `1248576` is fault-free near step `13424/15410`, retaining checkpoint
  12328 at mean chrF `23.50555969452345`; b3 `1250970` is fault-free near
  step `1694/15410`. All three owned jobs are running on A100-40GB; quota is
  `88.6%/41.1%`. Scientific progress stays AfriHG `3/11` and global freeze
  `0/8` because b0 tensor equivalence and terminal trials remain pending.
  Kombuys remains read-only with RTX 5090 untouched, held-out access is `0`,
  and Sheet E/F/G remain blank.

## AfriHG b1 improves at checkpoint 12328 — 18:14 SAST

- B1 job `1248576` completed its frozen step-12328 callback and resumed
  fault-free training beyond step `12853` on `srvrocgpu010` A100-40GB. Its
  exact artifact has `128` rows (`64/64` Xho/Zul), zero empty predictions,
  `64/63` unique predictions per language, and all prompts begin `[BOS]` and
  end `[EOS]<|assistant|>`.
- Xho/Zul chrF is `22.873427986122337/24.137691402924556`; registered mean
  chrF is `23.50555969452345`. This exactly matches `trainer_state.json`,
  improves b1's step-9246 score, and moves its retained validation-only
  checkpoint to 12328. Artifact and trainer-state SHA-256 values are
  `cd151e6d409410635d42ea5df4ac47b82cf657cbd5decb4bdaa0d1331e25d008`
  and `1807ebff82c0488b2a77f87e2b575316b55be386875c9bf7ae359aae682ac129`.
  This is within-run selector evidence, not terminal-valid evidence.
- B2 `1248577` remains in its step-6164 exact callback with both language
  segments active; b3 `1250970` is fault-free near step `1098/15410`. All
  three owned jobs are running on A100-40GB; quota is `88.6%/41.1%`.
  Scientific progress stays AfriHG `3/11` and global freeze `0/8` because b0
  tensor equivalence and terminal trials remain pending. Kombuys remains
  read-only with RTX 5090 untouched, held-out access is `0`, and Sheet E/F/G
  remain blank.

## AfriHG b3 starts; b2 reaches second validation boundary — 17:45 SAST

- B3 job `1250970` started at `17:15:39 SAST` on `srvrocgpu010` with one
  `NVIDIA A100-PCIE-40GB`. It verified all `694` source/config files from its
  immutable execution manifest (SHA-256
  `71a5e984da15fd42c49ef6f0acfb013155f65e902ff23e3eeb657867f36b78e3`),
  loaded the canonical pure `GatedDeltaNetForCausalLM` with the CUDA fast
  path, and declared complete `24,649/3,082` train/validation coverage. It was
  fault-free near step `525/15410`.
- B2 job `1248577` reached exactly step `6164/15410`, completed all `3,082`
  validation rows at health-only loss `2.3051664338801916`, and entered its
  frozen exact callback with automatic generation batch size 64 at
  `17:25:47 SAST`. No step-6164 exact artifact exists yet; it is tentatively
  expected around `18:25--18:40 SAST`.
- B1 `1248576` remains in its step-12328 exact callback; both language
  segments are active but the exact artifact is still absent. All three owned
  jobs are now running on A100-40GB with zero targeted faults. Quota is
  `88.6%/41.1%`. Scientific progress stays AfriHG `3/11` and global freeze
  `0/8` because b0 tensor equivalence and terminal trials remain pending.
  Kombuys remains read-only with RTX 5090 untouched, held-out access is `0`,
  and Sheet E/F/G remain blank.

## AfriHG b1 reaches fourth validation boundary — 17:02 SAST

- B1 job `1248576` reached exactly step `12328/15410` and completed full
  declared validation over all `3,082` rows at health-only loss
  `2.2270502267450114` in `140.5401 s`. It entered the frozen exact
  generation callback on `srvrocgpu010` A100-40GB; automatic batch size 64
  was established for segment 1 at `16:39:57 SAST`.
- No step-12328 exact artifact exists yet, so the loss is not selector
  evidence and retained checkpoint 9246 remains the audited within-run best.
  Based on prior callbacks, the exact artifact is tentatively expected around
  `17:35--17:50 SAST`. B2 `1248577` remains fault-free near step
  `5774/15410`; b3 `1250970` remains Resources-pending with provisional start
  `2026-08-22 02:34:16 SAST`.
- Owned work remains at the three-job A100-40GB cap; quota is
  `88.6%/40.9%`. Scientific progress stays AfriHG `3/11` and global freeze
  `0/8` because b0 tensor equivalence is pending. Kombuys remains read-only
  with RTX 5090 untouched, held-out access is `0`, and Sheet E/F/G remain
  blank.

## AfriHG b2 first exact selector artifact — 15:02 SAST

- B2 job `1248577` completed its frozen step-3082 callback and resumed
  fault-free training beyond step 3464 on `srvrocgpu010` A100-40GB. Its exact
  artifact has `128` rows (`64/64` Xho/Zul), zero empty predictions, `64/64`
  unique predictions per language, and all prompts begin `[BOS]` and end
  `[EOS]<|assistant|>`.
- Xho/Zul chrF is `21.24350617202978/20.996898987444474`; the registered mean
  chrF is `21.120202579737125`, exactly matching `trainer_state.json` and
  retaining checkpoint 3082 using validation-only evidence. Artifact and
  trainer-state SHA-256 values are
  `1f43c032f88bb1bb46b4ce8df0e5d5abdd5014ce9c13ec1234578151df52d15b`
  and `4f6ce1239e86c4f7138e52b0fd0ddc54841e59b10758020e8081110c2849ba8d`.
  This is within-run selector evidence, not terminal-valid evidence.
- B1 `1248576` remains fault-free near step `10505/15410`, retaining
  checkpoint 9246 at mean chrF `23.427158203796658`. B3 `1250970` is
  Resources-pending with provisional start `2026-08-22 02:34:16 SAST`.
  Owned work remains at the three-job A100-40GB cap; quota is
  `88.6%/40.9%`. Scientific progress stays AfriHG `3/11` and global freeze
  `0/8` because b0 tensor equivalence is pending. Kombuys remains read-only
  with RTX 5090 untouched, held-out access is `0`, and Sheet E/F/G remain
  blank.

## AfriHG b1 improves at checkpoint 9246; b2 enters first callback — 14:02 SAST

- B1 job `1248576` completed its frozen step-9246 callback and resumed
  fault-free training beyond step 9350 on `srvrocgpu010` A100-40GB. Its exact
  artifact has `128` rows (`64/64` Xho/Zul), zero empty predictions, `64/64`
  unique predictions per language, and all prompts begin `[BOS]` and end
  `[EOS]<|assistant|>`.
- Xho/Zul chrF is `23.160436060121018/23.693880347472295`, giving mean chrF
  `23.427158203796658`. This exactly matches `trainer_state.json`, improves
  b1's step-6164 score, and moves its retained validation-only checkpoint to
  9246. Artifact and trainer-state SHA-256 values are
  `7762ee77ec8f1d5d9befd58a7001f2b3590eb6bbcd5023a457dbb3073fbfeef8`
  and `7fa3ac45f71768986f8f3db90ee3cd16b8f0dc3c855e32c428bcb11ee0b81f42`.
  This is within-run selector evidence, not terminal-valid evidence.
- B2 `1248577` reached step `3082/15410`, completed all `3,082` validation
  rows at health-only loss `2.3783920654455306`, and entered its frozen exact
  callback with automatic generation batch size 64. No b2 exact artifact
  exists yet. B3 `1250970` remains Priority-pending. Owned work remains at the
  three-job A100-40GB cap; quota is `88.6%/40.9%`. Scientific progress stays
  AfriHG `3/11` and global freeze `0/8` because b0 tensor equivalence is
  pending. Kombuys remains read-only with RTX 5090 untouched, held-out access
  is `0`, and Sheet E/F/G remain blank.

## AfriHG b1 reaches third validation boundary — 13:02 SAST

- B1 job `1248576` reached exactly step `9246/15410`, completed all `3,082`
  declared validation rows at health-only loss `2.2297211188453114` in
  `138.1834 s`, and entered its frozen exact generation callback on
  `srvrocgpu010` A100-40GB. Automatic generation batch size 64 was established
  for segment 1 at `12:53:08 SAST`.
- No step-9246 exact artifact exists yet, so the health-only loss is not
  selector evidence and checkpoint 6164 remains the audited within-run best.
  The job is fault-free; based on its prior callbacks, the exact artifact is
  tentatively expected around `14:05--14:20 SAST`.
- B2 `1248577` is fault-free near step `2487/15410`; b3 `1250970` remains
  Priority-pending. Owned work remains at the three-job cap, entirely
  A100-40GB. Scientific progress stays AfriHG `3/11` and global freeze `0/8`
  because b0 tensor equivalence is pending. Quota is `88.6%/40.9%`; Kombuys
  remains read-only with RTX 5090 untouched, held-out access is `0`, and
  Sheet E/F/G remain blank.

## AfriHG b2 starts — 11:02 SAST

- B2 job `1248577` started at `10:51:01 SAST` on `srvrocgpu010` with one
  `NVIDIA A100-PCIE-40GB`. Its execution manifest verified all `694`
  source/config files, the pure-GDN CUDA fast path is available,
  `CUDA_VISIBLE_DEVICES=0`, and it loaded the canonical
  `GatedDeltaNetForCausalLM`. Coverage is `24,649/3,082` train/validation
  rows; the targeted log check found no traceback, CUDA error, OOM, NCCL
  error, or exception.
- B1 `1248576` remains healthy at step `7189/15410`, retaining checkpoint
  6164 at validation-only mean chrF `22.030107111540246`. B3 `1250970`
  remains Priority-pending without a scheduler estimate. Owned work is at the
  three-job cap: two running and one pending, all A100-40GB.
- Scientific progress is unchanged: AfriHG remains `3/11`, global freeze
  remains `0/8`, and b0 retained/final tensor equivalence is pending. Quota is
  `88.6%/40.9%`; Kombuys remains read-only with RTX 5090 untouched, held-out
  access is `0`, and Sheet E/F/G remain blank.

## AfriHG b1 improves at checkpoint 6164 — 10:32 SAST

- B1 job `1248576` completed its frozen step-6164 callback and resumed healthy
  training beyond step 6608 on `srvrocgpu010` A100-40GB. The exact artifact
  contains `128` rows (`64/64` Xho/Zul), zero empty predictions, `64/64`
  unique predictions per language, and clean BOS/assistant prompt boundaries.
- Xho/Zul chrF is `21.641679914561536/22.418534308518957`, giving mean chrF
  `22.030107111540246`. This exactly matches `trainer_state.json`, improves
  b1's step-3082 score, and moves its retained validation-only checkpoint to
  6164. Artifact SHA-256 is
  `0de4e760a86cbbf12805d4fb50f5b70c5c1f2b8f79a692f58c9417cb9b97aac3`.
  This is within-run selector evidence, not terminal-valid evidence.
- B2 `1248577` remains Resources-pending with scheduler estimate
  `2026-08-22 02:34:16`; b3 `1250970` remains Priority-pending without an
  estimate. AfriHG remains `3/11`, global freeze remains `0/8`, and b0 tensor
  equivalence is pending. Quota is `88.6%/40.7%`; all owned jobs are
  A100-40GB, Kombuys remains read-only with RTX 5090 untouched, held-out
  access is `0`, and Sheet E/F/G remain blank.

## AfriHG b1 reaches second validation boundary — 09:02 SAST

- B1 job `1248576` reached exactly step `6164/15410`, completed full declared
  validation over all `3,082` rows at health-only loss
  `2.2457383123022794` in `138.0391 s`, and entered its frozen exact
  generation callback on `srvrocgpu010` A100-40GB. No step-6164 exact
  artifact exists yet and the health-only loss is not selector evidence.
- The job has no targeted fault; based on the first callback, the exact
  artifact is tentatively expected around `10:05--10:20 SAST`. B2 `1248577`
  remains Resources-pending with scheduler estimate `2026-08-22 02:34:16`;
  b3 `1250970` remains Priority-pending without an estimate.
- Scientific progress is unchanged: AfriHG remains `3/11`, global freeze
  remains `0/8`, and b0 retained/final tensor equivalence is pending. Quota is
  `88.6%/40.7%`; all owned jobs are A100-40GB, Kombuys remains read-only with
  RTX 5090 untouched, held-out access is `0`, and Sheet E/F/G remain blank.

## AfriHG b1 first exact selector artifact — 06:25 SAST

- B1 job `1248576` completed its frozen step-3082 callback and resumed healthy
  training on `srvrocgpu010` A100-40GB. The exact artifact contains `128`
  rows (`64/64` Xho/Zul), zero empty predictions, `64/64` unique predictions
  per language, and clean BOS/assistant prompt boundaries.
- Xho/Zul chrF is `21.196464787124476/22.282564090083`, giving registered
  mean chrF `21.739514438603738`; this exactly matches `trainer_state.json`
  and retains checkpoint 3082 using validation-only evidence. Artifact
  SHA-256 is
  `96642a367fbe5cf9bd5c6131e075e5e3100d170ebb7fe3e7558a640bf0ad4a39`.
  This is within-run selector evidence, not a terminal-valid candidate.
- B1 was training beyond step 3176 at the live check. B2 `1248577` remains
  Resources-pending with scheduler estimate `2026-08-22 02:34:16`; b3
  `1250970` remains Priority-pending without an estimate. AfriHG remains
  `3/11` scientifically terminal-valid, global freeze remains `0/8`, and b0
  retained/final tensor equivalence is still pending. Quota is `88.6%/40.7%`;
  all owned jobs are A100-40GB, Kombuys remains read-only with RTX 5090
  untouched, held-out access remains `0`, and Sheet E/F/G remain blank.

## AfriHG b1 reaches first validation boundary — 05:24 SAST

- B1 job `1248576` reached exactly step `3082/15410`, completed full declared
  validation over all `3,082` rows at health-only loss
  `2.3020349857793234` in `136.2518 s`, and entered the frozen exact
  generation callback. Automatic generation batch size 64 was established for
  segment 1 at `05:16:03 SAST`.
- No step-3082 exact artifact exists yet, so the health-only loss is not
  selector evidence and no ranking changes. Based on prior callback duration,
  the exact artifact is tentatively expected around `06:10--06:25 SAST`.
  The job remains healthy on `srvrocgpu010` A100-40GB with no targeted fault.
- B2 `1248577` remains Resources-pending with scheduler estimate
  `2026-08-22 02:34:16`; b3 `1250970` remains Priority-pending. Owned work is
  at the three-job cap, entirely A100-40GB. AfriHG remains `3/11`
  scientifically terminal-valid and global freeze remains `0/8`; b0 exact
  retained/final tensor equivalence is still pending. Quota is
  `88.6%/40.6%`, Kombuys remains read-only with RTX 5090 untouched, held-out
  access remains `0`, and Sheet E/F/G remain blank.

## AfriHG b3 queued within the three-job cap — 04:54 SAST

- Stage-B b1 job `1248576` remains healthy on `srvrocgpu010` A100-40GB near
  step `2747/15410`; its first training boundary is expected around
  `05:10--05:20 SAST`, followed by full validation and frozen exact scoring.
  B2 job `1248577` remains pending for `Resources`.
- With exactly two owned jobs, no b3 duplicate, and the b3 output root absent,
  the next preregistered seed-42 candidate was submitted once as job
  `1250970`. It uses the same immutable
  `uniform-adapter-hpo-20260811-6aabf717` snapshot and the required
  `nlpgroup/a100/nlpgroup`, one `gpu:ampere`, 24-hour, 8-CPU, canonical
  working-directory envelope. It is pending for `Priority`, bringing owned
  work to the maximum three jobs without introducing A100-80GB or L40S work.
- Scientific progress is unchanged: b0 still awaits exact retained/final
  tensor equivalence, AfriHG remains `3/11` terminal-valid, and the global
  freeze remains `0/8`. Quota is home `88.6%`, scratch `40.6%`; Kombuys
  remains read-only with RTX 5090 untouched, held-out access remains `0`, and
  Sheet E/F/G remain blank.

## AfriHG b0 completes; b1 starts — 02:55 SAST

- Stage-B b0 job `1248575` completed `0:0` at `02:34:15 SAST` after
  `18:47:37` on `srvrocgpu010`. Its terminal step-15410 artifact has exactly
  `128` rows (`64/64` Xho/Zul), zero empty/debug-empty predictions, `64/64`
  unique predictions per language, and clean `[BOS]` and
  `[EOS]<|assistant|>` prompt boundaries. Terminal Xho/Zul chrF is
  `23.246064347495587/25.29784315222218`, mean `24.27195374985888`.
- The terminal score does not improve the retained validation-only best:
  checkpoint 12328 remains selected at mean chrF `24.419330770518705`.
  Terminal artifact, final-adapter, retained-checkpoint, execution-manifest,
  and trial-record SHA-256 values are respectively
  `3963d5be67a7492ed468e1fcc561ef394429e7715af62b7703e36538dc63939d`,
  `36283250957cac594ddf20926a95e8110b11b12cffce4502827d87032265adcd`,
  `43e5a923320415a7545f10d7d32054150a973513dfa473a90447e4a4614a8efb`,
  `630e9a5506288e57a8b6440e30941b1d1786d3870a039f26b39ec29b915da36a`,
  and `c3067b07bc81719ca08eae359e2c8131fd195f61821a4b921e40ef399b1a0fcc`.
  Exact retained-versus-final tensor equivalence still requires a compute-node
  check, so b0 is operationally complete but is not yet counted as
  scientifically terminal-valid; AfriHG remains `3/11` and the global freeze
  remains `0/8`.
- B1 job `1248576` started automatically at `02:34:16 SAST` on the same
  A100-40GB node. Its 694-file execution manifest verified, the pure-GDN CUDA
  fast path is available, and declared train/validation coverage is
  `24,649/3,082`; no targeted fault is visible. B2 job `1248577` remains
  pending for `Resources`, with scheduler estimate `2026-08-22 02:34:16`.
  Based on b0 wall time, b1's rough terminal ETA is around `21:20--21:40 SAST`
  on 21 August, subject to callback timing.
- HEX quota is home `88.6%`, scratch `40.6%`. All owned work remains
  A100-40GB only; Kombuys remains read-only with RTX 5090 untouched. Held-out
  adapter access remains `0`, Sheet E/F/G remain blank, and
  General/Monolingual/publication remain blocked.
