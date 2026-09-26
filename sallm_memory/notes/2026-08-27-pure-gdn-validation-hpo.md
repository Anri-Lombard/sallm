# Pure-GDN validation HPO — 2026-08-27

## A0 queue estimate improves — 23:31 SAST

- General a2 `1274502` is healthy at `4755/13640` on A100-80GB.
- The existing a0 exact-resume job `1275241` remains `AssocGrpGRES` pending,
  but its scheduler estimate improved to 28 August 14:23 SAST from 29 August
  13:08. No duplicate was submitted.
- General remains `1/3` terminal-valid and global freeze `7/8`; held-out
  access is zero and Sheet E/F/G remain blank.

## A0 exact resume authorized and queued — 20:38 SAST

- The user authorized the one prospective exact same-trial recovery frozen
  before timeout. The preflight confirmed source job `1271042` is terminal
  `TIMEOUT`, no active General a0 duplicate exists, no final adapter or prior
  resume manifest exists, and all five checkpoint-10912 hashes exactly match
  the preregistration.
- Resume job `1275241` was submitted once using the unchanged immutable
  `uniform-adapter-hpo-20260811-6aabf717` launcher and exact checkpoint path.
  Its Slurm envelope is `nlpgroup80/a100/nlpgroup80`, one
  `gpu:ampere80`, 24 hours, eight CPUs, and
  `--chdir=/home/lmbanr001/masters/sallm`. Batch-script readback confirms the
  only scientific execution addition is
  `SALLM_HPO_RESUME_FROM_CHECKPOINT=.../checkpoint-10912`.
- Job `1275241` is `AssocGrpGRES` pending because all four A100-80GB cards are
  occupied. A2 `1274502` remains running on the same GPU family. General
  remains `1/3` terminal-valid and global freeze remains `7/8`; held-out
  access is zero and Sheet E/F/G remain blank.

## A2 first boundary verifies — 20:28 SAST

- General Stage-A a2 job `1274502` is healthy on A100-80GB near step
  `3115/13640`. Its step-2728 artifact and sidecar match at SHA-256
  `889c17edad1bc6f749337e838f2007ff5ecf80463d63aa8e4d62db80e59cf68f`.
- Coverage is exactly 22,167 processed rows across the six registered
  families: SIB 2,970, News 3,095, NER 10,760, POS 1,800, AfriHG 3,082, and
  T2X 460, including exact AfriHG Xho/Zul counts 1,305/1,777. The frozen
  equal-family macro NLL is `0.9487367487872976`. This remains interim
  validation-only evidence and does not select or freeze a candidate.
- Other-user jobs `1274494/1274495/1274496` occupy the other three A100-80GB
  cards. The A100-40GB node is also fully occupied, while all 20 L40S cards
  are occupied with a long pending queue. No family switch or additional job
  is scientifically useful now. General remains `1/3` terminal-valid and
  global freeze remains `7/8`.
- Quota is home `88.6%` and scratch `44.1%`. The exact a0 checkpoint-10912
  resume remains unsubmitted pending scientific authorization; held-out
  access is zero and Sheet E/F/G remain blank.

## A0 timeout confirmed and recovery remains gated — 17:26 SAST

- General a0 job `1271042` reached step `12131/13640` without a model, data,
  evaluator, or training error, then Slurm ended it at `16:41:29` for the
  exact 24-hour time limit. Its terminal state is `TIMEOUT`; no final adapter
  exists and no post-step-10912 checkpoint was written.
- The complete checkpoint-10912 adapter, optimizer, scheduler, RNG, and
  trainer-state hashes reverify exactly against the prospective recovery
  draft. The failure mechanism is therefore confirmed as wall time alone.
  The exact-resume recovery remains unsubmitted pending scientific
  authorization.
- A2 job `1274502` remains healthy near step `1585/13640`. Other-user jobs
  `1274494/1274495/1274496` now occupy the other three A100-80GB cards, so no
  card is currently idle. General Stage-A remains `1/3` terminal-valid and
  the global family freeze remains `7/8`.

## A0 wall-time mechanism diagnosed — 16:27 SAST

- A0 job `1271042` remains healthy near `11964/13640`, but its allocation ends
  at `16:41:18`; the remaining training cannot finish at the observed rate.
  The cause is the fixed 24-hour allocation, not a model, data, evaluator, or
  training fault.
- Complete checkpoint `10912` contains adapter, optimizer, scheduler, RNG, and
  trainer state and remains the validation-only best at macro NLL
  `0.9838042431257504`. A prospective exact-resume protocol is drafted in
  `2026-08-27-pure-gdn-general-a0-walltime-recovery-preregistration.md`.
  It does not authorize submission; preserve the terminal state and obtain
  scientific authorization before recovery.
- A2 `1274502` remains healthy near step 1037. Other-user jobs
  `1274494/1274495` still occupy the remaining two A100-80GB cards. No
  independent General work is eligible until all three Stage-A candidates are
  terminal-valid.

## General a1 terminal-valid; a0 fourth boundary preserved — 15:24 SAST

- General Stage-A a1 job `1271043` completed `0:0` after its step-10912
  validation boundary and early stopping. Its sidecar matches artifact
  SHA-256 `acd76a934a9d7d5e12221741c451deb305f39dccd8cc4cf99310fb91fc86ebc6`;
  coverage is exactly 22,167 processed rows across all six registered
  families, including all 3,082 AfriHG rows. The retained validation-only
  checkpoint is step `8184` with macro NLL `0.9499744008920644`.
- CPU verifier `1274655` completed `0:0` and proved exact retained-to-final
  equality across all 424 adapter tensor keys and 71,762,560 values. General
  Stage-A is therefore `1/3` scientifically terminal-valid.
- A0 job `1271042` remains healthy at approximately `11433/13640`. Its
  step-10912 artifact and sidecar match at SHA-256
  `439f31199f759ba623f667faf8e6974695cd954cf8c95c4288bced56884c124d`,
  with exact full coverage and macro NLL `0.9838042431257504`; it improved
  again and therefore continued training. With about 80 minutes left on its
  24-hour allocation, terminal completion is no longer feasible at the live
  step rate. Preserve any timeout and diagnose before an authorised recovery.
- A2 job `1274502` started immediately when a1 released its card and remains
  healthy near step 500. Other-user jobs `1274494/1274495` continue to occupy
  the remaining two A100-80GB cards. Quota is home `88.6%`, scratch `44.1%`;
  held-out access is zero and Sheet E/F/G remain blank.

## AfriHG freezes scientifically; General a2 queued — 13:24 SAST

- Seed-87 jobs a2 `1271041` and isolated b7 `1271354` completed `0:0` with
  clean 128-row terminal artifacts. Terminal artifact SHA-256 values are
  `28a02a56fe552ce2e8fc1ae9e1c491aac352dd989e8f6d3183d53992b2ad5d27`
  and `b28a4f132e91fb6fffa55f166d8d247345208a78e77330ce53d3b59a54057fe5`.
  Both have exact `64/64` Xho/Zul coverage and unique predictions, zero empty
  output, clean prompt boundaries, and no generated EOS marker.
- A2 retained validation-only checkpoint `12328` at mean chrF
  `25.284648928979008`; b7 retained checkpoint `15410` at
  `25.274444347225618`. CPU verifiers `1274503/1274504` completed `0:0` and
  proved exact retained-to-final equality across all 424 tensor keys.
- The preregistered seed-42 ranking remains decisive and selects b7; seeds 13
  and 87 are robustness evidence only. AfriHG is frozen to seed-42 b7
  checkpoint `15410`. Freeze artifact
  `sallm_memory/artifacts/2026-08-27-pure-gdn-afrihg-freeze.json` has SHA-256
  `3cc31f0b2221bf30f1c795aa58420012313a56976f0d06b23ac49836a690f5db`.
  Global family freeze advances to `7/8`.
- Required corrected General Stage-A a2 seed-42 was submitted after
  absent-output/no-duplicate checks as job `1274502`. It is pending on
  `AssocGrpGRES`: General a0/a1 use two A100-80GB cards and other user jobs
  `1274494/1274495` use the other two. No other eligible candidate is added
  merely to fill capacity. Quota is home `88.6%`, scratch `44.0%`; held-out
  access is zero and Sheet E/F/G remain blank.

## Isolated b7 roundtrip-path amendment — 13:20 SAST

- Both seed-87 confirmations completed `0:0`. A2 retained checkpoint `12328`;
  isolated b7 retained checkpoint `15410`. Before roundtrip verification, the
  existing verifier is amended to accept an optional explicit `OUTPUT_ROOT`
  while retaining its canonical path as the default. This changes only which
  already-written adapter pair is loaded; key equality, tensor equality,
  hashing, resources, and all scientific rules remain unchanged.
- The isolated b7 path is
  `/scratch/lmbanr001/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/afrihg/confirm/b7/seed_87_a10080_retry_20260826`.
  The verifier source must be hashed before submission, and each result must
  be absent with no active duplicate. No validation metric informs this
  execution-only path correction.

## AfriHG confirmations enter terminal validation — 12:16 SAST

- A2 seed-87 `1271041` and isolated b7 seed-87 `1271354` reached the frozen
  `15410/15410` training endpoint and completed full-loss evaluation with
  losses `2.2606240779374187` and `2.3207444593861535`. Both remain running
  in their terminal exact generation callbacks; no terminal artifact or
  adapter roundtrip is accepted yet.
- General a0/a1 `1271042/1271043` remain healthy at approximately
  `9934/9943` of `13640`. All four A100-80GB cards remain occupied; quota is
  home `88.6%`, scratch `43.9%`; held-out access is zero and Sheet E/F/G
  remain blank.

## General third and AfriHG fourth boundaries verify — 09:11 SAST

- General Stage-A a0/a1 `1271042/1271043` completed their frozen step-8184
  evaluations and resumed healthy training. Artifact and sidecar SHA-256
  values match at
  `076eceac554ca520216f3efe03e36f1a9bfd943c63bd88cb1ac7b0549fd69c2a`
  and
  `e36e3b19b0bf8857905834ff6b3eb2283ee99bcff2386a63416bfe9c924a89a8`.
  Both again persist exact 22,167-row coverage and the complete registered
  aggregation inputs. A0 macro NLL improves to `0.9899649856125735`; a1
  improves narrowly to `0.9499744008920644`.
- A2 seed-87 `1271041` and isolated b7 seed-87 `1271354` completed their
  frozen step-12328 exact callbacks at `09:09:16` and `09:01:46` SAST and
  resumed healthy training. Artifact SHA-256 values are
  `d2278446b9ca6d61e1161c3dcdc5831957b181e96383ab6a9fe2fc83333b7516`
  and
  `25f892886d8d5cf5fc5b262912c44d77eb842fe77a0bd7ba98981308a1a206fb`.
  Both have exact `64/64` Xho/Zul coverage and unique predictions, no empty
  output, clean prompt boundaries, and no generated EOS marker.
- A2 Xho/Zul chrF is `24.17764501604741/26.391652841910606` (mean
  `25.284648928979008`). B7 Xho/Zul chrF is
  `24.164427243658533/26.100213870412286` (mean `25.13232055703541`). All
  values remain interim validation-only evidence. All four A100-80GB jobs
  remain active; held-out access is zero and Sheet E/F/G remain blank.

## AfriHG seed-87 third artifacts verify — 06:07 SAST

- A2 seed-87 `1271041` and isolated b7 seed-87 `1271354` completed their
  frozen step-9246 exact callbacks at `05:05:29` and `05:08:51` SAST and
  resumed healthy training. Artifact SHA-256 values are
  `3447c1d1727c3787031559fae901e08edefe197a88d2e4170d976de51fc00247`
  and
  `18f7dccc474290f6052b8d042b71465e2f0ad9234535e1c79e5c4b3bc8c2d230`.
- Both artifacts contain exactly 128 rows with `64/64` Xho/Zul coverage and
  unique predictions, zero empty predictions, zero bad `[BOS]` starts, zero
  bad `[EOS]<|assistant|>` boundaries, and zero generated EOS markers.
- A2 Xho/Zul chrF is `24.13242988510946/26.096308869203828` (mean
  `25.114369377156644`). B7 Xho/Zul chrF is
  `24.081453705605036/25.801406249488117` (mean `24.941429977546576`). These
  remain interim validation-only metrics and do not freeze or select a
  winner.
- General a0/a1 remain healthy beyond their second verified artifacts. All
  four A100-80GB cards remain occupied; quota is home `88.6%`, scratch
  `43.9%`; held-out access is zero and Sheet E/F/G remain blank.

## General second boundaries verify and improve — 04:04 SAST

- General Stage-A a0/a1 `1271042/1271043` completed their frozen step-5456
  evaluations and resumed healthy training. Artifact and sidecar SHA-256
  values match exactly at
  `f195b8bd3b219d60b268814c296698b4bda06d35311d55fb3bbb0dd28645f982`
  and
  `49438559f565c3c266e3716b4587c0a21be0147eea786c1cfc31c89189aaeb8c`.
- Both artifacts again persist exact 22,167-row six-family coverage, exact
  `1305/1777` Xho/Zul AfriHG coverage, summed assistant-token NLL, valid-token
  counts, per-family NLL, and the preregistered equal-family macro NLL. A0
  improves from `1.1208475241267701` to `1.0340581561869973`; a1 improves
  from `0.988150101648468` to `0.9501276418316617`.
- These remain interim validation-only within-run results. Both AfriHG
  confirmations remain healthy near their third boundary. All four
  A100-80GB cards stay occupied; held-out access remains zero and Sheet E/F/G
  remain blank.

## AfriHG seed-87 second artifacts verify — 02:02 SAST

- A2 seed-87 `1271041` and isolated b7 seed-87 `1271354` completed their
  frozen step-6164 exact callbacks at `01:21:12` and `01:23:23` SAST and
  resumed healthy training. Artifact SHA-256 values are
  `70de7ed7465c20650bf158e4d490f2a0da51e5b8bdc175f806252bc3a5bfa1e9`
  and
  `7bb7a97361c32ab173064775f8de0e43eace0eb08a67cfd6e0b82d0eba28c38a`.
- Both artifacts contain exactly 128 rows with exact `64/64` Xho/Zul
  coverage, `64/63` unique Xho/Zul predictions, zero empty predictions, zero
  bad `[BOS]` starts, zero bad `[EOS]<|assistant|>` boundaries, and zero
  generated EOS markers. The single repeated Zul prediction is ordinary
  output diversity, not the audited collapse failure.
- A2 Xho/Zul chrF is `23.421466073858753/24.38484956840115` (mean
  `23.90315782112995`). B7 Xho/Zul chrF is
  `23.32686167781927/24.873816070371294` (mean `24.10033887409528`). These
  remain interim validation-only metrics and do not freeze or select a
  winner.
- General a0/a1 `1271042/1271043` remain healthy beyond their first verified
  full-coverage artifacts. All four A100-80GB cards remain occupied; quota is
  home `88.6%`, scratch `43.9%`; held-out access is zero and Sheet E/F/G
  remain blank.
