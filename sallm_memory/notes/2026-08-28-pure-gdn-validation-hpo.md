# Pure-GDN validation HPO — 2026-08-28

## Corrected a0 recovery steady — 23:52 SAST

- Job `1276350` is healthy at step `11587/13640` on A100-80GB. The live rate
  remains stable near `6.6` seconds per step, the fault scan is empty, and the
  terminal window remains approximately 04:15--05:00 SAST on 29 August.
  General remains Stage-A `2/3` and global freeze `7/8` until verification.
- Quota remains `88.6%/44.3%`; one other association job is active and two
  A100-80GB cards are free but not yet eligible for Stage-B. Held-out access is
  zero and Sheet E/F/G remain blank.

## Corrected a0 recovery remains healthy — 22:52 SAST

- Job `1276350` is healthy at step `11043/13640` on A100-80GB with a stable
  observed rate near `6.7` seconds per optimizer step and no runtime fault.
  Training-only completion projects near 03:43 SAST on 29 August; allowing for
  the frozen terminal validation and artifact checks gives a current terminal
  window of approximately 04:15--05:00 SAST. General remains Stage-A `2/3`
  and the global freeze remains `7/8` until that artifact and roundtrip verify.
- Home/scratch quota remains `88.6%/44.3%`. One other association job is
  active, leaving two A100-80GB cards free, but no Stage-B job is eligible
  before this final Stage-A gate closes. Held-out access is zero and Sheet
  E/F/G remain blank.

## Corrected a0 resume confirmed — 22:40 SAST

- Job `1276350` is healthy on A100-80GB. Trainer progress jumped directly to
  step `10913/13640` from checkpoint-10912 and reached `10940` with no fault
  marker or step-0 restart. Home/scratch quota is `88.6%/44.3%`; one other
  association job is active, leaving two A100-80GB cards free. No later
  General trial is eligible until this Stage-A result verifies terminally.

## Corrected a0 recovery submitted — 22:34 SAST

- The user explicitly authorized one prospective implementation-correction
  recovery. Fail-closed preflight reverified source job `1271042` as `TIMEOUT`,
  failed recovery `1275241` as cancelled, no active owned duplicate, no final
  adapter or prior resume-specific manifest, and all five checkpoint-10912
  hashes exactly unchanged.
- Immutable overlay
  `uniform-adapter-hpo-general-a0-resume-correction-20260828-45fec06b`
  verifies 695 source/config files. Launcher, batch, and deployment-manifest
  SHA-256 values are `45fec06b...ceab`, `7f3d0644...0c2f`, and
  `8b6a744b...d9bb6`.
- The single authorized replacement is job `1276350`, submitted under
  `nlpgroup80/a100/nlpgroup80` with one `gpu:ampere80`, 24 hours, eight CPUs,
  and the frozen working directory. It started at 22:33:30. Its
  resume-specific execution manifest and sidecar match at SHA-256
  `c47a2cf9...879bd`, explicitly recording checkpoint-10912 and the corrected
  launcher hash. Trainer-level resume is confirmed: progress jumped directly
  to step `10913/13640` and continued past `10919`, with no step-0 restart or
  fault marker.

## A0 recovery contained after ignored resume — 11:49 SAST

- Recovery job `1275241` started at 11:39:37 but its log proved
  `resume_from_checkpoint=None` and a fresh step-0 start. It was cancelled at
  11:45:26 after 5:49, before any checkpoint or validation metric.
- Root cause is exact: the immutable 11 August launcher never consumes the
  exported resume variable. Checkpoint-10912's five state hashes remain
  unchanged, although the invalid start overwrote the root execution manifest,
  sidecar, and trial record; all are preserved as failure provenance.
- A path-validating resume overlay with SHA-256
  `45fec06bf2ea71d4f69a43a686c88b9b8d895750d85b797ed104ac59271cceab`
  passes local positive and missing-state fail-closed checks. It is prepared
  only. The explicit no-second-recovery rule blocks deployment or submission
  without new scientific authorization.
- General Stage-A remains `2/3` terminal-valid and the global freeze remains
  `7/8`. No owned GPU job is active; held-out access is zero and Sheet E/F/G
  remain blank.

## General a2 terminal-valid; a0 recovery starts — 11:45 SAST

- General Stage-A a2 `1274502` completed `0:0` after the frozen step-10912
  boundary and patience-2 early stopping. Its terminal artifact and sidecar
  match at SHA-256
  `98dffa8eb2ddcbdeb1338b8eb6593dce544dc40adb38513a09e6a55b3355c1cb`.
  Coverage is exactly 22,167 processed rows across all six registered families,
  including all 3,082 AfriHG rows with `1305/1777` Xho/Zul coverage. The fourth
  validation-only macro NLL is `0.979855076433135`; retained checkpoint 5456
  remains the within-run best at `0.9284030074439155`.
- A reusable path-explicit verifier was added without changing tensor
  comparison semantics; SHA-256 is
  `a138101a0be2dbb856c07bc2d5d5e59fd31436c2c30de4788287bf5ba0d75184`.
  CPU job `1275799` completed `0:0` and proved exact retained-to-final equality
  across all 424 adapter tensor keys and 71,762,560 values. General Stage-A
  therefore advances to `2/3` scientifically terminal-valid.
- Exact a0 checkpoint-10912 recovery `1275241` started automatically on the
  released A100-80GB at 11:39:37. It is the only owned GPU job and remains the
  final Stage-A gate. Quota is home `88.6%` and scratch `44.2%`; held-out access
  is zero, Sheet E/F/G remain blank, and the global family freeze stays `7/8`.

## General a2 healthy; shared 80GB cap is the critical path — 09:55 SAST

- General Stage-A a2 `1274502` remains healthy on A100-80GB at approximately
  `10074/13640`. At the observed `~6.58 s/step`, it should enter the frozen
  step-10912 validation boundary around 11:25 SAST. If the second consecutive
  non-improvement triggers the frozen patience-2 stop, terminal validation is
  expected around 12:15--12:45; otherwise the 24-hour allocation ends at
  14:23:46.
- Exact a0 checkpoint-10912 recovery `1275241` remains pending and is the next
  owned job scheduled for the released card. No duplicate was submitted.
- The `nlpgroup80` association currently uses all four A100-80GB cards:
  `1274502` is ours and `1274494/1274495/1274496` are another group member's
  48-hour jobs. Their current allocation ends are 29 August 13:08/13:08/16:41,
  and that user has additional jobs queued. The shared association cap, not a
  model or evaluator fault, is therefore the current throughput constraint.
- General remains `1/3` terminal-valid and the global family freeze remains
  `7/8`. Quota is home `88.6%` and scratch `44.1%`; held-out access is zero,
  Sheet E/F/G remain blank, and no L40S or Kombuys model work is active.

## General a2 third boundary verifies — 06:37 SAST

- General Stage-A a2 job `1274502` completed its frozen step-8184 validation
  and resumed healthy A100-80GB training.
- The sidecar matches artifact SHA-256
  `1b93d30a686c63835866408cbd0468edab453bdae9021f0d4019e3899457cea3`.
  Coverage remains exact across all 22,167 processed rows in the six
  registered families, including all 3,082 AfriHG rows with `1305/1777`
  Xho/Zul coverage, under `equal_family_assistant_token_nll_v1`.
- Validation-only macro NLL is `0.9491996698865588`, worse than the retained
  step-5456 value `0.9284030074439155`; checkpoint 5456 therefore remains the
  within-run best. This is interim evidence and does not select or freeze a
  General winner.
- A0 recovery `1275241` remains `AssocGrpGRES` pending with scheduler estimate
  28 August 14:23 SAST. General remains `1/3` terminal-valid and global family
  freeze remains `7/8`; quota is home `88.6%`, scratch `44.1%`, held-out
  access is zero, and Sheet E/F/G remain blank.

## General a2 second boundary verifies — 01:32 SAST

- General Stage-A a2 job `1274502` completed its frozen step-5456 validation
  and resumed healthy training.
- The artifact sidecar verifies SHA-256
  `9db07a487a83365545a15eea211edb8d2fb06f63a76511f84211f37a047e6505`.
  It contains exact 22,167-row six-family coverage, including all 3,082
  AfriHG rows with `1305/1777` Xho/Zul coverage, and the registered
  equal-family assistant-token NLL protocol.
- Validation-only macro NLL improved from `0.9487367487872976` at step 2728
  to `0.9284030074439155` at step 5456. This remains interim evidence and
  does not select or freeze a winner.
- A0 recovery `1275241` remains `AssocGrpGRES` pending with scheduler estimate
  28 August 14:23 SAST. General remains `1/3` terminal-valid and global family
  freeze remains `7/8`; held-out access is zero and Sheet E/F/G remain blank.
