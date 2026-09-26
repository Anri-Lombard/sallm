# Pure-GDN validation HPO — 2026-08-29

## General b3 approaches its second boundary — 23:37 SAST

- B3 `1276434` is healthy at step `5353/13640`, about 12 minutes from the
  frozen step-5456 boundary. Its artifact is absent as expected and remains
  due around 00:00--00:20 on 30 August.
- B0/B1/B2 `1276431/1276432/1276433` remain healthy near steps
  `10014/9830/9983`; all four targeted fault scans are empty. Their
  step-10912 artifact window remains approximately 01:30--02:05.
- All four A100-80GB cards remain occupied by required work. Stage-B is
  `0/8` terminal-valid, global freeze `7/8`, quota `88.6%/44.5%`, held-out
  access zero, and Sheet E/F/G blank. No action is due.

## General Stage-B remains healthy — 22:34 SAST

- B0/B1/B2/B3 `1276431/1276432/1276433/1276434` remain healthy on all four
  A100-80GB cards near steps `9458/9283/9425/4804`; Slurm reports every job
  running with no restart or failure state, and targeted fault scans are empty.
- B3's step-5456 artifact remains due around 23:55--00:20. B0--b2 remain on
  course for step-10912 artifacts around 01:30--02:05 on 30 August.
- Stage-B remains `0/8` terminal-valid, global freeze `7/8`, quota
  `88.6%/44.5%`, held-out access zero, and Sheet E/F/G blank. No intervention
  or submission is due while all four cards are occupied.

## General Stage-B continues cleanly — 21:34 SAST

- B0/B1/B2/B3 remain healthy near steps `8919/8754/8892/4276` with empty
  fault scans and stable step times. All four A100-80GB cards remain occupied
  by required work.
- B3's step-5456 artifact remains projected around 23:55--00:20; b0--b2
  step-10912 artifacts remain projected around 01:30--02:05 on 30 August.
- Stage-B remains `0/8`, global freeze `7/8`, quota `88.6%/44.5%`, held-out
  access zero, and Sheet E/F/G blank. No intervention is due.

## General b0--b2 third boundaries verify — 20:32 SAST

- B0/B1/B2 `1276431/1276432/1276433` produced sidecar-matching step-8184
  artifacts with exact 22,167-row six-family coverage and `1305/1777`
  AfriHG Xho/Zul rows. Artifact SHA-256 values are
  `d5e82ffe3638d8751f0ab03ab5413f9b37c7c8fab0bd18cb6a1575bd8e9e9982`,
  `b42dd821df0d43136f2ed6129209ce5197c7fd543ef664fa4b546342cda22def`,
  and `8c945d292dde51b65c3f22a653a50511687929d6ac57b5752603617417be3205`.
- Validation-only macro NLL is `0.9508059313758785`, `0.9539537704610211`,
  and `1.0190495201500418`. B0 records its first within-run non-improvement,
  retaining step 5456; b1 and b2 improve and retain step 8184. No
  cross-candidate ranking or selection is performed.
- All four jobs remain healthy near `8376/8220/8350/3738`. B3's step-5456
  artifact is projected around 23:55--00:20, while b0--b2 step-10912
  artifacts are projected around 01:30--02:05 on 30 August.
- Stage-B remains `0/8`, global freeze `7/8`, quota `88.6%/44.5%`, held-out
  access zero, and Sheet E/F/G blank.

## General b3 first artifact verifies — 19:31 SAST

- B3 `1276434` produced a sidecar-matching step-2728 artifact at SHA-256
  `4872027be09dfe261cf659e7fb74d9811c746a172b0c374c9eb159889cc1849e`.
  Coverage is exactly 22,167 rows across all six families, including all 3,082
  AfriHG rows (`1305/1777` Xho/Zul). Validation-only macro NLL is
  `1.006736571720808`; this is within-run retention evidence only.
- B3 resumed healthy near step `3208`. B0/B1/B2 remain healthy near
  `7986/7842/7965`, with their step-8184 artifacts still due around
  20:05--20:35 SAST. All fault scans are empty.
- Stage-B remains `0/8` terminal-valid, global freeze `7/8`, quota
  `88.6%/44.5%`, held-out access zero, and Sheet E/F/G blank.

## General b3 enters its first frozen boundary — 18:31 SAST

- B3 `1276434` reached exact step 2728 and entered its first frozen validation;
  the artifact is not yet present. Its artifact window is approximately
  18:45--19:05 SAST.
- B0/B1/B2 remain healthy near `7446/7311/7424` with empty fault scans. Their
  step-8184 artifacts remain projected around 20:05--20:35 SAST. All four
  A100-80GB cards remain occupied by required work and no intervention is due.
- Stage-B remains `0/8`, global freeze `7/8`, quota `88.6%/44.4%`, held-out
  access zero, and Sheet E/F/G blank.

## General Stage-B steady on four cards — 17:30 SAST

- B0/B1/B2/B3 remain healthy near steps `6906/6781/6884/2287` with empty
  fault scans and stable `6.6--6.9` second step times. B3's first artifact is
  still absent as expected and remains projected around 18:30--18:50 SAST.
- All four A100-80GB cards remain productively occupied. The next b0--b2
  artifacts remain projected around 20:05--20:35 SAST; no job action is due.
- Stage-B remains `0/8`, global freeze `7/8`, quota `88.6%/44.4%`, held-out
  access zero, and Sheet E/F/G blank. The 4 September target is unchanged.

## General Stage-B remains healthy; delayed-heartbeat clock corrected — 16:31 SAST

- An apparent two-hour progress gap was a delayed heartbeat timestamp, not a
  compute stall. Live HEX time, log mtimes, and consecutive progress records
  show all four jobs advancing normally near `6.7` seconds per step. No job
  was changed.
- B0/B1/B2/B3 are healthy near steps `6380/6263/6362/1762` with empty fault
  scans. B3's first frozen artifact is projected around 18:30--18:50 SAST;
  the next b0--b2 artifacts are projected around 20:05--20:35 SAST.
- The preceding second-boundary entry is corrected from 14:24 to 16:24 SAST.
  Stage-B remains `0/8`, global freeze `7/8`, quota `88.6%/44.4%`, held-out
  access zero, and Sheet E/F/G blank.

## General Stage-B second boundaries verify — 16:24 SAST

- B0/B1/B2 `1276431/1276432/1276433` produced sidecar-matching step-5456
  artifacts with exact 22,167-row six-family coverage and `1305/1777`
  AfriHG Xho/Zul rows. Artifact SHA-256 values are
  `b403ba75b32e5718306f7cd9f4ddf29f83fe949532ac0589c8486122190850ce`,
  `e7c9a0271117a7ae94de5728a68ab4e3ca8c7617ed831718895612779410bd64`,
  and `fbdcc51b89b2ecdfa7e4635e540703cd0918bc943693a913f3af9b90fd822bcf`.
- Validation-only macro NLL is `0.9451928643283297`, `0.9699125326068634`,
  and `1.0699096838669606` for b0/b1/b2. Each improves its own step-2728
  value, so step 5456 is retained within the run; no cross-candidate ranking
  or selection is performed.
- All four jobs are healthy: b0/b1/b2 resumed near `6332/6217/6317`, while
  b3 `1276434` is near step `1711`. The next b0--b2 frozen artifacts are
  projected for approximately 20:05--20:35 SAST.
- Stage-B remains `0/8` terminal-valid, global freeze `7/8`, quota
  `88.6%/44.4%`, held-out access zero, and Sheet E/F/G blank.

## General Stage-B fills the fourth card — 13:10 SAST

- B3 job `1276434` started on A100-80GB at 13:08:27 after its preregistered
  queue wait. Its execution-manifest SHA-256 is
  `1344f1c7d2ea38b1f90ffc0986e49e8296eaa44f8f3668b6029fea1ea4255bdf`;
  the job has the exact `nlpgroup80/a100/nlpgroup80` resources, verified all
  694 frozen source/config files, and reached healthy model loading with the
  GatedDeltaNet fast path available and no fault marker.
- B0/B1/B2 `1276431/1276432/1276433` remain healthy near steps
  `4743/4661/4737`. All four eligible A100-80GB association cards are now
  occupied by the four owned required Stage-B jobs; no later candidate is
  eligible for submission until a slot releases.
- Stage-B remains `0/8` terminal-valid and global freeze `7/8`. Quota is
  `88.6%/44.4%`; held-out access is zero and Sheet E/F/G remain blank.

## General Stage-B continues after first artifacts — 12:09 SAST

- B0/B1/B2 `1276431/1276432/1276433` remain healthy near steps
  `4217/4144/4207` with empty fault scans. Their step-5456 artifacts are absent
  as expected and are now projected for approximately 14:40--15:10 SAST.
- B3 `1276434` remains `AssocGrpGRES`-pending behind other-user job `1274494`;
  no other user's job was modified. Stage-B remains `0/8` terminal-valid and
  global freeze `7/8`.
- Quota is `88.6%/44.4%`; held-out access remains zero and Sheet E/F/G remain
  blank. The complete-table target remains 4 September best case.

## General Stage-B first artifacts verify — 10:09 SAST

- B0/B1/B2 `1276431/1276432/1276433` produced sidecar-matching step-2728
  artifacts with exact 22,167-row six-family coverage and `1305/1777`
  AfriHG Xho/Zul rows. Artifact SHA-256 values are
  `ee49a2b0686c8b07ac207c3e3ca0f9e61fdf4a94bbe245e4ab98ee321cd8bb33`,
  `5159eae7098d17024b5497672af15b39864824aa061b2cc532bb8cbd791ffa5d`,
  and `44e2ce9e7d052fee8b352b262d88f3ce2b8536623b2a29dabd686db9a22b74d2`.
- Validation-only macro NLL is `0.9817947401411412`, `1.0586443713118137`,
  and `1.240997314146131` for b0/b1/b2. These are within-run checkpoint
  retention values only; no candidate ranking or selection is performed.
- All three jobs resumed healthy near steps `4173/4100/4167`; their next
  frozen artifacts are projected around 12:45--13:15 SAST. B3 `1276434`
  remains `AssocGrpGRES`-pending behind other-user job `1274494`.
- Stage-B remains `0/8` terminal-valid, global freeze `7/8`, held-out access
  zero, and Sheet E/F/G blank. Quota is `88.6%/44.4%`.

## General Stage-B first boundaries due — 08:02 SAST

- B0/B1/B2 `1276431/1276432/1276433` remain healthy near steps
  `2162/2122/2153` with empty fault scans. Their first frozen validation
  artifacts remain projected around 09:15--09:35 SAST.
- B3 `1276434` remains `AssocGrpGRES`-pending behind other-user job `1274494`.
  Stage-B remains `0/8`, global freeze `7/8`, held-out access zero, and Sheet
  E/F/G blank; quota remains `88.6%/44.3%`.

## General Stage-B approaches first boundaries — 07:01 SAST

- B0/B1/B2 `1276431/1276432/1276433` remain healthy near steps
  `1860/1822/1851` on three A100-80GB cards with empty fault scans. Their
  first frozen validation artifacts are now projected around 08:50--09:20.
- B3 `1276434` remains `AssocGrpGRES`-pending behind other-user job `1274494`,
  with the same movable 13:08 Slurm estimate. Stage-B remains `0/8` and global
  freeze `7/8`; held-out access is zero and Sheet E/F/G remain blank.

## General Stage-B remains healthy — 05:59 SAST

- B0/B1/B2 `1276431/1276432/1276433` are healthy near steps
  `1044/1023/1034` on three A100-80GB cards with empty fault scans. Their
  first frozen boundaries remain projected around 09:10--09:40 SAST.
- B3 `1276434` remains `AssocGrpGRES`-pending behind other-user job `1274494`;
  its current Slurm estimate remains 13:08 SAST. Stage-B is `0/8`, global
  freeze `7/8`, held-out access zero, and Sheet E/F/G blank.

## General Stage-B first wave healthy — 04:58 SAST

- B0/B1/B2 jobs `1276431/1276432/1276433` are healthy near steps
  `488/484/488` on three A100-80GB cards, with empty fault scans and verified
  startup manifests. Their first frozen step-2728 boundaries are projected
  around 09:10--09:40 SAST.
- B3 `1276434` remains `AssocGrpGRES`-pending because other-user job `1274494`
  holds the fourth card; Slurm currently estimates 13:08 SAST, but that is not
  guaranteed. Stage-B remains `0/8` terminal-valid and global freeze `7/8`.
- Quota remains `88.6%/44.3%`; held-out access is zero and Sheet E/F/G remain
  blank. The 4 September best-case table target is unchanged.

## General Stage-A closes; Stage-B starts — 04:00 SAST

- Corrected a0 recovery `1276350` completed `0:0` at 03:56:33 SAST. Its
  terminal artifact and sidecar verify at SHA-256
  `95c3abc990a9c7774b5d15ccecd492194dc441e0b27c47f698106388742f6e05`.
  Coverage is exactly 22,167 processed rows across all six families, including
  all 3,082 AfriHG rows (`1305/1777` Xho/Zul). Terminal validation-only macro
  NLL is `0.9830990526022072`, improving step 10912, so checkpoint 13640 is
  retained.
- Retained trainer-state, adapter, and final-adapter SHA-256 values are
  `dafe76d421c5183b8e50799e4c62b9b8c6a067fce673dcddda2cb3d367e81b22`,
  `ed6c826006fc08301c6b93f311b433030fcd4577716fe1431491b30531ff560f`,
  and `f1386c7abe4ec5d18b09966ed666dcdb86409c44a842a61d5c1197d18018c0b3`.
  CPU verifier `1276429` completed `0:0` and proved exact equality across 424
  keys and 71,762,560 values. General Stage-A is scientifically `3/3`.
- After absent-output/no-duplicate and immutable-registry checks, Stage-B b0,
  b1, b2, and b3 were submitted once as jobs `1276431`, `1276432`, `1276433`,
  and `1276434`. B0--b2 are running on three A100-80GB cards; b3 is
  `AssocGrpGRES`-pending because other-user job `1274494` holds the fourth.
  The three live execution manifests were written successfully with SHA-256
  `67b89552...26103d`, `ecb840c3...bb404`, and `32371bdd...274e`.
- Global freeze remains `7/8`; Stage-B is `0/8` terminal-valid. Held-out
  access is zero, Sheet E/F/G remain blank, quota is `88.6%/44.3%`, and the
  complete-table target remains 4 September best case.

## Corrected General a0 recovery reaches 97% — 02:53 SAST

- Job `1276350` is healthy at step `13236/13640` on A100-80GB. The fault scan
  remains empty and the live estimate is about 45 minutes to the end of
  training. The terminal validation/artifact window remains 04:15--05:00.
- General remains Stage-A `2/3`, global freeze `7/8`, held-out access zero,
  and Sheet E/F/G blank. Quota remains `88.6%/44.3%`.

## Corrected General a0 recovery reaches 93% — 01:53 SAST

- Job `1276350` is healthy at step `12684/13640` on A100-80GB. The observed
  rate remains stable near `6.6` seconds per step and the fault scan is empty.
  Training-only completion remains near 03:38 SAST and the terminal
  validation/artifact window remains approximately 04:15--05:00.
- General remains Stage-A `2/3` and global freeze `7/8`; held-out access is
  zero and Sheet E/F/G remain blank. Quota remains `88.6%/44.3%`.

## Corrected General a0 recovery steady — 00:52 SAST

- Job `1276350` is healthy at step `12141/13640` on A100-80GB. The live rate
  remains stable near `6.6` seconds per optimizer step and the fault scan is
  empty. Training-only completion remains near 03:38 SAST; the frozen terminal
  validation and artifact window remains approximately 04:15--05:00.
- General remains Stage-A `2/3` and global freeze `7/8` until the terminal
  artifact, exact 22,167-row coverage, sidecar, and retained-to-final adapter
  roundtrip verify. Held-out access is zero and Sheet E/F/G remain blank.
- Home/scratch quota is `88.6%/44.3%`. One other association job is active,
  leaving two A100-80GB cards free, but Stage-B is not yet eligible.
