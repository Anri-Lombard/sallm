# General b5 and b6 terminal verification

At the 20:20 UTC monitoring pass, b5 1312084 and b6 1312231 were
COMPLETED 0:0 in 5:27:06 and 5:29:02 respectively. Both saved final adapters.
All five scheduled validation artifact sidecars passed for each candidate,
with the exact equal-family protocol, 22,167 processed rows, six-family
raw/processed counts and AfriHG Xhosa/Zulu counts. Original and resume
execution-manifest sidecars passed. The initial read-only sidecar command
used the wrong working directory; it was corrected to resolve each sidecar's
relative filename. No file or scientific job was changed by that check.

CPU verifier 1313549 completed 0:0 in five seconds using the unchanged
verify_adapter_roundtrip.sbatch, SHA-256
`a138101a0be2dbb856c07bc2d5d5e59fd31436c2c30de4788287bf5ba0d75184`.
Both comparisons are exact across 424 keys and 71,762,560 values:

- b5 retained checkpoint 8184; adapter hash
  `6dfeee7a81d38cd387105777a713a322b992c0e096fe1431ff98a83be944d560`;
  final hash `e7600fde2badf200e68f27ab16651122e778c36cd34858e02aa03693e38fc47a`.
- b6 retained checkpoint 13640; adapter hash
  `0d5c74c7931371465f1369daa5f998a52c731dfa5cbcf53e7dde558225d46ce5`;
  final hash `4ac11baa3c64b77a0f6cd32785778f24ef6f5e7132ed47310d50a43a0b32f66f`.

Step-13640 validation artifact hashes, b5 then b6:
`53ed3513dd58f0cf8148fb8e431dc581776c3d9aa1086bbd3eb7eebedf4bea34`,
`7923335c29a3fa6d76f0d0d3844244b5dab0d67cc603b66712a32be2e251cc54`.
Verifier log: /scratch/lmbanr001/masters/sallm/logs/jobs/gdn-general-b56-verify-20260906.out,
SHA-256 `69a6e618da80c27c2d0e4e2d352c64723b397ae45c58dced28de133e0303fd6f`.

General Stage-B is now 7/8 terminal-verified. B3 1312957 is still healthy
at step 12780/13640 on A100-80GB, about 96 minutes of training plus validation
remaining at observation. Keep it running. Three 80GB cards are unallocated,
but no confirmation is eligible until b3 verifies and all eleven seed-42
candidates are ranked under the frozen validation-only rule. Switching b3
would discard unsaved progress and has no demonstrated net benefit.
Quota: home 90.1%, scratch 51.5% (154/300GB). No held-out result was used,
no official test was run and no Sheet value changed. Preserve all jobs.
