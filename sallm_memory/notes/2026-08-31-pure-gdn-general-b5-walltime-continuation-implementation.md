# Pure-GDN General b5 wall-time continuation implementation — 2026-08-31

Status: prospective and frozen at 21:45 SAST before any b5 continuation
submission or recovery manifest creation.

Original b5 job `1277423` ended `TIMEOUT`/`0:0` at the fixed 24-hour wall,
without a final adapter. Checkpoint `10912` is the numerically latest complete
scheduled state. No b5 job is active, no final adapter exists, and no
resume-specific manifest exists.

The original root is preserved in read-only archive
`/scratch/lmbanr001/masters/sallm/recovery_archives/general-stage-b/b5-seed42-pre-recovery.tar`,
mode `0444`, size `245,811,200` bytes, SHA-256
`3f52518bf028a1a921ea7c4718654b5bd21e3f27115aa761c8a689dd92051595`.

Frozen SHA-256 bindings are:

- execution manifest:
  `662ed16cf8705aa230f94a721fa981eae8cf13d3965d212429cd837f551d6c19`;
- adapter:
  `252af47bbbb411a87ed3bc59bec8596a2845d8afc9c0cdfb093759be71434d5f`;
- optimizer:
  `7853c01504a5d823055b33eb31bd10400736a1750217f0074b29d2cf04018294`;
- scheduler:
  `b403be7999c394122e0db5d991abdd751bfe923ce60b4cb961d4b1de55a58413`;
- RNG state:
  `3a5fa191be89653519d53a698bb731547927ec65d27015072e0de1b7e71c27d3`;
- trainer state:
  `51a64b5a7c3f09296dd8cb7b61bd8f4ef54bb0202bae9e695b66fc8f9d695f70`.

The b5-only wrapper
`scripts/run_general_stage_b_b5_walltime_recovery.sh` has SHA-256
`40f4ec9323c991d9cbaee19f03324d832b444f474b24f1ebd029c806bb575c27`.
It binds b5, seed 42, the existing root, checkpoint 10912, every hash above,
the immutable 695-file corrected-launcher snapshot, and the exact original
runtime. It requires an explicit recovery-bundle path. Runtime-verifier
SHA-256 remains
`cfe2eb873f53d83de2c2c296c6a172f74a8059ee4bd11f296785c444e9525b23`.
The intended immutable HEX bundle is
`/home/lmbanr001/masters/sallm_snapshots/pure-gdn-general-stage-b-b5-recovery-20260831-40f4ec93`.
Four focused wrapper tests, shell syntax, shfmt, and diff checks pass.

This is the one continuation authorized for the original administrative
timeout. It changes no scientific or runtime input and must advance directly
from checkpoint 10912 to step 10913. Any model, data, training, validation, or
evaluator output makes it terminal. No held-out or cross-candidate metric
informed it.
