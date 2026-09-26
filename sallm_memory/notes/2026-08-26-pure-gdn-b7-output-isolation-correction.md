# Pure-GDN b7 seed-87 output-isolation correction, 2026-08-26

Preregistered at 16:45 SAST after job `1271040` startup and before any
validation metric or result was produced or inspected.

- Job `1271040` was intended to restart cancelled b7 seed-87 in an isolated
  output directory. The 11 August immutable wrapper predates output overrides,
  ignored the exported override, and reopened the canonical partial directory.
- Startup rewrote only the canonical `execution_manifest.json` and sidecar
  before `1271040` was cancelled after about 39 seconds. Original job
  `1270629`'s manifest hash remains recorded as
  `8888487e2d650175a44ca2e2eb52d0a31aab323f2af942ffa46ba93f8ccbf584`;
  `1271040` wrote replacement hash
  `d806c8497dd28f1b2eb4e07a79a98535562f47c8878959313452254cf7b01ea7`.
  Preserve and quarantine the entire canonical b7 seed-87 partial directory;
  do not select, resume, or overwrite it again.
- The original and current wrappers differ in exactly three lines: the current
  wrapper allows explicit run ID, output directory, and logging directory
  overrides. Freeze that execution-only overlay at SHA-256
  `38aba8e35f74efc3827769aeb261759b6751d73eb8d4575e75b7c2a0f99e52a1`.
  It still resolves every scientific setting and launches every downstream
  script from unchanged immutable HPO snapshot
  `uniform-adapter-hpo-20260811-6aabf717`.
- After verifying the isolated output and logging roots are absent and no
  duplicate is active, submit b7 seed-87 once on A100-80GB through this overlay.
  Keep candidate, seed, registry, model, data, evaluator, training protocol,
  resources, and selection rules unchanged. No held-out data may be accessed.

## Execution

- The correction note and overlay were deployed read-only and reverified at
  SHA-256 `fd8e53fa2be58216a52a6b9564b3890baa8e896d38c32e1299936da623cc698f`
  and `38aba8e35f74efc3827769aeb261759b6751d73eb8d4575e75b7c2a0f99e52a1`.
- Corrected b7 seed-87 is job `1271354`. It started on A100-80GB at about
  `16:44 SAST` and wrote only to the isolated path, with execution-manifest
  SHA-256 `2b2738fa21f59dbf8cb82f559bb458ee1372b0a933291957fc6d926290ce8b40`.
