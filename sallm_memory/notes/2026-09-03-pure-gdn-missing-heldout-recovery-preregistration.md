# Pure-GDN missing held-out recovery preregistration — 3 September 2026

## Authority and reporting status

The user explicitly authorized fixing the five failed current-adapter
evaluation bundles and obtaining their missing results. This is a separately
disclosed recovery evaluation. It does not erase or relabel the original
terminal failures, and its outputs must be reported as recovery results rather
than as the original one-time runs.

No missing-family test metric exists, so no News, T2X, NER, POS, or AfriHG
held-out score can influence this recovery. Already observed Base, SIB, and
Intent test scores must not change the recovery order, adapters, prompts,
decoding, or implementation.

## Frozen scientific inputs

For every family, reuse the exact frozen Base checkpoint, Multi adapter, Mono
adapter or adapters, task definitions, language rows, prompt set, zero-shot
setting, decoding configuration, metric definition, and structural verifier
from its original family freeze. No training, HPO, checkpoint selection,
prompt change, decoding change, or metric change is authorized.

The five recovery bundles are exactly:

- News: Multi and Mono for English and Xhosa.
- T2X: the single Xhosa Mono arm; no duplicate Multi arm.
- NER: Multi and Mono for Tswana, Xhosa, and Zulu.
- POS: Multi and Mono for Tswana, Xhosa, and Zulu.
- AfriHG: Multi and Mono for Xhosa and Zulu.

## Authorized execution-only corrections

Only these already-diagnosed runtime defects may be corrected:

1. provide an isolated writable Hugging Face cache root;
2. materialize and hash the exact offline metric modules and dataset inputs
   required by the frozen task definitions, without model inference;
3. make lm-eval's include-path shim use the same lexical task root that
   lm-eval itself uses when the virtual environment is a symlink;
4. use isolated recovery result and protocol roots so original failed jobs,
   logs, empty roots, manifests, and snapshots remain untouched.

Any different failure stops the affected bundle for review. A recovery bundle
may be restarted only if it fails before model inference and before writing a
prediction or metric. Once an arm writes a prediction or metric, its recovery
outcome is final.

## Frozen locations and execution order

- recovery ID: `pure-gdn-heldout-recovery-20260903-v1`
- cache root:
  `/scratch/lmbanr001/masters/sallm/data/official_recovery_cache/20260903_v1`
- result root:
  `/scratch/lmbanr001/masters/sallm/results/official_test/familywise_recovery_20260903`
- protocol root:
  `/scratch/lmbanr001/masters/sallm/manifests/official_test/familywise_recovery_20260903`

After local regression tests, immutable cache/manifests, metric-free
preflights, and absent-output/no-duplicate checks pass, submit at most four
A100-80GB recovery jobs concurrently. Start T2X, NER, POS, and AfriHG; submit
News when the first slot releases. This order is fixed from expected runtime
and the four-card limit, not from any held-out metric.

Every HEX access begins with `/scratch/slurm/bin/purequota`. Use only
`nlpgroup80/a100/nlpgroup80`, `gpu:ampere80:1`, eight CPUs, and a maximum
48-hour wall time. Do not use L40S or Kombuys for pure-GDN and do not modify
another user's job.

The locally tested implementation is frozen by these SHA-256 values:

- `src/main/sallm/evaluation/lm_eval_runner.py`:
  `72afd04a6eea586c0f488de6427e9e85a6eb4abc3670a0df748d205e12808592`
- `scripts/cache_pure_gdn_official_recovery_assets.py`:
  `088589df69f4f27716a37b0a6fd55ee46bf9215f8a40c6d97bc8034edf56fb42`
- T2X wrapper:
  `616f6d777b8712818249ef23f392a229d2dbf8b9e3aa559581d9e4624aacabf6`
- NER wrapper:
  `fc82c9077ca06a248e9b2272f152d603f4e742454d8026ca397729dcb03331f5`
- POS wrapper:
  `d4a8a4c02b1e85ecb1862943ed05fcbf25ad9cf2e9b2d9094b877419af6c4b45`
- AfriHG wrapper:
  `9f1eedf4e1d6ec91518959d5e2b8e5d4455c8c3c648f3575e0f044de4f478dd0`
- News/Intent wrapper:
  `174a0ff055f9db11cf576ba2ca359f43509ad2d977dc522ee2961def64aafd92`

Focused local verification is `17 passed`, all five wrapper syntax checks
pass, and Ruff is clean for the changed Python files.

## Verification and Sheet rule

Do not open scores until the entire family recovery bundle, exact coverage,
structural sidecars, artifact hashes, and retained frozen inputs verify. Then
record the scores as report-only recovery evidence. Update the Sheet only
after exact API readback, and label the recovery provenance in the research
notes. General remains paused and column G remains blank.
