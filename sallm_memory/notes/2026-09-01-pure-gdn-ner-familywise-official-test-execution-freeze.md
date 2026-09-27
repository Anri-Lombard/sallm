# Pure-GDN NER family-wise official-test execution freeze

Frozen prospectively at `2026-09-01T16:45:27Z`, after all NER Multi and Mono
adapters were validation-frozen and before any current-adapter NER official
held-out access.

The one-time family bundle contains exactly six report-only arms: the frozen
Multilingual NER adapter evaluated on Tswana, Xhosa, and Zulu, followed by the
three language-specific Monolingual adapters. Each arm uses its unchanged
MasakhaNER config and must cover exactly `5,000` evaluation rows. No result may
select, correct, retry, prompt, decode, schedule, or otherwise affect training.

The immutable execution snapshot is
`/home/lmbanr001/masters/sallm_snapshots/pure-gdn-ner-official-test-20260901-v1-3ba8dd4b`.
The wrapper SHA-256 is
`3ba8dd4b12281a0628d844479469044043d06624a6ab93584d8986dd8b8c51a1`;
the reused structural verifier SHA-256 is
`daf6d8cf004441cdf7d56a232d83b6e193b3ccbbacf93c88202541c0166985d9`.
Manifest-freeze job `1283916` completed `0:0` and wrote these immutable
manifests:

- source: `c4065ddd16c82a9b905ff88c70aab062ca9b4b04606283e9d1199019340cf7f2`
- base: `179cece038b83822ce21f76040de7135d7d12c4aa327d3d22fbca6be7011cc54`
- Multi NER: `28a76d4a72ec09884bcdb163b9d92665d5fef230ccd8b01a89172dada063e94a`
- Mono Tswana: `d58310b1a5cf17d7e9d0de4cb6d4d406cd8686b18bdea1bb374976f4984e7b0a`
- Mono Xhosa: `e34928381b26255db85e848df8910368e41b79e18f65c8f15edebf3151b3a8de`
- Mono Zulu: `7912ee37263f68d9cb45ef34d1a27f8ab62d98299f071f48cd7a1b1e4b3eee66`

The base checkpoint is
`/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model`.
The frozen Multi adapter is
`/scratch/lmbanr001/masters/sallm/checkpoints/adapter_hpo_v3/pure_gdn/ner/stage_b/b7/seed_42/final_adapter`,
with adapter/config SHA-256 values
`06b4685fcdb43089ea2de5ad4be1f476b9397cbd9cc57467f291534b1a5309f5`
and `e823b4136c4b02e2a0287056c687245ff53c1dd6d7c315181c9acd5b7937ca4e`.
The frozen Mono final adapters are under
`/scratch/lmbanr001/masters/sallm/checkpoints/pure_gdn_mono_familywise_v1/ner`:

- Tswana `tsn/seed_42_cacheid_correction_20260901/final_adapter`, adapter/config
  hashes `7631a299aff94b615edfa36601bf5335cf71d7d95d5b19b7cd0489f7b371acb4`
  and `03946008dd8656eaaea6bc5610ca6183608ff2ceca67120fa5b7134c670596d1`
- Xhosa `xho/seed_42_cacheid_correction_20260901/final_adapter`, adapter/config
  hashes `2fbf8ba2c1b5db57857900f7a1564322d935902e717318e25e4e236d52cf05ad`
  and `886706f2d864be8f325fbf08f3a5203614c955df84daa47049ca31189d69d24e`
- Zulu `zul/seed_42_cacheid_correction_20260901/final_adapter`, adapter/config
  hashes `a0f2ed4226ab372d3a9a0b86ba3fe800e2eb82c778d034324adbea462f3d2e42`
  and `3b356376d0c5bf7449dd66733cd046a6e85b870bfc6be70c724430fd2a5384dd`

Metric-free preflight job `1283917` completed `0:0`. It verified all immutable
manifests and resolved the six configs without loading held-out rows. The
resolved-config SHA-256 values, in execution order, are:

- Multi Tswana: `e41d2bcb328f339e18d95690dcb2fba7f33690a10aab2a93fccada90d7e43b45`
- Multi Xhosa: `aaf83cde2e745e447bfb1c600c420cc9b346923fad58e953a74525c4ec1fa618`
- Multi Zulu: `ec07ab2cec3a504f5dba61f2bc7c7b3f1eb404ff5d0eb20efcdd95dd22c72d51`
- Mono Tswana: `0b69093e65fcc7b3dc51224df123f4cae55004ea6b5d9d02d450633ce0c5dd17`
- Mono Xhosa: `6fef257a88414bdb92acd9f180781a4386872b4ee7d9bddb7254878b5466d79e`
- Mono Zulu: `db788e1665a7c6e4d3901da3987201429139764ea3a7c9cdb262b7fecdd7d3a1`

The output root is fixed to
`/scratch/lmbanr001/masters/sallm/results/official_test/familywise_20260901/ner`.
The wrapper fails closed if any output already exists, verifies every immutable
manifest before task access, and structurally verifies exact task, language,
row coverage, finite metrics, and artifact agreement after each arm. Any
post-payload failure is terminal for the affected arm; it is not retried.
Sheet E/F/G remain blank until all official artifacts and exact readback verify.
