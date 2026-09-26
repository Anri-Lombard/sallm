# Pure-GDN family-wise execution — 2026-09-01

## Metric-free CUDA gate passed

Job `1279771` completed `0:0` in `00:04:06` on A100-80GB. Result SHA-256 is
`d1f13d836e18b77f2af86db70466b5c19818a52a0e4daad34d2b645b81f1248e`.
The immutable source, runtime, exact base-checkpoint root, save/reload, finite
BF16 forward/backward with complete finite gradients, and deterministic
generation checks all passed. No task or held-out data was used.

## T2X official arm failed closed after payload

T2X official job `1280426` used the frozen b7 seed-42 checkpoint-1932 adapter
transferred byte-for-byte from Kombuys. The destination adapter/config hashes
exactly match the frozen values
`f95725b9e5ba175b46a38ad89c61fcbbd459cac2b63339b9d64afd2fb3ff2bc0`
and `e21938cac6ebfa81faa7855b72f9d0ad7ce9dd892f08412ccd9ea2bdb77235ea`.
All three immutable 715-file source/runtime/base/adapter manifests verified.

The job failed `1:0` after `00:00:33`. It loaded the exact base and adapter,
opened the official T2X test split, and prepared all 378 rows, then failed
before generation because the offline runtime lacked the Hugging Face
`evaluate` BLEU module. No prediction, metric, summary, structural
verification, or Sheet value was produced. The empty output directories and
complete log are preserved. Because the failure occurred after official
payload access, the family-wise amendment requires fail-closed terminality:
never retry, correct, or reopen this T2X official arm. Its reportable outcome
is a missing official result caused by evaluator dependency failure, not a
model score.

## Non-General work occupies released cards

General b1/b2 continuation jobs `1279470/1279471` completed `0:0`; their
terminal artifacts still require read-only verification. General b4
`1279652` remains running and must finish unchanged.

After metric-free dry runs plus absent-output and no-duplicate checks, the
first reduced post-BOS candidates started on A100-80GB in frozen
candidate-major order:

- News a0 seed 42: `1280436`;
- SIB a0 seed 42: `1280437`;
- Intent a0 seed 42: `1280444`.

They use the frozen enhanced registry, a0 recipe, corrected validation-only
macro F1, exact immutable source snapshot, local pure-GDN base, and no
held-out data. The four running owned jobs are General b4 plus these three
classification candidates. Held-out metrics remain absent and Sheet E/F/G
remain blank.

### News a0 pre-payload environment correction

Original News a0 `1280436` failed `1:0` after `00:00:35` during Hydra
interpolation because `PURE_GDN_MODEL` was absent. The immutable manifest and
GDN runtime had verified, but no model or dataset was loaded and no checkpoint
or metric exists. The manifest-only root is preserved. Prospective correction
SHA-256 is
`ced4b6a52f92871aa2689bd4699a02d693d449b5b6fe30a4326b12b0b22117e7`.
Isolated unchanged replacement `1280455` exports that variable to the same
frozen base path and is running. SIB `1280437`, Intent `1280444`, News
replacement `1280455`, and General b4 `1279652` now occupy all four cards.

## General b4 released the next non-General slot

General b4 continuation `1279652` completed `0:0` in `05:28:46`. Its terminal
artifact still needs read-only coverage and retained-to-final verification.
After confirming the a1 output was absent, no matching job existed, the frozen
registry SHA-256 remained
`8fdd6ea5f192a1d2ef2267353162b9a640ccd09ae0d889cb5a999645198bb726`,
and only three owned A100-80GB submissions were active, News a1 seed 42 job
`1280921` was submitted in the frozen candidate-major order. It uses the same
immutable snapshot, base checkpoint, offline runtime, and corrected News
environment binding. News a0/a1, SIB a0, and Intent a0 now occupy all four
cards; held-out state and Sheet E/F/G are unchanged.

## SIB a0 completed and released the next slot

SIB a0 seed-42 job `1280437` completed `0:0` in `02:08:43`. Its retained
checkpoint is step `526`; trainer state records validation-only macro F1
`0.05760368663594471`, and the terminal log covers all `2,970` declared rows.
This remains non-terminal evidence until sidecar and exact retained-to-final
adapter verification finish. The released card was filled with preregistered
SIB a1 seed-42 job `1281116` after absent-output, no-duplicate, immutable-
registry, and active-cap checks. News a0/a1, Intent a0, and SIB a1 now occupy
all four A100-80GB cards.

## Intent a0 completed and released the next slot

Intent a0 seed-42 job `1280444` completed `0:0` in `03:31:07`. Its terminal
log covers the exact `4,155` validation rows and its retained checkpoint is
step `881`, with validation-only macro F1 `0.001135873084394561`. This remains
non-terminal evidence until sidecar and exact retained-to-final verification
finish. The released card was filled with preregistered Intent a1 seed-42 job
`1281334` after absent-output, no-duplicate, immutable-registry, and active-cap
checks. News a0/a1, SIB a1, and Intent a1 now occupy all four A100-80GB cards.

## Both News seed-42 predecessors completed

Corrected News a0 `1280455` completed `0:0` in `03:56:07`, retaining step
`1629` at validation-only macro F1 `0.47238216653156473`. News a1 `1280921`
completed `0:0` in `02:18:22`, retaining step `543` at validation-only macro
F1 `0.2079369214708845`. Both terminal logs cover the exact `3,095`
validation rows and both final adapters exist; sidecar and exact retained-to-
final verification are still pending. Their two released cards were filled in
frozen order by News a2 `1281511` and SIB a2 `1281512` after all absence,
duplicate, immutable-registry, and active-cap checks passed. These jobs run
alongside SIB a1 and Intent a1.

## All reduced seed-42 candidates are now launched

SIB a1 seed-42 job `1281116` completed `0:0` in `02:07:19`, retaining step
`526` at validation-only macro F1 `0.05760368663594471` with exact `2,970`-
row terminal coverage. Sidecar and exact retained-to-final verification remain
pending. The released card was filled with Intent a2 seed-42 job `1281721`
after all frozen preflight checks passed. Every preregistered News/SIB/Intent
seed-42 candidate has now been launched once; active jobs are Intent a1/a2,
News a2, and SIB a2.

## Seven reduced candidates round-trip verified; NER cache-ID correction frozen

News a2 `1281511` and SIB a2 `1281512` completed `0:0`. CPU jobs
`1281966`-`1281972` then proved exact retained-to-final adapter equality for
all seven completed reduced candidates: SIB a0/a1/a2, Intent a0, and News
a0/a1/a2. Each comparison covered `424` keys and `71,762,560` values. Intent
a1/a2 remain the only running reduced seed-42 candidates.

The first NER Mono jobs, Tsn `1281964` and Xho `1281965`, failed `1:0` after
`00:00:43` before any dataset row loaded because the current `datasets` runtime
computed cache IDs absent from the extant exact offline cache. No metric,
checkpoint, or final adapter exists. CPU probes `1281974` and `1281976`
confirmed a config-ID compatibility mismatch rather than absent source data.
The failed jobs, logs, and roots are preserved. A prospective alias-only
correction is frozen in
`2026-09-01-pure-gdn-ner-mono-offline-cache-id-correction.md`; it requires an
exact offline row-count and file-hash probe before isolated replacement jobs.

The three cache aliases were created only after their canonical targets and
absence were verified. CPU probe `1282013` completed `0:0` and loaded the exact
pinned train/validation rows: Tsn `1441/499`, Xho `1441/817`, and Zulu
`1441/836`. Canonical dataset-info and train/validation Arrow hashes remained
unchanged. Isolated Tsn `1282016` and Xho `1282017` replacements then started
on A100-80GB. Their execution-manifest SHA-256 values are
`611d2d37dfbc696ba3f61e2d2478494aba8ad30e2a1b04de99d008fa60dde3ec`
and `700e1a8598b3e533b1fea119c2d3048b77484a3b45802bcae38349579a280efd`.
Both verified all `715` immutable files, loaded/tokenized all `1,441` train
rows and the exact five-template validation expansions (`2,495` Tsn and
`4,085` Xho rows), and entered training. Together with Intent a1/a2, all four
A100-80GB cards are again doing required work.

## Intent a1 verified; all three NER Mono arms active

Intent a1 `1281334` completed `0:0` in `05:51:19`, with exact `4,155`-row
validation coverage and retained checkpoint `2643`. CPU verifier `1282232`
completed `0:0` and proved exact retained-to-final equality across `424` keys
and `71,762,560` values. This makes eight of nine reduced seed-42 candidates
terminal-valid; Intent a2 is the only remaining one.

The released A100-80GB card was filled by isolated NER Zulu Mono job `1282231`
after active-cap, absent-output, alias, canonical-data-hash, snapshot, and
frozen-recipe checks passed. Its execution-manifest SHA-256 is
`39dcb588f8cb3a429100f439127558dee39508869610b2d94621278f87d8f8ec`.
It verified all `715` immutable files and loaded/tokenized the exact `1,441`
train rows plus the `4,180`-row five-template validation expansion. NER Tsn,
Xho, and Zulu now train alongside Intent a2 on all four A100-80GB cards.

## Intent seed-42 grid complete; first POS Mono arm started

Intent a2 `1281721` completed `0:0` in `04:43:53`, with exact `4,155`-row
validation coverage and retained checkpoint `1762`. Its final adapter exists;
metric-free CPU roundtrip verifier `1282671` is submitted once. This completes
all nine reduced News/SIB/Intent seed-42 runs, with eight terminal-valid and
Intent a2 pending only exact retained-to-final equality.

The released A100-80GB card was filled by POS Tswana Mono job `1282672` after
active-cap, absent-output, no-duplicate, immutable-snapshot, base-checkpoint,
and frozen-recipe checks passed. It uses the frozen POS winner learning rate
`1.5e-4`, the fixed Mono LoRA recipe, seed/data-seed 42, the existing
`llama_pos_tsn` Mono configuration, and validation-only checkpoint selection.
Its execution-manifest SHA-256 is
`eaf017e309517cbe499ab1646761e4f85974ae4d9c6f03bd05793dbf2ea90ae0`.
NER Tsn/Xho/Zulu and POS Tswana now occupy all four A100-80GB cards.

## Reduced seed-42 rankings frozen; NER Tswana Mono verified

Intent a2 verifier `1282671` completed `0:0` and proved exact
retained-to-final equality across `424` keys and `71,762,560` values. All nine
News/SIB/Intent seed-42 candidates are now terminal-valid. CPU ranking job
`1282779` applied the frozen validation-only protocol and wrote three hashed
artifacts:

- News: a0 then a1 then a2, SHA-256
  `ffb124a7c4aeef576ebcd11d0f97e3b727c60405f629c70c9bce1e4185bca048`;
- SIB: a0 then a1 then a2 after the preregistered lower-LR tie-break,
  SHA-256
  `c9ef792c306bae3b8664d64cbbb1bcdef0f1444fc1b328b49dc251bc5187a49e`;
- Intent: a2 then a1 then a0, SHA-256
  `b63f78beffe4f800103d6b25b1ca92139ba08e886e2878d52e276e944ea63b36`.

The frozen seed-13 finalists are News a0/a1, SIB a0/a1, and Intent a2/a1.
No held-out metric informed these rankings.

NER Tswana Mono job `1282016` completed `0:0` in `04:42:16`. It stopped under
the fixed early-stopping rule at step `2172`, where validation-only F1 was
`0.5834360027378009`. Verifier `1282784` proved exact retained-to-final
equality across `424` keys and `71,762,560` values. NER Xhosa `1282017` and
Zulu `1282231` remain healthy.

## POS Mono evaluator scope corrected prospectively

Initial POS Tswana job `1282672` validated all `750` declared rows, then
failed `1:0` because the evaluator demanded the three-language four-prompt
Multilingual grid and rejected the declared Tswana fifth prompt. It produced
no checkpoint, final adapter, or selection artifact. Its root and log are
preserved, and its validation output is excluded.

The prospective correction is frozen in
`2026-09-01-pure-gdn-pos-mono-validation-scope-correction.md`, SHA-256
`9a9336bb934286c17523d6fc001820f101473b9c6ea2bf5e8431e7cb865584e4`.
The evaluator now enforces the exact language by template grid declared by
the validation dataset. The focused test suite passes `8/8`, and Ruff plus
formatting checks pass. Immutable corrected snapshot
`pure-gdn-pos-mono-scope-correction-20260901-bd406341` verified all `715`
source and config files. Its deployment manifest SHA-256 is
`439f6bb902dd6fe605c3d565d4170946663fa3b289c2dc2b913a9301fa636530`.

Isolated corrected Tswana `1282788` and first-run Xhosa `1282789` were
submitted after absent-output, no-duplicate, source-hash, active-cap, and
immutable-recipe checks. Both started on A100-80GB with the unchanged frozen
POS recipe. They run alongside NER Xhosa and Zulu, so all four cards are again
occupied with required non-General work.

## NER Xhosa verified; POS Zulu filled the released card

NER Xhosa Mono `1282017` completed `0:0` in `06:01:32`. Its retained
checkpoint is step `1991`, with validation-only F1 `0.53664556282858`.
Metric-free CPU verifier `1283303` completed `0:0` and proved exact
retained-to-final equality across `424` keys and `71,762,560` values. The
retained and final weight-file SHA-256 values are
`10da37a2b861e9ee5e4b658ed4e7f42efffa26f68c8d8bfe26a7ab4dc761e200`
and `2fbf8ba2c1b5db57857900f7a1564322d935902e717318e25e4e236d52cf05ad`.

The released card was filled immediately by first-run POS Zulu Mono job
`1283304` after the frozen absent-output, no-duplicate, immutable snapshot and
source-hash, base, recipe, GPU-family, and active-cap checks all passed. The
metric-free dry run confirmed the unchanged POS recipe and isolated output.
Its execution-manifest SHA-256 is
`b31ac58a1505686a5e4b658ed4e7f42efffa26f68c8d8bfe26a7ab4dc761e200`.
It verified the immutable snapshot, loaded the exact `753/750` train/eval
rows, and entered the fixed `1,425`-step training loop. It runs with corrected
POS Tswana `1282788`, POS Xhosa `1282789`, and NER Zulu `1282231`, restoring
four-of-four A100-80GB use. Held-out and Sheet E/F/G state are unchanged.

## NER Mono family frozen; AfriHG Xhosa fills the released card

NER Zulu Mono `1282231` completed `0:0` in `05:01:39`, retaining checkpoint
`1629` at validation-only F1 `0.5095029239765582`. Metric-free CPU verifier
`1283371` completed `0:0` and proved exact retained-to-final equality across
`424` keys and `71,762,560` values. NER Tswana, Xhosa, and Zulu are therefore
all terminal-valid and frozen before official testing.

The first AfriHG Xhosa preflight attempted the launcher's nominal dry-run on
the login node. It wrote only an execution manifest and stopped before model
or data loading because `nvidia-smi` is unavailable there. That two-file root
is preserved as `seed_42_failed_login_dryrun_20260901` and excluded.

Canonical AfriHG Xhosa Mono `1283372` was then submitted once after the output
was clean, duplicate and active-cap checks passed, and the transferred family
freeze artifact matched SHA-256
`3cc31f0b2221bf30f1c795aa58420012313a56976f0d06b23ac49836a690f5db`.
It uses the frozen winner LR `0.0001223079850011719`, fixed rank `16`, alpha
`32`, dropout `0.05`, warmup `0.03`, and seed/data-seed `42`. Its execution
manifest SHA-256 is
`0e3e4ce929636d6a5317f2c3027adb9cb69f3b6dbc4187304b4fcbfd913f731a`.
The job verified all `715` immutable files, proved the GatedDeltaNet fast path,
loaded exact `10,440/1,305` train/validation rows, and entered the unchanged
6,525-step training loop on A100-80GB. POS Tsn/Xho/Zul and AfriHG Xho use all
four cards. The official NER Multi plus three Mono arms are the next family
release when a card becomes eligible; AfriHG Zulu follows. No held-out
artifact or Sheet E/F/G cell changed.

## NER official verifier corrected before held-out access

Independent review caught that the first metric-free NER preflight paired an
lm-eval task pack with the T2X generation-artifact verifier and incorrectly
declared Tswana as `5,000` rows. No held-out row had been loaded. The initial
snapshot, jobs `1283916/1283917`, and hashes are preserved as superseded
preparation.

The prospective task-pack correction is frozen at SHA-256
`537c4d18d0e2228c84135fd713902734275bed62b00d0b4629a1e70e1f2ebfc7`.
Its verifier checks exact `_test` tasks, task-pack artifact agreement, finite
per-prompt F1, and exact `4,980/5,000/5,000` Tswana/Xhosa/Zulu coverage while
recording no metric values. Focused tests pass `5/5`, and it also verifies the
three already finalized Base official artifacts structurally. Corrected
manifest job `1283922` and metric-free preflight `1283923` completed `0:0`
against immutable snapshot `pure-gdn-ner-official-test-20260901-v3-731102b2`.
The six-arm NER bundle remains next for a released A100-80GB card. Held-out
access and Sheet E/F/G remain unchanged.

## POS Zulu completed; official NER queued once

At 00:30 SAST on 2 September, POS Zulu Mono `1283304` had completed `0:0`
after `07:20:50`, covered all `750` declared validation rows at its terminal
boundary, and wrote its final adapter. It still needs the metric-free sidecar
and exact retained-to-final roundtrip before the POS family can freeze.

The released owned submission slot was filled once by the frozen corrected
official NER family bundle as job `1285944`. Slurm readback confirms
`nlpgroup80/a100/nlpgroup80`, `gpu:ampere80:1`, eight CPUs, a 24-hour limit,
and the required checkout. It is `AssocGrpGRES`-pending because another user
holds the fourth association card; POS Tswana `1282788`, POS Xhosa `1282789`,
and AfriHG Xhosa `1283372` remain running. No NER held-out row has yet been
loaded, and Sheet E/F/G remain blank.

## NER closed missing; reduced confirmations started

Official NER replacement `1285997` passed every frozen manifest and loaded the
model plus adapter, then failed `1:0` before loading any held-out row because
the offline runtime could not resolve the first dataset. Its result root has
zero files or metrics. The post-payload rule makes NER terminally missing; no
retry is authorized.

The released card now runs frozen News a0 seed-13 confirmation `1286000`.
Execution-manifest SHA-256 is
`a23ed55e921c78cd640cee0ba9b65ffea31fc4d943ce9e95054c85405d403ede`;
its first fixed boundary covered all `3,095` validation rows. POS Tswana
`1282788` is still non-terminal at step `760/1,425` inside another fixed
`750`-row boundary, rather than the previously assumed final boundary.
AfriHG Zulu `1285972` is healthy at step `1,033/8,885`. No interim value has
changed scheduling, and Sheet E/F/G remain blank.

At 03:31 SAST, the fourth owned submission slot was filled by the next frozen
candidate-major confirmation, SIB a0 seed 13 job `1286171`. Its output and
matching job were absent, the immutable registry hash matched, and the owned
active cap passed. It is `AssocGrpGRES`-pending behind the three running owned
jobs and shared association use. No held-out or Sheet state changed.

## POS frozen and official bundle queued

POS Tswana `1282788` completed `0:0` with exact `750`-row validation coverage.
CPU verifier `1286172` proved retained checkpoint `570` exactly equals the
final adapter across `424` keys and `71,762,560` values. POS Mono is therefore
`3/3` terminal-valid and frozen alongside the Stage-A a2 Multi winner.

Independent review blocked the first POS official preparation before held-out
access because the shared verifier still assumed NER task IDs/F1 and the POS
task YAMLs used mutable `raw/main` URLs. The minimal correction parameterizes
task suffix and metric, tests the real POS no-suffix/token-accuracy artifact
shape, and pins all twelve task files to MasakhaPOS revision
`376f4161f0425584d4bd7664122b56fa026926d3`. Nine focused tests, Ruff, and
wrapper syntax checks pass; final re-review found no substantial blocker.

Immutable snapshot `pure-gdn-pos-official-test-20260902-v1-63f90569` and
manifest/preflight jobs `1286173/1286174` completed cleanly. The preflight
verified `718` source/config files per manifest, all exact artifacts and
runtime, and six resolved configs while leaving the official result root
absent. One-time six-arm POS official job `1286189` is now
`AssocGrpGRES`-pending as the fourth owned submission. News `1286000`, SIB
`1286171`, and AfriHG Zulu `1285972` are running. No current-adapter POS
held-out value has been opened and Sheet E/F/G remain blank.

## News/SIB confirmations verified; POS pre-payload correction queued

News a0 seed-13 job `1286000` and SIB a0 seed-13 job `1286171`
completed `0:0`. CPU verifiers `1286716/1286717` proved exact
retained-to-final equality: News retained step `1629`; SIB retained step
`526`. These are validation-only robustness confirmations. News and SIB a1
seed-13 jobs `1286721/1286722` are now running on A100-80GB. Preserve the
zero-second News launch `1286718`, whose comma-delimited Slurm export was
parsed incorrectly and produced no output.

Initial POS official job `1286189` failed after three seconds before payload
because execution tried to overwrite a read-only resolved config created by
preflight. The result root stayed absent: no model, data, evaluator,
prediction, or held-out metric was opened. The minimal TDD correction makes
preflight create and hash each config and makes execution verify rather than
rewrite it. Fresh isolated snapshot
`pure-gdn-pos-official-test-20260902-v3-41626421`, manifest `1286723`, and
metric-free preflight `1286724` completed `0:0`; independent review found no
blocker. Replacement `1286725` was submitted once and is association-cap
pending as the fourth owned slot. AfriHG Zulu `1285972` continues running.
Sheet E/F/G remain blank.

## News/SIB frozen; POS terminally missing; Intent and News Mono active

News a1 seed-13 `1286721` and SIB a1 seed-13 `1286722` completed `0:0`
with exact `3,095`/`2,970`-row terminal validation coverage. CPU verifiers
`1287267/1287268` proved retained-to-final equality across `424` keys and
`71,762,560` values for each adapter, at retained steps `543/526`.

The reduced two-seed ranking seam now explicitly requires seeds `42` and
`13`; five focused tests and Ruff pass. Immutable ranker snapshot
`pure-gdn-reduced-confirm-ranking-20260902-358d54fe` has script SHA-256
`358d54fefdf4db60b6d041b28ca96ea07dca81ec1a5fbf51b0393668aba31147`.
CPU job `1287273` froze News to a0 under ranking SHA-256
`fd554517a4f1cbb81416e9522f3146ca0fe7b7944185a77f9b74380a748829a2`
and SIB to a0 under ranking SHA-256
`24b3d47c2575a39d8c9c5378df4bdfec2f8c02c892e04104831e8e2fc4d9f346`.
News wins by the two-seed validation mean; SIB's exact mean tie resolves to a0
under the preregistered lower-learning-rate tie-break. No held-out evidence
entered either decision.

POS replacement `1286725` passed all six immutable manifest/config checks and
entered the first Multi Tswana payload, then failed `1:0` after `42` seconds
while creating the Hugging Face dataset-cache root because
`/home/lmbanr001/.cache/huggingface` did not exist. The POS result root has no
files: no prediction, row, summary, or metric artifact exists. Because the
one-time replacement entered payload, POS official is terminally missing and
must never be retried or corrected.

Released slots now run frozen Intent finalists a2/a1 at seed 13 as jobs
`1287271/1287272`. First News Mono arm, English job `1287274`, is
association-cap pending with the frozen a0 family LR `3e-5` and fixed Mono
recipe. AfriHG Zulu `1285972` remains running. These are the four active owned
A100-80GB submissions. Sheet E/F/G remain blank.

## AfriHG Mono frozen and official bundle queued

AfriHG Zulu Mono `1285972` completed `0:0` at retained checkpoint `7108`.
CPU verifier `1287725` proved exact retained-to-final equality across `424`
keys and `71,762,560` values, freezing AfriHG Mono `2/2`.

The prospective AfriHG official freeze uses the exact local Xhosa/Zulu cache
and adds only Multi and Mono arms because accepted Base results already exist.
CPU job `1287732` wrote manifests but failed before wrapper execution because
the read-only wrapper was called directly. Corrected preflight `1287733`
invoked the unchanged wrapper through `bash`, verified all six manifests and
four config sidecars, and completed `0:0`; the official result root remained
absent. One-time four-arm job `1287734` is association-cap pending as the
fourth owned A100-80GB submission. Held-out metrics remain unopened and Sheet
E/F/G remain blank.

## AfriHG official terminally missing; Intent a2 exact; SIB Mono starts

AfriHG official job `1287734` verified the six manifests and four resolved
configs, loaded the base and frozen Multi adapter, and entered the first Xhosa
test task. It then failed `1:0` after `28` seconds when `datasets` attempted to
create `/home/lmbanr001/.cache/huggingface`. The result root contains only
directories and zero files: no prediction, summary, structural verification,
or metric exists. The bundle is terminal under its one-time rule and must
never be retried; AfriHG current-adapter official results remain missing.

Intent a2 seed-13 confirmation `1287271` completed `0:0` with all `4,155`
validation rows. CPU verifier `1288158` proved exact retained-step `1762` to
final equality across `424` keys and `71,762,560` values. Intent a1 remains
running. The two released A100-80GB slots were filled in frozen order by SIB
Afrikaans Mono `1288159` and SIB English Mono `1288160`; News English Mono
`1287274` continues. Four owned submissions are active and Sheet E/F/G remain
blank.

## Intent frozen; Mono reaches 12/21

Intent a1 confirmation `1287272`, News English Mono `1287274`, and SIB
Afrikaans/English Mono `1288159/1288160` completed `0:0` with complete fixed
validation coverage. CPU verifiers `1288213`-`1288216` each proved exact
retained-to-final equality across `424` keys and `71,762,560` values.

The tested reduced-confirm ranker then used exactly seeds `42` and `13` for
Intent a1/a2. CPU job `1288218` completed `0:0` and froze a1 at family LR
`8e-5`; ranking SHA-256 is
`5151121aa34b5f7299f6aa830c10b70d37c821a7308eb3e10813123b7b01e89f`.
No held-out evidence entered the decision.

Including the already-frozen T2X arm, `12/21` Mono checkpoints are frozen.
Released cards now run News Xhosa `1288219`, SIB Northern Sotho `1288220`,
SIB Southern Sotho `1288221`, and queue Intent English `1288222`. All use
their frozen family LR and fixed Mono recipe. Sheet E/F/G remain blank.

## Mono reaches 16/21

News Xhosa `1288219`, SIB Northern/Southern Sotho `1288220/1288221`, and
Intent English `1288222` completed `0:0` with fixed validation coverage.
CPU verifiers `1288931`-`1288934` each proved exact retained-to-final equality
across `424` keys and `71,762,560` values. News Mono is frozen `2/2`, SIB is
`4/6`, Intent is `1/4`, and the full Mono program is `16/21` frozen.

Released cards now run SIB Xhosa `1288936`, SIB Zulu `1288937`, Intent Xhosa
`1288938`, and queue Intent Zulu `1288939`. Intent Southern Sotho is the sole
unsubmitted Mono arm. Held-out and Sheet E/F/G remain unchanged.

## SIB Mono frozen; final Intent arm submitted

SIB Xhosa/Zulu `1288936/1288937` completed `0:0`, both retaining checkpoint
`88`. Intent Xhosa `1288938` completed `0:0`, retaining checkpoint `250`.
CPU verifiers `1288988`-`1288990` completed `0:0` and proved exact
retained-to-final equality for every adapter across `424` keys and
`71,762,560` values. SIB Mono is therefore frozen `6/6`; Intent is `2/4`, and
the full Mono program is `19/21` frozen.

After the frozen absence, duplicate, config, active-cap, and immutable-recipe
checks passed, the final unsubmitted arm, Intent Southern Sotho `1288991`, was
submitted once at LR `8e-5`. It runs beside Intent Zulu `1288939` on
A100-80GB. No held-out metric was opened and Sheet E/F/G remain blank. SIB is
now eligible for its frozen one-time official bundle.

## Mono reaches 20/21

Intent Zulu `1288939` completed `0:0`, retaining checkpoint `750`. CPU
verifier `1289009` completed `0:0` and proved exact retained-to-final equality
across `424` keys and `71,762,560` values. Intent is frozen `3/4` and total
Mono progress is `20/21`. Intent Southern Sotho `1288991` is the sole
remaining Mono job and is running on A100-80GB. Held-out metrics and Sheet
E/F/G remain unchanged.

## Mono complete; SIB official bundle launched

Intent Southern Sotho `1288991` completed `0:0`, retaining checkpoint `250`.
CPU verifier `1289188` completed `0:0` and proved exact retained-to-final
equality across `424` keys and `71,762,560` values. All `21/21` Monolingual
adapters are now frozen before official testing.

SIB cache job `1289189` materialized the six fixed test configurations with
exactly `204` rows each and recorded `metrics_computed=false`. The first
metric-free manifest and preflight chain `1289204/1289205` passed, but an
independent review found that its immutable snapshot omitted the structural
verifier. No held-out inference had started and the result root remained
absent. The preserved pre-payload chain was superseded under the correction
note with SHA-256
`1620c6559b0b937ef1a51862664b0801cbbd19fae184404f060bf2bff2e7e854`.

The corrected snapshot contains the verifier, checks a full cache-tree SHA-256
inventory, and requires sealed resolved configs. Cache sealing job `1289206`,
manifest job `1289207`, and metric-free preflight `1289208` completed `0:0`.
All twelve configs and sidecars were set read-only before one-time SIB
Mono/Multi official job `1289209` was submitted on A100-80GB. Metrics remain
uninspected and Sheet E/F/G remain blank.

## SIB pre-payload kernel correction launched

SIB job `1289209` failed `1:0` after two seconds during runtime-manifest
verification. The sealed cache tree had verified, but the `v2` snapshot used
an older verifier that compared raw Linux kernel build strings from the CPU
manifest node and A100-80GB node. The official result root remained absent:
no model, dataset, prediction, summary, structural verification, or metric
artifact exists.

The final prospective pre-payload correction is frozen under note SHA-256
`4c6fd9934109174f9da8fad2445de20e1f6d86ab68065ebf35eec9c88710b93f`.
It reuses the already-tested public-seam platform normalizer without changing
any scientific input. Fresh immutable manifest and metric-free preflight jobs
`1289210/1289211` completed `0:0`, all twelve resolved configs were sealed,
and the result root was still absent. Single replacement SIB Mono/Multi job
`1289212` was submitted once on A100-80GB. Official metrics remain uninspected
and Sheet E/F/G remain blank.

## SIB official bundle verified

Replacement SIB job `1289212` completed `0:0` in `00:18:42`. All twelve
Mono/Multi language arms produced their fixed five-prompt, `1,020`-row
artifacts, and all twelve structural-verification sidecars passed. The sealed
48-file result-tree manifest SHA-256 is
`ad98b6c0bbf2c03d101fbc79232654d4bfdb125abb317840290a18fc227136d9`.
Only after this complete-bundle verification were the report-only held-out
scores opened. The six-language macro F1 is `0.1367692769759137` for Multi
and `0.07555764695434825` for Mono. These values cannot influence News,
Intent, or any other action. Sheet E/F/G remain blank pending the complete
obtainable non-General set and exact readback.

## News terminal runtime failure

One-time News job `1290454` opened the first English held-out task and then
failed `1:0` in the frozen `lm-eval` symlink-path `relative_to` logic before
any example, prediction, result, structural sidecar, or metric artifact was
written. The result root contains zero files. Because the held-out payload was
opened, the attempt is terminal under the frozen one-time rule. News remains
missing and must not be retried under the original official protocol. The
failure record is
`sallm_memory/notes/2026-09-03-pure-gdn-news-official-terminal-runtime-failure.md`
with SHA-256
`6cff6ecd8fe03cf6a19880d057f769bf2e69cc95982d7d27bbb4e2aa74da97ee`.

## Intent official bundle verified

Initial Intent job `1290452` failed before model, data, task, evaluator, or
metric loading because the prepared protocol configs remained writable after
an earlier setup command aborted. Its result root stayed empty. The single
prospective sealing-only correction is recorded under note SHA-256
`63c3e13a96485018d14f817f404c218f6e67c5ba7be6d7ffa80a6016018d01e5`.
Corrected manifest `1290455`, preflight `1290456`, seal `1290457`, and
official job `1290458` all completed `0:0`.

All eight fixed Mono/Multi language arms produced exactly 32 sealed files,
five prompts per arm, and the preregistered expanded-row counts: `3,110` for
English and `3,200` each for Xhosa, Zulu, and Southern Sotho. Every structural
sidecar reports `verified=true` and `metric_values_included=false`. Result-tree
manifest SHA-256 is
`44a89fde7b63361e586a2135e1dc46a756032a6fad9b8613ac55174832534576`.
Only after complete verification were report-only scores opened. Four-language
macro held-out F1 is `0.002614876682742` for Mono and
`0.001161031235439` for Multi.

## Obtainable non-General close-out and Sheet readback

Base remains accepted `16/16`. The only terminally obtainable current-adapter
Mono/Multi bundles are SIB and Intent, covering `10/21` Mono rows and `10/20`
Multi rows. News, T2X, NER, POS, and AfriHG remain terminally missing under
the original one-time protocol.

The Google Sheet write to `GDN Results!C10:G19` was read back through the
Sheets API. All ten dates format as `3 Sep`; every SIB and Intent Mono/Multi
cell exactly matches the sealed report artifact; Base column D is unchanged;
text wrapping and date formats are intact; and General column G remains blank
for all ten rows. No held-out score informed a run, retry, checkpoint,
selection, prompt, or correction.
