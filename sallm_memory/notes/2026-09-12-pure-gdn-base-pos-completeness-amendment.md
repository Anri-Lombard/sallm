# Pure-GDN Base POS completeness amendment — 12 September 2026

The corrected Base matrix was 15/16 because job `1326145` failed while
resolving the pinned MasakhaPOS Tswana training URL. Its preserved output has
zero files: no test row, prediction, response, or metric was produced or
inspected. The job did initialize the model, so the original one-shot rule
correctly marked the lane terminal.

The user explicitly requested a complete GDN comparison on 12 September. This
authorizes a post-hoc completeness amendment, not a rewrite of the original
protocol. It reuses the exact sealed Base checkpoint, task definitions, raw
prompting, batch size 1, decoding, metric, aggregation, and 7,216-row official
task pack. The only execution changes are a cache proven by the completed
General POS run and fresh runtime/output paths. Any resulting values must be
labelled post-hoc completeness results and must not be used for recipe or
checkpoint selection.

CPU seal `1334558` completed `0:0` in 24 seconds. It verified the old binding,
empty failed output, failed-log SHA-256, identical Base/General POS task files,
the metric-free 7,216-row General structural sidecar, and exact resolved-config
equality after changing only the output path. New binding:
`e468d523db1afa608a275ca867051dbaab1258f1ef501107db2a038271a9483b`.

The single A100-40GB `gpu:ampere` job `1334564` was submitted with a 48-hour
limit but never started because all HEX A100 and L40S nodes were drained for
patching. On 13 September the user explicitly authorized moving this missing
lane to Kombuys. Job `1334564` was held as `JobHeldUser` while still
pre-payload: its claim and output remain absent.

The exact checkpoint already present on Kombuys matched all six HEX file
hashes. The 3.3 MB sealed source snapshot, original protocol references and
157 MB offline cache were transferred exactly. The evaluation-critical
packages matched except `datasets`; a private runtime overlay pins the HEX
version `4.8.5` without changing the shared environment. Model/BF16 loading
passed on the idle RTX 3080 Ti using 257 MB. The resolved config is identical
after replacing only checkpoint, task include and output path prefixes.

Kombuys binding `a0f0be1b09b82de8a8d8998d15bc1ba7bed70863646480d25a6ba3e4f769562e`
covers 1,970 source, checkpoint, cache, environment and protocol files. Tmux
session `gdn-base-pos-completeness` began at `2026-09-13T08:37:56Z` on GPU1.
This adds a disclosed RTX 3080 Ti cross-host execution amendment; it does not
change the scientific recipe. Do not inspect metric values until the run
finishes and its structural sidecar verifies `12` tasks, `7,216` rows, and
`metric_values_included=false`.
