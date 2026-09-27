# Repository Cleanup Audit, September 2026

Audit of `main` at `94815d5` (30 August 2026) and of the pending paper branch
`paper/architecture-comparison-2026-09` (PR #135, draft, tip `d55520e`), done
on 27 September 2026. The paper branch had not merged, so the cleanup branch
starts from `main` and avoids the files that PR #135 changes wherever possible.

Evidence is from `git ls-files`, `rg` for references, `git log` recency, a
Hydra compose of every entry config, and the test and pre-commit runs. Items
marked *inference* are judgements, not checked facts.

## Summary

| | main | paper branch adds |
| --- | --- | --- |
| Tracked files | 637 | +717 new, 62 modified |
| `src/conf` YAML | 513 | +98 new, about 36 modified |
| Tests | 12 files, 89 tests | +51 files |
| Notes (`sallm_memory/`) | none | 304 files, 7.4 MB |
| Paper runners (`scripts/paper/`) | none | 176 files, 1.6 MB |

PR #135 already conflicts with `main` in 16 files (`git merge-tree`: the
`ops/slurm` launchers, `pyproject.toml`, `uv.lock`, most evaluation modules,
and two files that main deleted but the paper branch modified). Merging the
paper branch into this cleanup branch gives exactly the same 16 conflicts, so
the cleanup adds none.

Main is already fairly lean. Most of the sprawl the cleanup was meant to
address (dated notes, probes, canaries, duplicate runners) arrives with PR #135.

## Applied on `cleanup/2026-09`

| Change | Evidence |
| --- | --- |
| Deleted `src/conf/finetune/factory.py` | Older 76-line copy of `build_tokenizer`/`build_model` from `src/main/sallm/models/factory.py`; no imports or references anywhere; last touched 2025-09-18. Hydra does not load `.py` files. |
| Deleted `src/main/sallm/evaluation/model_loader.py` | Empty file (0 bytes), never imported. |
| Deleted `poetry.toml`, `.yamlfmt`, `.grype-ignore.yml` | The project uses uv and hatchling; no yamlfmt hook exists; CI and tests read `.grype.yaml` only. No references to any of them. |
| Removed `[tool.yamllint]` from `pyproject.toml` | yamllint reads only `.yamllint`, which already holds the rules, so this table was a divergent duplicate. |
| Replaced `.gitignore` | The file on main listed every rule twice. The replacement is PR #135's grouped version, byte-for-byte, so the merge applies cleanly. |
| Moved `req.txt`, `environment.yml` to `archive/envs/` | A 2025 conda environment (`sallm-ner`) and its `pip freeze`, with `file:///home/conda/...` paths. Superseded by `uv.lock`; not referenced. |
| Moved `scripts/get_wandb_sweep_info.py`, `scripts/generate_xlstm_configs.py` to `archive/scripts/` | One-off tools with no references; the generator already produced the `xlstm_*` finetune configs. `repo_root` in the generator was adjusted for its new depth. |
| Moved `docs/hpo_mamba_sweeps_last_14_days.md`, `docs/masakhaner-x-issue.md` to `archive/docs/` | A generated report dated 2026-02-03 and an upstream request draft. They were already excluded from markdownlint. |
| Fixed `tokenizer/train.py` and `tokenizer/process.py` defaults | Both defaulted to a `configs/` directory that no longer exists. `train.py` also read the keys `train_data_file`/`output_file`, but `bpe.yaml` defines `train_data_dir`/`output_dir`, so the script raised a `KeyError` with the shipped config. |
| Rewrote the README layout and run sections | It now covers corpus preparation, tokenizer training, pretraining, fine-tuning, evaluation and HPO, and gives a layout table that matches the tree. |
| Added `requirements-paper-2026-09.lock` | Exact package versions of the local paper venv (transformers 4.57.3). |
| Added `docs/DEPENDENCY_UPGRADE_ASSESSMENT.md` | Assessment only; no versions changed. |

## Ranked findings

### Delete (done, or safe once PR #135 lands)

1. **Dead code on main.** The two files listed above, now deleted.
2. **Superseded one-off scripts on the paper branch**, outside `scripts/paper/`.
   The branch adds 98 top-level `scripts/` files. 47 have one-off names
   (canary, probe, smoke, pilot, recovery, gate, verify, diagnose, or dated,
   e.g. `train_final_gated_deltanet_l40s_4gpu_shallowwide_full_b6ga8_listconfigfix_20260705.sh`),
   and 48 hard-code cluster paths. About 20 of them are pinned by tests, which
   would have to go with them. *Inference:* once the paper is submitted,
   archive every script that neither `scripts/paper/README.md` nor a paper
   result names.

### Archive (move, keep history)

1. **`sallm_memory/`: lab notebook, 7.4 MB.** It has 287 notes dated
   2026-05-18 to 2026-09-12, 48 `.md.sha256` seals, a 766 KB
   `sallm_progress.md` and a 1.66 MB PNG poster.
   - Nothing in `src/`, `tests/` or `scripts/paper/` reads it.
   - Three things do read it: the hash guard in
     `scripts/run_pure_gdn_hpo_correction_canary.py` (lines 312-330, which
     refuses to run if two preregistration notes change),
     `scripts/observe_poc.py`, and `observability_app/vite.config.js`.
   - This needs the user's decision (see follow-ups).
2. **`observability_app/` + `scripts/observe_poc.py` +
   `tests/test_observe_poc.py`** form a single unit: a Vite/React dashboard
   over the notes. Keep or archive the three together.
3. **Superseded launcher versions in `scripts/paper/monomulti/`**, e.g.
   `run_mamba_retrain.sh` and `_v2` sit next to the `v3` that the README
   names. Leave them in place. `scripts/paper/**` must stay verbatim, and
   moving files would mean updating the README mapping.

### Merge (duplicates)

1. **Generation runners.** The paper branch has seven, five of them sharing
   one docstring:
   - `generation/run_generation_direct.py` (base)
   - `full_ft/run_generation_direct_fullft.py` (2-line adapter-optional change)
   - `fft_rollout/runners/gen_direct.py` (67 diff lines)
   - two batch-size-1 xLSTM variants
   - `run_generation_protocol.py`
   - `full_matrix/run_generation_unit.py`

   They are frozen paper copies and must not change. The maintained path
   should be one generation evaluator in `src/main/sallm/evaluation/`
   (adapter-optional, batch size 1 for xLSTM, per issue #137), with the paper
   copies left as they are.
2. **Other near-copies** (`news_score`, `sib_score`, `prefix_eval`,
   `intent_eval`, `general_prompt_lm_eval`, and two `train_fft.py` that differ
   by 171 lines) follow the same pattern: port the final behaviour into the
   package once, and keep the verbatim copies.
3. **Factories.** `src/conf/finetune/factory.py` was the only real duplicate
   and is now gone. `data/factory.py`, `models/factory.py` and
   `training/factory.py` build different things and do not overlap.

### Keep

- **`src/conf`.** All 284 entry configs compose (`src/conf/config.yaml`
  needs a `finetune=` override, which is how it is designed). Configs are
  selected by name at run time, so "no references" does not prove a config is
  dead. Candidates for the user to prune: `base/rwkv_125m`,
  `base/recurrent_gemma_125m` and `base/llama_400m` (in neither paper),
  `finetune/llama_sa_general_all_v2` and `_v3` (`_v2` is used in a test),
  `finetune/llama_sentiment_ft`, and `eval/run_mamba_instruct_xho`. The
  MzansiLM tag keeps the paper-era versions either way.
- **`tokenizer/tokenizer/tokenizer.json`** (2 MB). It is tracked despite the
  `*.json` ignore rule, and was deliberately restored in `efcdb33`. However,
  the configs point at `tokenizer/sallm_bpe_tokenizer`, which only the paper
  branch contains; which one is the paper tokenizer needs confirming.
- **`scripts/pretokenize_dataset.py`.** It is unreferenced, but it is part of
  the documented pretraining toolchain.
- **`scripts/paper/**`.** Kept verbatim.

## Test coverage gaps (main)

- Modules that no test imports by name: `fine_tune/run.py` (487 lines),
  `training/callbacks.py` (421), `configs/evaluation.py` (359),
  `training/factory.py` (233), `data/loaders/mix.py` (209),
  `data/multitask.py` (202), `data/transforms/template_strategies.py` (188),
  `data/adapters/huggingface.py` (170), and all model/data factories.
- The paper branch adds tests for the loaders, model and training factories,
  training run, and callbacks' general validation, which closes most of this
  once it lands.
- The remaining high-value gaps are the ones behind open bugs:
  - special-token ids copied into model configs (#138, #139);
  - validation metrics that change between trained and untrained adapters
    (#142);
  - xLSTM batch-size handling under left padding (#137);
  - a Mamba-2 grouped-norm parity test (#136).

## Tooling

- **ruff, ty, yamllint, markdownlint, shfmt** are all wired into
  `.pre-commit-config.yaml`, and CI runs pre-commit, pytest and Grype. All
  hooks pass on this branch; pytest passes 89/89.
- **`.gitignore` ignores `*.json` and `*.csv` globally**, which is a footgun:
  new JSON or CSV fixtures and configs are silently untracked.
  *Inference:* scope these rules to output directories once the paper is out.
- **Two ruff/ty settings exclude `trash/`**, a directory that no longer
  exists. They are harmless and were left alone to avoid conflicts with
  PR #135.
- **The paper venv cannot run main's tests.** The requested
  `~/Desktop/Masters/sallm/.venv` has transformers 4.57.3, while main
  requires transformers>=5.5. The venv's editable install also points at the
  live checkout. Tests on this branch therefore ran in a separate
  `uv sync --frozen --extra dev` venv inside the cleanup clone, matching CI.
- **PR #135 relaxes main's pins** (`transformers>=5.5,<6` becomes `>=4.53.1`;
  `lm-eval>=0.4.12` becomes `>=0.4.0`; `aiohttp`, `urllib3`, `wandb` and
  `tornado` floors drop) and rewrites `uv.lock` to transformers 4.57.3. Merging
  it as-is undoes the security upgrade from PR #126. See
  `docs/DEPENDENCY_UPGRADE_ASSESSMENT.md`.

## GitHub issues

Nothing was posted. The draft comments below need the user's go-ahead.

| Issue | State on main | Proposed action |
| --- | --- | --- |
| #125 sqlitedict advisory | Fixed by `092805d` (PR #126): sqlitedict is excluded in `[tool.uv]`, the Grype exception was removed, and the main CI scan passes | Close: "Fixed in 092805d: `exclude-dependencies = ["sqlitedict"]`, the Grype waiver was removed, and the security scan is green on main." |
| #124 Restore green CI, port research tooling | CI is green (#126, #127). The port is PR #135 | Close when PR #135 merges, or now as superseded by #135 |
| #136 Mamba-2 gated RMSNorm | Still valid: `registry.py` maps to transformers' `Mamba2ForCausalLM` | Keep open (upstream transformers behaviour; live paper code) |
| #137 xLSTM ignores the mask under left padding | Still valid: no batch-size-1 guard in the evaluators | Keep open; fix in the package after the rollout |
| #138, #139 EOS/PAD and BOS/EOS ids | Still valid: `models/factory.py` sets no `*_token_id` | Keep open; the fix touches published checkpoints |
| #140 NFD normaliser | Still valid: `tokenizer/train.py` uses `NFD()`. PR #135 adds NFC to the scorers | Keep open (tokenizer choice for future runs) |
| #141 AfriXNLI prompt_1 braces | Still valid: `afrixnli_eng.yaml` lists `afrixnli_eng_prompt_1` | Keep open (upstream lm-eval; local override pending) |
| #142 Degenerate in-training validation metrics | Still valid | Keep open |
| #143 xLSTM native kernel | Still valid: `base/xlstm_125m.yaml` sets no kernels | Keep open (the paper branch deliberately keeps the native kernel, per `bbada7b`) |
| #119 Collaboration request (external) | Not code | The user's call |
| #43 pure-GDN HPO plan, #44 xLSTM HPO plan | Protocol plans, superseded by the architecture-comparison paper (*inference*) | The user's call: close as superseded, or keep until the paper records the protocols |
| #35, #36, #37, #38, #40 per-task Mamba trackers (2025, empty bodies, `stale`) | Obsolete | Close: "Superseded by the Mamba-2 lanes of the architecture-comparison work (PR #135)." |
| #39 Mamba-2 T2X reproduction | Stopped lane, with evidence in the notes | The user's call; close as obsolete if the paper covers Mamba-2 T2X |

## Follow-ups that need the user's judgement

1. **Where `sallm_memory/` should live.** Options: keep it in the repo; move
   it to `~/Desktop/Masters/Notes/`; or keep only the 48 sealed
   preregistration notes and their `.sha256` files and move the rest. Moving
   it breaks the hash guard in `run_pure_gdn_hpo_correction_canary.py` and
   the observability app unless those move or are archived too.
2. **How to land PR #135 against main's dependency upgrade.** Either merge
   with the paper pins (reverting PR #126's security floors), or merge the
   code and keep main's pins plus `requirements-paper-2026-09.lock` for
   reproduction.
3. **After the paper deadline (12 October):** archive the one-off top-level
   scripts and their tests, and port the final runner behaviour into
   `src/main/sallm`.
4. **Configs to prune** (listed under Keep).
5. **Which tokenizer is canonical:** `tokenizer/tokenizer/` or
   `tokenizer/sallm_bpe_tokenizer/`.
6. **The issue actions in the table above.**
