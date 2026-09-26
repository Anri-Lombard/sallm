# Mono/Multi rescoring: the General protocol per task

The Mono and Multi rescore must reuse exactly what produced the paper's General numbers, and the same test split.

- **Prompts:** one prompt per task and language, chosen on validation (the highest mean validation score over the four General adapters). Every model and every regime uses it. Do not reselect per regime; that would put the regimes on different prompts.
- **Adapters:** the adapter under test replaces the General adapter.
- **Unchanged:** base model, dtype, answer position, scorer and metric.

| Task | Scorer | Host | Per-arch dtype | Answer position | Metric |
|---|---|---|---|---|---|
| News | natural first-label-token scorer (v15 news snapshot), batch 1, run twice and checked | Kombuys (HEX only after copying the v15 snapshot) | bf16 weights + bf16 autocast for all 4; merge_lora=True, tie=False for all 4 | fixed normalized chat template, context ends `<|assistant|>\n` + 8 spaces; first token where labels diverge; full-vocab log-softmax | support-weighted F1 |
| SIB-200 | natural first-label-token scorer (retained-sib-natural-first-token v5 + balanced-code core v4) | Kombuys (HEX after copying 3 files + YAMLs) | fp32, no autocast, batch 1 right-padded; xLSTM merge_lora=True, tie=False, padded to chunk size | adapter's own template; longest shared prefix of the 7 rendered `{assistant: label}` turns (ends `\n` + 8 spaces) | support-weighted F1 |
| Intent | lm-eval 0.4.9.2 loglik, `run_prefix_eval.py --mode train` (appends `"\n        "`) | Kombuys (portable to HEX) | mzansilm bf16; mamba2 fp32 tie=False (unfixed base); xlstm fp32 merge_lora tie=False; gdn bf16 | `<|assistant|>\n` + 8 spaces; full-label loglik | lm-eval weighted F1 |
| NER | `run_general_sequence_eval_20260917_v2.py` (BOS-repair runner) | HEX, L40S | mzansilm bf16; mamba2 fp32 on EOS-fixed base; xlstm fp32 merge_lora (xLSTM venv); gdn bf16 | runner renders `[BOS]        <|user|>\n        {content}[EOS]<|assistant|>`; greedy generate_until | micro span F1 |
| POS | same runner (or the bounded `run_gdn_mono_pos.py` with `--language`) | HEX, L40S | same as NER but mamba2 on the unfixed base | adapter chat template; closed 17-tag scoring | micro token accuracy |

Selected prompts (shared by all models and regimes):

| Task | afr | eng | nso | sot | tsn | xho | zul |
|---|---|---|---|---|---|---|---|
| News | | p2 | | | | p4 | |
| SIB-200 | p4 | p3 | p4 | p3 | | p5 | p5 |
| Intent | | p4 | | p1 | | p2 | p2 |
| NER | | | | | P2 | P5 | P5 |
| POS | | | | | P3 | P3 | P3 |

Sources: News and SIB `/scratch/alombard/sallm/results/mamba_sixfamily_priority_replacement_20260915_v10/validation/news/final_shared_prompt_selection.json` and `.../official/protocols/sib_validation/selection.json`; Intent `.../general-prompt-official-kombuys-20260915-v1/selection/SELECTION.json`; NER/POS `/scratch/lmbanr001/masters/sallm/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/selection/SELECTION.json`.

The paper's General News and SIB numbers are the v10 rerun outputs: `/scratch/alombard/sallm/results/mamba_sixfamily_priority_replacement_20260915_v10/official/news/<arch>/summary.json` and `.../official/sib/<arch>.json`. All 48 cells match data/source-snapshot.csv. The older 20260914 official folders used different prompts and are superseded.

Existing full-matrix Mono/Multi outputs on HEX:
- `official/prompt/*` (News, SIB, Intent) came from `run_lm_eval_unit.py`, which does **not** use the General protocol:
  - News and Intent go through the lm-eval chat path with `target_delimiter=""`, so the label is scored directly after `<|assistant|>` with no newline and 8 spaces, using full-label loglik and the base tokenizer.
  - SIB goes through the raw path with no chat template, using afrobench prompt text.
  - None of these can be reused.
- `official/sequence/{22,23,24,27,28,29,78,80}` (GDN NER/POS Mono/Multi) match General GDN request-for-request and can be reused.

## News (MasakhaNEWS eng, xho)

- Entry point (official, General-locked): Kombuys `/scratch/alombard/sallm_snapshots/news-general-corrected-official-20260914-v15/scripts/run_news_natural_first_token_official.py` (sha b13617ef...), wrapped by `run_frozen_official_scorer_v10.py` (sha 45dc17b2...) for the v10 prompt gates. Scoring code: `scripts/run_mamba_news_interface_diagnostic.py::_score_interfaces` (28d9706a...) plus helpers `scripts/run_mamba_news_common_validation.py` (a620218a...). Runtime PYTHONPATH `<v15 snapshot>/src/main` (needs v15 `classification_metrics` 8dc06c3f; the v6 runtime adds an extra space and gives newline + 9 spaces).
- Prompt text: `<v15>/src/conf/templates/masakhane_news_classification/lm_eval_p{2,4}.yaml` field `prompt`, `.format(headline=, text=)`.
- Test data: `/scratch/alombard/sallm/assets/news-general-official-test-20260914-v1/{eng,xho}-test.tsv` (MasakhaNEWS rev fa3b5fff; eng 948, xho 297).
- Why the official script can't take a Mono/Multi adapter: it only accepts `arm.id == "general"` from `asset_root/runtime_adapters/general`, pins adapter/base/template hashes, and requires `CUDA_VISIBLE_DEVICES == "1"` plus the v10 validation reports.
- Adaptation: a new driver `score_trainpos.py news`, written in a new directory without editing the frozen files. It:
  - imports the common and diagnostic modules (`diag.common = common`);
  - loads `load_model_and_tokenizer(ModelEvalConfig(checkpoint=BASE, peft_adapter=ADAPTER, dtype="bfloat16", device="cuda:0", merge_lora=True, tie_word_embeddings=False))`;
  - sets `tok.chat_template = common.NORMALIZED_CHAT_TEMPLATE`, prefixed with `"{{- bos_token -}}\n"` when the adapter's own template has that line (it matches training; see the caveat below);
  - builds `ev = ClassificationEvaluator(tok, max_samples_per_lang=None)`;
  - scores each row with `diag._score_interfaces(...)["first_token_prediction"]`;
  - reports `common.manual_classification_metrics(gold, pred)["weighted_f1"]`.
- Mono: each language adapter scores only its own language's rows. Multi: one adapter scores both.
- Caveat: General News used template df244b52 (no BOS) for MzansiLM, xLSTM and the v10 Mamba adapter, and the BOS version (820317d0) for GDN. The adapters were trained with the BOS template c681a1d8, so the General Mamba (and possibly MzansiLM/xLSTM) News score lacks the training [BOS]. The choice for Mono/Multi is either:
  - (a) mirror General exactly, per architecture: the same template General used for that architecture; or
  - (b) use the training-faithful template everywhere and rerun General News for the three non-GDN models (cheap, about 3 min each).
  I recommend (b), since the whole point is one consistent protocol. Otherwise take (a) and state it.
- Runtime (Kombuys 3080 Ti, eng+xho 1245 rows): 150 s (mzansilm), 190 s (mamba), 230 s (xlstm), 275 s (gdn) wall. A Mono adapter is proportionally less (eng 76%, xho 24% of rows).

Command template (Kombuys; the driver does not exist yet):
```
export HF_HOME=/scratch/alombard/sallm/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline \
  FLA_DISABLE_BACKEND_DISPATCH=1 SALLM_SKIP_MAMBA_KERNEL_CHECK=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu>
PYTHONPATH=/scratch/alombard/sallm_snapshots/news-general-corrected-official-20260914-v15/src/main \
/scratch/alombard/sallm/.venv/bin/python score_trainpos.py news --arch <a> --base <BASE> --adapter <ADAPTER> \
  --langs eng:p2,xho:p4 --tsv-dir /scratch/alombard/sallm/assets/news-general-official-test-20260914-v1 --out <json>
```

## SIB-200 (afr, eng, nso, sot, xho, zul)

- Entry point (official, General-locked): Kombuys `/scratch/alombard/sallm/snapshots/sib-general-corrected-official-replacement-20260914-v1/run_sib_natural_first_token_official_replacement.py` (2b850274...). Scoring: `/scratch/alombard/sallm/snapshots/retained-sib-natural-first-token-20260914-v5/run_sib_natural_first_token_eval.py::score_prompt` (b247505a...). Core: `retained-sib-balanced-code-20260914-v4/run_sib_balanced_code_eval.py` (866e9d82...). Runtime `retained-sib-evaluator-20260913-v6/src/main` (identical to HEX `downstream-generation-20260914-v8/src/main`).
- Prompt text: `sib_validation/sallm_sib_{lang}_val_prompt_{N}.yaml` `doc_to_text` with `{{text}}` replaced, applied to test rows (Davlan/sib200 rev 38977a66, 204 rows per language).
- Why the official script can't take a Mono/Multi adapter: fixed model list, `EXPECTED_PROMPTS`, tokenizer/template hashes tied to the General token audit, and per-model file hashes.
- Adaptation: `score_trainpos.py sib`, about 50 lines. It:
  - loads the core with `scorer.load_core(...)`;
  - loads `scorer.load_model_and_tokenizer(scorer.ModelEvalConfig(checkpoint=BASE, dtype="float32", device="cuda:0", peft_adapter=ADAPTER, merge_lora=M, tie_word_embeddings=T))`, with M=True, T=False for xLSTM and None/None otherwise;
  - sets `pad = tok.pad_token_id or tok.eos_token_id` and `mult = ClassificationEvaluator._get_model_chunk_size(model)`;
  - for each row calls `score_prompt(...)` and takes `prediction(scores)[0]`;
  - reports `summarize(rows)["metrics"]["f1"]`.
  It fails loudly if the first label tokens are not unique under the adapter's template.
- Mono: one adapter per language, run on its own language only. Multi: one adapter, all six.
- Runtime (Kombuys 3080 Ti, 6 languages, 1224 rows): 35-110 s wall per adapter.

Command template (Kombuys):
```
PYTHONPATH=/scratch/alombard/sallm/snapshots/retained-sib-evaluator-20260913-v6/src/main \
/scratch/alombard/sallm/.venv/bin/python score_trainpos.py sib --arch <a> --base <BASE> --adapter <ADAPTER> \
  --langs afr:4,eng:3,nso:4,sot:3,xho:5,zul:5 [--merge-lora --no-tie  # xlstm] --out <json>
```

## Drivers written and smoke-tested (Phase 1)

`drivers/score_sib_trainpos.py` and `drivers/score_news_trainpos.py` (copies on Kombuys in `/scratch/alombard/sallm/results/monomulti_rescore_inventory_20260924/`) wrap the frozen scoring functions and skip the General-only gates. Both were checked against the official v10 General GDN outputs on the first 30 test rows per language:
- SIB (afr, nso): 60/60 predictions identical, max |logprob diff| 0.0038.
- News (eng, xho): 60/60 identical.

The smoke ran on Kombuys GPU0 (RTX 5090), while General ran on GPU1 (3080 Ti). For bit-level parity, run on GPU1 or rescore General on the same GPU.

## Bases (must equal the base each adapter was trained on)

| arch | Kombuys base | HEX base | adapters that reference it |
|---|---|---|---|
| mzansilm | `/scratch/alombard/sallm/results/retained_standardized_20260913_v1/bases/mzansilm` (uctnlp/mzansilm-125m@7f017bc) | `/scratch/lmbanr001/masters/sallm/checkpoints/sallm-llama-125m/final_model` (pytorch_model.bin 7388b67c...) | task adapters were trained on the HEX path. Verified on 24 Sep that it is weight-identical to uctnlp/mzansilm-125m@7f017bc: all 272 tensors are equal, and lm_head is tied. The hub Intent adapters' `adapter_config.base_model_name_or_path` wrongly says `anrilombard/sallm-mamba-125m` (a metadata slip; targets are q_proj/v_proj). |
| mamba2 | `retained_standardized_20260913_v1/bases/mamba2` (weights 7eaa6b8a) | `full-matrix-retained-bindings-20260916-v1/bases/mamba2`; NER: `results/mamba_eos_fix_20260924/bases/mamba2_eosfix` | all Mamba adapters; weights byte-identical across copies |
| xlstm | `retained_standardized_20260913_v1/bases/xlstm` (native-3epoch-20260531@ba2ff84) | `results/downstream_standardized_20260913_v1/bases/xlstm` | two W&B sweep adapters (Multi NER 4vfuoadl, Multi SIB k6ob2x17) reference `anrilombard/sallm-xlstm-125m`, a different hub repo (same file sizes; identity unverified) |
| gdn | `/scratch/alombard/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model` | `/scratch/lmbanr001/masters/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model` (tree 5ceca7d0) | all GDN adapters |

Adapter files: copy Kombuys-only adapters to HEX (or HEX-only adapters to Kombuys) into a new directory. Check tree_sha256 against manifest.csv after the copy.

## Intent (InjongoIntent eng, sot, xho, zul)

General protocol (the numbers in data/general-intent-belebele-trainpos.csv):
- Script: Kombuys `/scratch/alombard/sallm_snapshots/general-prefix-fix-20260924/run_prefix_eval.py` (sha256 d0066af2...476a), driven by `run_all.sh` (General) and `run_control.sh` (control); `build_csv.py` extracts the protocol prompts.
- Args: `--spec <units.json> --unit-id <id> --arch <arch> --mode train --packs injongointent_all --output <new dir>`.
- `--mode train` patches `HFLM.apply_chat_template` to append `"\n        "` after `<|assistant|>`; it asserts every context ends with `<|assistant|>\n        ` and every continuation is a valid label.
- lm-eval 0.4.9.2 loglikelihood, 0-shot, test split, `apply_chat_template=True`, batch `auto:4` (max 64). Runs all 5 prompts; the score is lm-eval weighted F1 on the protocol prompt: eng p4, sot p1, xho p2, zul p2.
- Prompt selection: `.../general-prompt-official-kombuys-20260915-v1/selection/SELECTION.json` (sha 79bf1ef2...). Rule: highest unweighted mean validation score across the four General adapters, lowest id on ties; validation carved from train. One prompt per language, shared by every model. Caveat: the selection used the older mean-logprob scorer, before the prefix fix, and all validation means sit at 0.1-0.4 F1, so the choice is effectively arbitrary. It is still the frozen rule, so reuse it unchanged for Mono and Multi. The same prompts are in HEX `full-matrix-execution-20260916-v1/build_inventory.py` (`INTENT_PROMPTS`).
- Per-architecture interface: mzansilm bf16; mamba2 fp32 + `tie_word_embeddings=False` on the **unfixed** base `retained_standardized_20260913_v1/bases/mamba2` (loglik only, so the EOS fix does not matter; keep the unfixed base to match General); xlstm fp32 + `merge_lora=True` + `tie_word_embeddings=False`; gdn bf16.
- Env: one venv `/scratch/alombard/sallm/.venv` (torch 2.9.1, transformers 4.57.3, peft 0.18.1, patched xLSTM backend 5f20208d).

Nothing is hard-coded to the General adapters; `--spec` provides base and adapter plus their tree sha256 (checked with `prepare_bindings.tree_sha256`). The control run is the template: data/intent-position-control.csv scored the Multi MzansiLM adapter this way.

Command template (Kombuys):
```
# spec.json entries: {"unit_id": "mono-intent-<arch>-<lang>", "architecture": "<arch>", "base": <same base as General units.json for arch>,
#                     "base_tree_sha256": ..., "adapter": <adapter dir>, "adapter_tree_sha256": <manifest tree_sha256>}
cd /scratch/alombard/sallm_snapshots/general-prefix-fix-20260924
export CUDA_VISIBLE_DEVICES=<g> HF_HOME=/scratch/alombard/sallm/hf HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1 \
  FLA_DISABLE_BACKEND_DISPATCH=1 MAMBA_SCAN_IMPL=cuda TMPDIR=/scratch/alombard/sallm/tmp/general_prefix_fix \
  PYTHONPATH=/scratch/alombard/sallm_snapshots/downstream-generation-20260914-v8/src/main:/scratch/alombard/sallm_snapshots/full-matrix-execution-20260916-v1
/scratch/alombard/sallm/.venv/bin/python run_prefix_eval.py --spec <spec.json> --unit-id <id> --arch <arch> \
  --mode train --packs injongointent_all --output /scratch/alombard/sallm/results/monomulti_rescore_20260925/<id>-train
```
Adaptations needed:
- For Mono, keep only the matching language's protocol-prompt task (the pack still runs all 4 languages x 5 prompts; restricting `--packs` to one language's pack, if one exists, saves about 4x).
- `build_csv.py` keys rows by architecture (or by `unit_id` when the id starts with `control`). With several units per architecture, key on `unit_id` (for example by prefixing ids with `control-`, or with a 3-line change in a copied `build_csv.py`).
- The script also runs on HEX. The dependencies match (`downstream-generation-20260914-v8` manifest f2171f80, `prepare_bindings.py` 9876aa3e, same library versions, datasets in `/scratch/lmbanr001/hf-cache`). Copy the script to a new HEX dir, set `HF_HOME=/scratch/lmbanr001/hf-cache`, and use the xLSTM runtime venv for xlstm. First rerun one General unit on HEX and diff it against Kombuys, since near-ties in log-likelihood can flip across GPUs.
- All Mono/Multi Intent adapters use chat template c681a1d8 or ef7dead4 (identical rendering to General), so the `<|assistant|>\n        ` assert should pass.

Runtime: Intent-only is about 20 min for mzansilm, gdn and xlstm and about 45 min for mamba2 (estimated from the General runs on a shared RTX 5090 / 3080 Ti). Do not co-schedule mamba2 with xLSTM on one GPU (OOM).

## NER (MasakhaNER tsn, xho, zul) and POS (MasakhaPOS tsn, xho, zul)

General protocol:
- NER runner: HEX `/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260917_v2/control/run_general_sequence_eval_20260917_v2.py` (sha 9cad0519..., the BOS-repair runner). Do not use `run_general_sequence_eval_20260914.py` for NER: it dropped `[BOS]` for mzansilm and xlstm. v2 renders `"[BOS]        <|user|>\n        {content}[EOS]<|assistant|>"` with `apply_chat_template=False`.
- POS: same v2 runner (identical POS path; v2 only requires `--adapter`). General POS outputs are in `general_sequence_official_test_20260916_v1/raw/pos/`.
- Protocol JSON: `general-sequence-official-test-20260916-v1/general_sequence_validation_hex_protocol_20260915_v3.json` (sha 2041716c; equals `full-matrix-execution-20260916-v1/general_sequence_protocol_template.json`). For mamba2 NER use `mamba_eos_fix_20260924/protocols/general_ner_protocol_eosfix.json`.
- Selection: `general_sequence_validation_20260916_hex_v5_final_amendment_v2/selection/SELECTION.json` (sha ce45c0b2...). Rule: highest unweighted mean validation score across the four General adapters. One prompt per task-language, shared by all models: NER tsn P2, xho P5, zul P5; POS P3 for all three.
- NER: lm-eval `generate_until`, greedy (`do_sample: false`), stop at `</s>` and `<|im_end|>`, lm-eval default max_gen_toks 256 (not verified on disk), batch `auto:4` (max 16), 1024-token input cap, seed 42, 0-shot. Scorer: `format_span` filter + `utils.span_f1_agg` (micro span F1).
- POS: closed 17-label UPOS decoding one token at a time, mean continuation log-prob, `pad_to_multiple_of=64`, micro token accuracy, 1024-token cap.
- Per arch: mzansilm bf16; mamba2 fp32 + `tie_word_embeddings=False`, NER base `mamba_eos_fix_20260924/bases/mamba2_eosfix` (eos 1, pad 2; weights byte-identical to the unfixed base), POS base `downstream_standardized_20260913_v1/bases/mamba2` (unfixed, as General); xlstm fp32 + merge_lora with the xLSTM runtime venv; gdn bf16. General ran on L40S.

Hard-coding: `verify_binding` requires `--checkpoint/--adapter` to equal `protocol.models[arch].base_path/adapter_path` and checks the sha256 of every listed file. So each Mono/Multi adapter needs its own protocol JSON copy that changes only `models.<arch>.adapter_path`, `adapter_files` (the actual files; some adapters ship `adapter_model.bin`) and `adapter_tree_sha256`. Check that the non-model fields are byte-identical with `jq -S 'del(.models)'`. The task-level test releases `general_sequence_official_test_20260916_v1/release/{NER,POS}_TEST_ACCESS_RELEASED_V1.json` can be reused.

Command template (HEX, one GPU; L40S to match General):
```
PY=/home/lmbanr001/masters/sallm/.venv/bin/python   # xlstm: /scratch/lmbanr001/masters/sallm/results/standardized_adapter_recovery_20260914_hex_v1/runtime/.venv/bin/python
bundle=/scratch/lmbanr001/masters/sallm_snapshots/general-sequence-official-test-20260916-v1
repair=/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260917_v2/control
export HF_HOME=/scratch/lmbanr001/hf-cache HF_DATASETS_CACHE=/scratch/lmbanr001/hf-cache/datasets HF_HUB_CACHE=/scratch/lmbanr001/hf-cache/hub \
  HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True FLA_DISABLE_BACKEND_DISPATCH=1 MAMBA_SCAN_IMPL=cuda \
  PYTHONPATH=/scratch/lmbanr001/masters/sallm_snapshots/downstream-generation-20260914-v8/src/main:$bundle:$repair
$PY $repair/run_general_sequence_eval_20260917_v2.py --task <ner|pos> --phase test --architecture <arch> \
  --checkpoint <base_path in unit protocol> --adapter <adapter dir> --protocol <unit_protocol.json> \
  --selection /scratch/lmbanr001/masters/sallm/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/selection/SELECTION.json \
  --release /scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260916_v1/release/<NER|POS>_TEST_ACCESS_RELEASED_V1.json \
  --output /scratch/lmbanr001/masters/sallm/results/monomulti_rescore_20260925/<task>/<unit>.json
```
- NER always scores all three languages (no language flag); for Mono, keep only the matching language.
- POS: the bounded runner `full_matrix_execution_20260916_v1/gdn-mono-pos-official-20260920-v2/run_gdn_mono_pos.py` adds `--language` and matches General row for row (prompt_sha256 and input_token_count). Use it for Mono POS to cut cost by about 3x.
- POS context comes from the adapter's chat template. 52 adapters share General's c681a1d8, 44 use ef7dead4 (renders the same `<|assistant|>\n        ` prefix), and one differs: xLSTM Multi NER 4vfuoadl uses df244b52. NER is unaffected because v2 renders the prompt itself.

Already done with this protocol and reusable as-is (prompt_hash equals General GDN's on every row):

| Regime | Task | tsn | xho | zul | unit |
|---|---|---|---|---|---|
| GDN Mono | NER span F1 | 67.83 | 52.68 | 50.26 | u22/u23/u24 (matching language only) |
| GDN Multi | NER span F1 | 78.35 | 70.00 | 73.61 | u78 |
| GDN Mono | POS acc | 85.08 | 85.63 | 87.57 | u27/u28/u29 |
| GDN Multi | POS acc | 85.64 | 83.44 | 86.01 | u80 |

Residual risk: 27-29 ran on L40S; the GPU type for 22-24/78/80 is unconfirmed, and bf16 greedy decoding can diverge on rare tokens across GPU types.

The other full-matrix Mono/Multi units (prompt/18-21, 25, 26, 30-35, 77, 79, 81, 89) used the lm-eval chat path without the answer prefix (News, Intent) or the raw path (SIB). They are **not** the General protocol and cannot be reused.

Runtime (elapsed_seconds): NER, 3 languages, L40S: 400-900 s per adapter (mzansilm 659, mamba2 891, xlstm 688, gdn 778; GDN Mono 395-730). POS, 3 languages: mzansilm 45 min, gdn 97 min, xlstm 108 min, mamba2 124 min; POS Mono single language: about 15-30 min (GDN: 1716/953/911 s).
