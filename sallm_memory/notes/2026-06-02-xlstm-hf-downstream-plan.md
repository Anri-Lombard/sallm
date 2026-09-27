# xLSTM 3-epoch base: HF publish and downstream gate plan

## Context

- Completed base run: `xlstm_h736_ctx2048_native_4gpu_ddp_3epoch_resume_20260531`.
- Main train job `880318` completed cleanly at `67498/67498`; resume job `880319` resumed from `checkpoint-67498` and no-opped cleanly; audits `880320` and `880321` completed cleanly.
- Final pretrain-style audit:
  - xLSTM final PPL `17.48`.
  - Mamba base PPL `44.99`.
  - LLaMA base PPL `9.98`.
- Final clean generation-loss audit:
  - xLSTM final PPLs: T2X Xho `479.63`, AfriHG Xho `497.88`, AfriHG Zul `568.79`.
  - Mamba base PPLs: T2X Xho `347.77`, AfriHG Xho `438.50`, AfriHG Zul `605.85`.
  - LLaMA base PPLs: T2X Xho `64.20`, AfriHG Xho `205.48`, AfriHG Zul `267.32`.
- Interpretation: xLSTM is useful architecture evidence and improves held-out pretrain loss versus Mamba, but it is not a LLaMA replacement. Downstream should be gated cheaply before running the full suite.

## Recommended sequence

1. Publish the completed xLSTM base model to Hugging Face.
   - Prefer a new explicit repo or revision rather than silently overwriting `anrilombard/sallm-xlstm-125m`.
   - Suggested repo: `anrilombard/sallm-xlstm-125m-native-3epoch-20260531`.
   - Upload the exact audited base model after verifying whether the canonical model should be `final_model`, `checkpoint-67498`, or `checkpoint-60000`.
   - Note: trainer state says `best_model_checkpoint=checkpoint-60000`, while the completed final checkpoint is `checkpoint-67498`; verify which artifact the audits used before publishing.
   - Keep the repo private initially unless the user explicitly wants it public.
   - Include model card details: recipe, job IDs, pretrain audit, clean generation audit caveat, W&B links, tokenizer, source checkpoint path, and known xLSTM generation/chunking constraints.
   - Status 2026-06-02: done. Published private repo `anrilombard/sallm-xlstm-125m-native-3epoch-20260531` at commit `ba2ff845335c8cbf750f8f6f3ebc09468008fef9`.

2. After HF upload is verified, clean scratch before launching downstream.
   - Current scratch pressure after completion was `/scratch` `93.9%`.
   - Once the HF model and local lightweight artifacts are verified, clear pretraining dataset/model payloads from scratch to recover space before NER/POS.
   - Cleanup should be explicit and conservative:
     - Keep the uploaded/audited canonical xLSTM checkpoint until HF integrity is confirmed.
     - Keep lightweight logs, summaries, manifests, and audit rows locally.
     - Do not delete anything without an explicit user approval command.
   - Candidate cleanup targets after approval:
     - redundant xLSTM canary checkpoint directories,
     - intermediate checkpoints from the completed 3-epoch run if final/HF copy is verified,
     - stale pretraining dataset/cache payloads that can be regenerated or are no longer needed for downstream.
   - Status 2026-06-02: done. Scratch quota refreshed from `93.9%` to `58.5%` after deleting old xLSTM canaries/intermediate checkpoints and the `anrilombard___mzansi-text-tokenized` cache. Retained current xLSTM `final_model`, `fresh_pretrain_summary.json`, and best checkpoint `checkpoint-60000`.

3. Run a narrow NER/POS gate first.
   - Goal: see whether xLSTM gets nonzero structured-task results where Mamba got `0`.
   - Do not start the full downstream wave until this gate has signal.
   - Recommended tag: `xlstm_downstream_3epoch_20260602_pad64`.
   - Base-checkpoint source for jobs should be the verified HF model id or the canonical scratch checkpoint, not the older LLaMA-budget xLSTM base.
   - Gate jobs:
     - base eval: `masakhaner_xho`, `masakhapos_xho`, optionally `masakhaner_all`, `masakhapos_all`;
     - mono fine-tune/eval: `xlstm_ner_xho`, `xlstm_pos_xho`;
     - multi fine-tune/eval: `xlstm_ner_all`, `xlstm_pos_all`.
   - Gate success condition:
     - any nonzero, parseable NER/POS score is meaningful because Mamba got zero;
     - if both NER and POS remain zero after base and fine-tuned xLSTM, stop and inspect outputs/templates/parsing before spending more GPU.
   - Status 2026-06-02: submitted as tag `xlstm_ner_pos_gate_20260602_pad64` using base checkpoint `anrilombard/sallm-xlstm-125m-native-3epoch-20260531`.
   - Submitted jobs:
     - `881653` base `masakhaner_xho` eval;
     - `881654` base `masakhapos_xho` eval;
     - `881655` base `masakhaner_all` eval;
     - `881656` base `masakhapos_all` eval;
     - `881657` mono `masakhaner_xho` finetune;
     - `881658` mono `masakhaner_xho` eval;
     - `881659` mono `masakhapos_xho` finetune;
     - `881660` mono `masakhapos_xho` eval;
     - `881661` multi `masakhaner_all` finetune;
     - `881662` multi `masakhaner_all` eval;
     - `881663` multi `masakhapos_all` finetune;
     - `881664` multi `masakhapos_all` eval.
   - Manifest: `outputs/final_submissions/xlstm_ner_pos_gate_20260602_pad64_all.csv`.

4. If NER/POS gate passes, expand downstream in stages.
   - Monolingual:
     - NER/POS for Xho/Zul/Tsn first;
     - then News/SIB/Intent;
     - then T2X/AfriHG.
   - Multilingual:
     - `xlstm_ner_all`, `xlstm_pos_all`, `xlstm_news_all`, `xlstm_sib_all`, `xlstm_injongointent_all`, `xlstm_afrihg_all`.
   - General:
     - run `xlstm_sa_general_all` only after mono/multi results are sane.

5. For each submitted downstream wave, create/update a heartbeat.
   - Watch submitted job IDs.
   - Start every HEX pass with `/scratch/slurm/bin/purequota`.
   - Pull only lightweight manifests, summaries, logs, and result rows.
   - Update the daily note and only update `sallm_progress.md` when defensible downstream results land.
   - Status 2026-06-02: done. Active hourly watcher: `sallm-xlstm-ner-pos-gate-watch`.

## Decision rules

- Immediate positive signal: xLSTM gets nonzero NER or POS where Mamba got zero.
- Strong positive signal: xLSTM beats Mamba on NER/POS and at least some classification/general tasks.
- Weak/negative signal: xLSTM only improves pretrain-style PPL but remains weak downstream.
- Stop condition: NER/POS remain zero after both base and fine-tuned gate runs; investigate output format, parser, and task formulation before running the full suite.
