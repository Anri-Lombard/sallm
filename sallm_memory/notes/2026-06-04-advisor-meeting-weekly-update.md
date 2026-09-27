# Advisor meeting update - 2026-06-04

Coverage window: Thursday 2026-05-28 through Wednesday 2026-06-03.

## Executive summary

This week moved the SALLM comparison from "Mamba failure forensics" into a concrete xLSTM comparison path.

1. The Mamba claim is now scoped and defensible: under the tested Hugging Face pure-Mamba2 SALLM setup, Mamba is weak/fragile for source-conditioned generation and decoder-only POS/NER. This is not a universal claim that all Mamba implementations fail.
2. A strict approximately 125M xLSTM base was trained to a final 3-epoch point and published privately to Hugging Face.
3. xLSTM clearly improved over Mamba on pretrain-style language-model loss, but did not consistently improve over Mamba on clean source-conditioned generation loss.
4. xLSTM NER/POS base prompting was still zero, but fine-tuned Xhosa NER produced nonzero F1 across all prompts. This is the first positive evidence that xLSTM may avoid the all-zero downstream collapse after supervised tuning.
5. The full monolingual xLSTM downstream wave was submitted, but live monitoring became blocked by HEX/VPN SSH timeouts on 2026-06-03. The last verified state showed no direct job failure.

## Mamba status

No new Mamba training was completed this week after the 2026-05-28 supervisor framing. The current Mamba result remains:

- POS D4 full-data LoRA: token accuracy `0.1960`, exact length `0.0400`; still overgenerated.
- NER D4 full-data LoRA: non-O recall `0.1772`, BIO F1 `0.1769`, all-O rate `0.9375`.
- Interpretation: the model can be fine-tuned somewhat, but decoder-only POS/NER remains poor even after task simplification and constrained evaluation.

The important nuance for the advisors: the defensible claim is about the tested HF pure-Mamba2 path, recipe, prompts, and budget. We still have not trained the original official Mamba directory as a direct control against the Hugging Face implementation.

## xLSTM base training

We first tried to use the A100 node path because the L40S queue was slow, but that exposed a kernel/stability problem rather than a model-quality result.

- Native A100 xLSTM ran stably but too slowly: around 3.7-4.1 seconds/step, not enough for the full 67,498-step 3-epoch run inside the walltime.
- Triton kernel canaries failed:
  - `874730` failed with CUDA illegal memory access around step 7/20.
  - `874732` failed with CUDA illegal memory access around step 2/20.
  - `874736` failed with `ValueError: not enough values to unpack`.
- Decision: use the stable native L40S path with checkpoint/resume instead of spending the week debugging kernels.

Final xLSTM run:

- Run tag: `xlstm_h736_ctx2048_native_4gpu_ddp_3epoch_resume_20260531`
- Canary: `880317`
- Main train: `880318`
- Resume/no-op: `880319`
- Pretrain audit: `880320`
- Clean generation-loss audit: `880321`
- Architecture: `h736_l12_h4_chunk64`, `126,901,952` parameters.
- Main train completed cleanly: `67498/67498` steps, epoch `3.0871`, elapsed `1-17:39:46`, exit `0:0`.
- Training summary: `train_loss=3.85037`, `train_runtime=149951.4838`, `train_samples_per_second=21.606`, `train_steps_per_second=0.45`.
- Best checkpoint: `checkpoint-60000`, best metric `2.969733`.

## xLSTM base results

Pretrain-style audit:

| Model | Mean NLL/token | PPL |
| --- | ---: | ---: |
| LLaMA baseline | `2.47215` | `9.98` |
| xLSTM final 3-epoch | `2.97127` | `17.48` |
| Mamba base | about `3.996` | `44.99` |

This is the cleanest positive xLSTM base result: xLSTM is much better than Mamba as a general LM under the same audit, though still behind LLaMA.

Clean source-conditioned generation-loss audit:

| Task | xLSTM final PPL | Mamba base PPL | LLaMA PPL |
| --- | ---: | ---: | ---: |
| T2X Xho | `479.63` | `347.77` | `64.20` |
| AfriHG Xho | `497.88` | `438.50` | `205.48` |
| AfriHG Zul | `568.79` | `605.85` | `267.32` |

Interpretation for advisors:

- xLSTM learned better general language modeling than Mamba.
- That improvement did not transfer cleanly to source-conditioned generation.
- xLSTM is worse than Mamba on T2X Xho and AfriHG Xho, slightly better on AfriHG Zul, and well behind LLaMA on all three generation-loss gates.
- This suggests the bottleneck may not be only "can the base model learn text"; it may involve the conditional format, source preservation/copying, alignment between prompt and target, or the decoder-only seq2seq framing.

## Hugging Face publish and scratch cleanup

The final xLSTM checkpoint was published privately:

- Repo: `anrilombard/sallm-xlstm-125m-native-3epoch-20260531`
- Commit: `ba2ff845335c8cbf750f8f6f3ebc09468008fef9`

After the publish, old xLSTM canaries, intermediate checkpoints, and large dataset/cache payloads were cleaned from scratch. Scratch usage dropped from about `93.9%` to about `58.5%`. The retained remote artifacts are the final model, `fresh_pretrain_summary.json`, and `checkpoint-60000`.

## xLSTM NER/POS gate

After publishing the xLSTM base, we submitted a narrow NER/POS gate before launching broad downstream evaluation.

Gate tag: `xlstm_ner_pos_gate_20260602_pad64`

Base checkpoint: `anrilombard/sallm-xlstm-125m-native-3epoch-20260531`

Submitted jobs:

- Base evals: `881653` masakhaner_xho, `881654` masakhapos_xho, `881655` masakhaner_all, `881656` masakhapos_all.
- Mono fine-tune/eval: `881657/881658` masakhaner_xho, `881659/881660` masakhapos_xho.
- Multi fine-tune/eval: `881661/881662` masakhaner_all, `881663/881664` masakhapos_all.

Verified gate results so far:

| Setting | Task | Result |
| --- | --- | --- |
| Base xLSTM | masakhaner_xho | all 5 prompts F1 `0.0` |
| Base xLSTM | masakhapos_xho | all 4 prompts token accuracy `0.0` |
| Base xLSTM | masakhaner_all | all 15 prompt/language entries F1 `0.0` |
| Base xLSTM | masakhapos_all | all 12 prompt/language entries token accuracy `0.0` |
| Mono fine-tuned xLSTM | masakhaner_xho | nonzero F1 across all 5 prompts: `0.22605`, `0.24460`, `0.24105`, `0.25363`, `0.24038` |

This is a useful but still cautious result. Base xLSTM does not solve NER/POS by prompting alone. However, fine-tuned Xhosa NER does not repeat the all-zero behavior, which is meaningfully different from the earlier Mamba collapse pattern.

The Xhosa POS fine-tune job `881659` completed cleanly with `train_loss=1.1185359771052996` over 20 epochs, but the corresponding eval `881660` had not been verified before HEX access became blocked.

## Full monolingual xLSTM downstream wave

Because the Xhosa NER gate was nonzero, the full monolingual phase was submitted.

Run tag: `xlstm_downstream_3epoch_20260602_pad64`

Manifest: `outputs/final_submissions/xlstm_downstream_3epoch_20260602_pad64_mono.csv`

Submitted job range: `884610` through `884651`.

Tasks included:

- MasakhaNEWS: English, Xhosa.
- MasakhaNER: Tswana, Xhosa, Zulu.
- MasakhaPOS: Tswana, Xhosa, Zulu.
- SIB: Afrikaans, English, Northern Sotho, Southern Sotho, Xhosa, Zulu.
- InjonjoIntent: English, Southern Sotho, Xhosa, Zulu.
- Generation tasks: AfriHG Xho/Zul and T2X Xho.

Last verified state before the HEX/VPN gap:

- `884612` masakhanews_xho fine-tune completed cleanly, but with high `train_loss=40.11538929658778`; eval was still needed before interpreting it.
- `884610` masakhanews_eng and `884614` masakhaner_tsn were still running.
- No direct job failure had been observed.

From 2026-06-03 03:02 onward, monitoring was blocked by SSH timeouts before `purequota`, so current job state is unknown until the VPN/HEX connection works again.

## Concrete examples to explain in the meeting

Example 1: xLSTM learned general LM better than Mamba, but not conditional generation.

- On pretrain-style loss, xLSTM PPL `17.48` vs Mamba PPL `44.99`.
- On T2X Xho clean generation loss, xLSTM PPL `479.63` vs Mamba PPL `347.77`.
- Meeting interpretation: a lower base LM loss is not enough to guarantee better source-conditioned output. We need to inspect whether the model preserves source tokens, follows the output schema, and handles long conditional targets.

Example 2: NER/POS is still justified as genuinely difficult for this setup.

- Base xLSTM NER/POS prompting was zero across Xhosa and all-language gates.
- Earlier Mamba D4 full-data LoRA also remained poor: POS token accuracy `0.1960`, NER BIO F1 `0.1769`, all-O rate `0.9375`.
- Meeting interpretation: the failure is not just that we forgot to use enough examples in the prompt. Even after supervised adaptation, Mamba struggled, and xLSTM only became nonzero after fine-tuning on Xhosa NER.

Example 3: xLSTM fine-tuned NER has a real positive signal.

- Mono Xhosa NER xLSTM F1 by prompt: `0.22605`, `0.24460`, `0.24105`, `0.25363`, `0.24038`.
- Meeting interpretation: this does not prove xLSTM is good yet, but it proves the downstream path is not dead in the same way as base prompting and gives a reason to finish the monolingual wave.

Example 4: not all blockers are model blockers.

- A100 Triton kernel canaries failed with CUDA illegal memory access, while native kernels were stable but too slow on A100.
- L40S native completed successfully.
- Meeting interpretation: if advisors ask why the xLSTM training path took time, the answer is that we had to choose a stable execution path before drawing scientific conclusions.

## Open questions for advisors

1. Is it still worth completing the full xLSTM downstream sweep if the base generation-loss audit does not beat Mamba on T2X/AfriHG Xho?
2. If xLSTM beats Mamba on NER/POS but not on generation, should we frame this as task-dependent architecture behavior rather than a single architecture ranking?
3. Do we need the original official Mamba implementation as a control before making thesis-level claims, or is it enough to scope the claim to the Hugging Face implementation?
4. If monolingual xLSTM results are noisy, should the next effort go to parser/output inspection, prompt/schema repair, or multi/general downstream scaling?
5. Should a later architecture candidate such as HGRN/BabyHGRN be treated as a fallback, or should the thesis now focus on explaining the Mamba/xLSTM/LLaMA differences?

## Recommended next work

1. Restore VPN/HEX access and immediately check `881660`, `881661`-`881664`, and `884610`-`884651`.
2. Pull and summarize completed xLSTM NER/POS and monolingual downstream artifacts before submitting the multi phase.
3. For any NER/POS zero or malformed output, inspect raw generations and parser-validity before calling it a model failure.
4. Keep the official Mamba implementation comparison as a scoped control. It is valuable, but it should not block summarizing the current HF Mamba evidence honestly.
5. Use the meeting to decide whether the next research effort should prioritize source-conditioned generation/copying diagnostics or downstream classification/tagging results.

## Files and artifacts to mention

- Advisor note from last week: `sallm_memory/notes/2026-05-28-supervisor-meeting-mamba-xlstm-defensibility.md`
- xLSTM run notes: `sallm_memory/notes/2026-05-31.md`, `sallm_memory/notes/2026-06-01.md`, `sallm_memory/notes/2026-06-02.md`
- Current live-monitoring note: `sallm_memory/notes/2026-06-03.md`
- HF downstream plan: `sallm_memory/notes/2026-06-02-xlstm-hf-downstream-plan.md`
- High-level tracker: `sallm_memory/sallm_progress.md`
- Final xLSTM artifact sync root: `outputs/remote_artifacts/xlstm_h736_ctx2048_native_4gpu_ddp_3epoch_resume_20260531/`
- xLSTM monolingual submission manifest: `outputs/final_submissions/xlstm_downstream_3epoch_20260602_pad64_mono.csv`
