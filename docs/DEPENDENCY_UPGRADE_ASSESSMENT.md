# Dependency Upgrade Assessment, 27 September 2026

Assessment only. No versions were changed anywhere. Sources: `pyproject.toml` and `uv.lock` on branch `paper/architecture-comparison-2026-09` (tip `d55520e`) and on `main`; `uv pip freeze` of the local paper venv; cluster version records in `scripts/paper/**` and `sallm_memory/notes/**`; PyPI JSON; upstream release notes, PRs and source at the tagged versions. "Confirmed" means read in a primary source; "inferred" means reasoned from one; "unverified" means not checked.

## Correction to the brief

The `v8 / v15 / v6` names in `scripts/paper/README.md` are frozen copies of SALLM's own `src/main`, put on `PYTHONPATH` by the runners. They are not library pins. The library versions behind the paper numbers come from the cluster venv records:

- HEX live venv (`monomulti/reselect/hex/mamba2_train.sh`): torch 2.9.1, transformers 4.57.3, trl 0.26.2, peft 0.18.1, mamba_ssm 2.3.2.post1, datasets 4.8.5.
- xLSTM venv (`xlstm_train.sh`): the same plus accelerate 1.12.0, xlstm 2.0.5, mlstm_kernels 2.0.2.
- Kombuys and HEX kernels (notes from 28 Jun to 3 Aug): torch 2.9.1+cu128, causal-conv1d 1.6.2.post1, flash-linear-attention / fla-core 0.5.1.
- lm-eval 0.4.9.2 (`monomulti/protocol.md`). triton 3.5.1 (`uv.lock`, pulled in by torch 2.9.1).

`req.txt` (last changed 3 Mar 2026: torch 2.8.0, transformers 4.53.1, triton 3.2.0, mamba-ssm 2.2.4) is stale and does not record the paper runtime.

## Version table

| Package | Paper runtime | main `uv.lock` | Latest (PyPI, date) | Relevant to our code | Risk |
|---|---|---|---|---|---|
| transformers | 4.57.3 | 5.15.1 | 5.17.0 (09 Sep) | See the transformers section. Fixes Mamba-2 beam cache reordering. Leaves #136, #137 and #143 unfixed. Breaks paper-branch code in 3 places | High |
| torch | 2.9.1+cu128 | 2.9.1 | 2.14.0 (02 Sep) | Prebuilt wheels for mamba-ssm 2.3.2.post1 and causal-conv1d 1.7.0 stop at torch 2.10 (confirmed, release assets). Later versions mean source builds on HEX/Kombuys. torch 2.14 pins triton ~=3.8.0. No fix for our code identified | High |
| flash-linear-attention | 0.5.1 | absent (main dropped the `pure-gdn` extra) | 0.5.2 (27 Jul) | Needed for transformers 5.x: FLA 0.5.1's `FLALayer` has no `get_max_length`, so under 5.x the cache "can't instantiate abstract class" (FLA #1009, seen on 5.13.0). 0.5.2 adds it. The FLA cache still has no `reorder_cache` (confirmed in `fla/models/utils.py` v0.5.2), so GDN and FLA Mamba-2 beam search stays cache-free. Also adds Blackwell GDN autotune limits (#953, #1000), relevant to the RTX 5090, and fused GDN q/k/v short-conv in the no-cache path (#972, numerics unverified) | Medium (required with tf 5.x) |
| mamba-ssm | 2.3.2.post1 | not locked | 2.3.2.post1 (09 May) | No newer release. Requires triton>=3.5 and tilelang==0.1.8 | None |
| causal-conv1d | 1.6.2.post1 | not locked | 1.7.0 (20 Aug) | Only change: packed-sequence initial states (#118). Not used by us | Low |
| xlstm / mlstm_kernels | 2.0.5 / 2.0.2 | not locked | 2.0.6 / 2.0.6 (Sep) | mlstm_kernels 2.0.5 swaps the arbitrary-length wrapper for a padding wrapper (#14) and drops "triton native" (#15). 2.0.6 casts qk_scale to fp32 (#20) and removes batch-size specialisation (#17). xLSTM logits can shift. Neither touches #137, which lives in transformers' `modeling_xlstm` | Medium |
| datasets | 4.8.5 | 4.4.2 | 5.0.1 (28 Jul) | 5.0 changes the `IterableDataset.shuffle` default to multi-shard buffering (release note). SALLM has a `streaming` flag, default False, so impact is unverified. Merging onto main as-is downgrades 4.8.5 to 4.4.2 | Medium |
| tokenizers | 0.22.2 | 0.22.2 | 0.23.2 (03 Sep) | transformers 5.17.0 requires >=0.23.1 and 4.57.3 requires <=0.23.0, so the version is forced by transformers. Both trees set `backend_tokenizer.decoder = ByteLevel()` and depend on the NFD normaliser | Low-Medium |
| peft | 0.18.1 | 0.18.1 | 0.21.0 (15 Sep) | Still rejects Mamba-2 `out_proj`: `_check_lora_target_modules_mamba` blocks `{"out_proj","conv1d"}` for `model_type` mamba2 in 0.21.0 (confirmed). FLA's `Mamba2Config.model_type` is also "mamba2", so the FLA class is blocked too. Upgrading does not unblock the frozen Mamba-2 LoRA protocol. Release notes include transformers v5 fixes (#2934, #2937) | Low (fixes nothing we need) |
| lm-eval | 0.4.9.2 | 0.4.12 | 0.4.13 (31 Aug) | 0.4.10: HF backend moves behind the `lm_eval[hf]` extra. 0.4.11: afrobench Belebele config version 2 to 3 (#3551). Our `belebele_*_prompt_N` tasks come from upstream; whether the prompt content changed is unverified. 0.4.12: `TaskManager.load` shape change (`lm_eval_runner.py` builds a `TaskManager`). 0.4.13: AfriXNLI prompt_1 Jinja-brace fix (#3944) | Medium (changes task definitions) |
| accelerate | 1.12.0 | 1.12.0 | 1.15.0 (09 Sep) | Nothing relevant found. Release notes only skimmed (unverified) | Low |
| trl | 0.26.2 | 0.26.2 | 1.14.0 (25 Sep) | The v1 migration touches us in two places: packing `"bfd-requeue"` is renamed (we pass a bool), and automatic `None` stripping is removed from SFT preprocessing (impact on our chat datasets unverified). trl 1.14 requires datasets>=4.7.0, above main's 4.4.2 | Medium |
| huggingface-hub | 0.36.0 | 1.29.0 | 2.0.0 (24 Sep) | Blocked. transformers 5.17.0, datasets 5.0.1 and tokenizers 0.23.2 all require <2.0. transformers 4.57.3 requires <1.0, which is why the paper stack is on 0.36 | Blocked |
| triton | 3.5.1 | 3.5.1 | 3.8.0 (28 Aug) | Follows torch. FLA, mlstm_kernels and mamba-ssm Triton kernels are autotuned per Triton release, so a change can shift numerics and speed | Tied to torch |

## transformers 4.57.3 to 5.x in detail (confirmed from tagged source)

What 5.x fixes:

- **Mamba-2 beam search.** In 4.57.3, beam search reorders only `past_key_values` (`generation/utils.py`), so `cache_params` is never reordered. In 5.17.0, Mamba-2 uses a `Cache` with `LinearAttentionCacheLayerMixin.reorder_cache`, which `index_select`s the conv and recurrent states. With transformers' own `Mamba2ForCausalLM`, cached beam search becomes correct.
- **xLSTM beam search fails loudly.** 5.17.0 walks `ALL_CACHE_NAMES`, which includes `cache_params`. `xLSTMCache` has no `reorder_cache`, so the call raises `ValueError` instead of producing silently wrong beams. It is a fail-loud change, not a fix.

What 5.x does not fix:

- **#136, Mamba-2 gated RMSNorm.** `MambaRMSNormGated.forward` is byte-identical in 4.57.3 and 5.17.0. It takes `mean(-1)` over all channels with no grouping. The FLA-class workaround (`fft_rollout/hooks/sitecustomize.py`) stays necessary.
- **#137, xLSTM attention mask.** `modeling_xlstm.py` in 5.17.0 has zero references to `attention_mask`. Batch-1 xLSTM decoding stays necessary.
- **#143, xLSTM kernel defaults.** `xLSTMConfig.chunkwise_kernel` still defaults to `"chunkwise--native_autograd"` in 5.17.0.

What 5.x breaks in paper-branch code:

- `src/main/sallm/training/trainer.py:252,579` read `self.tokenizer`. transformers 5 moved it to `processing_class`, and main fixed this in `b439481`.
- The frozen MzansiLM 125M config (`hidden_size=512`, 9 heads) is rejected. Main has `models/llama_compatibility.py` and `tests/test_llama_compatibility.py`, confirmed in the audit note of 30 Aug.
- FLA 0.5.1 cache (see table) for GDN and FLA Mamba-2.
- The default `from_pretrained` dtype becomes `auto` (v5 notes). Any load that relies on the old fp32 default changes precision. The rollout passes an explicit dtype per architecture; other runners are unverified.
- Minor: `use_auth_token` removed, `logging_dir` is tensorboard-only, Python >=3.10, hub >=1.5 <2.

## The main vs paper divergence

- **Confirmed:** on 30 Aug, main moved to `transformers>=5.5.0,<6` (locked 5.15.1), `lm-eval>=0.4.12`, hub 1.29.0 and `protobuf==5.29.6`. The commits are "Upgrade ML dependencies and restore security scan" and "Remove vulnerable lm-eval cache dependency", with a Grype policy. The paper branch stayed on 4.57.3 / 0.4.9.2 / hub 0.36.0 and still carries the FLA 0.5.1 `pure-gdn` extra that main lacks.
- **Inferred merge consequences:**
  1. Taking main's `pyproject`/`uv.lock` silently swaps the runtime that produced every paper number. `uv sync` after the merge would no longer reproduce the paper.
  2. The paper code needs the three fixes above: `processing_class`, keeping main's LLaMA compatibility class, and FLA >= 0.5.2 re-added as an extra.
  3. The beam policy in `fft_rollout/rollout.py` (`BEAM_MODE`, all non-Transformer architectures cache-free) is written against 4.57.3 behaviour. Under 5.x it is still correct for GDN and FLA Mamba-2, while xLSTM would raise instead of mis-decoding. The comments would be stale.
  4. lm-eval 0.4.12 scores Belebele and other tasks against newer task configs, so lm-eval numbers are not comparable to the 0.4.9.2 paper numbers without a parity run.
  5. datasets goes from 4.8.5 to 4.4.2.
- **Recommendation:** before merging, tag the paper commit together with its `uv.lock` (for example `paper-runtime-2026-10`). State in `scripts/paper/README.md` that the paper numbers require that lock. Merge the code, and treat any re-scoring on the main stack as a new measurement, not a reproduction.

## AfriXNLI note (confirmed)

lm-eval #3944 fixes single-brace `{premise}`/`{hypothesis}` in all 35 AfriXNLI `prompt_1` YAMLs, not only English. The paper uses prompt_1 only for eng. `data/base-official.csv` maps sot to prompt_5, xho to prompt_3 and zul to prompt_4, and #3944 says prompts 3-5 were already correct. The paper's exclusion of English AfriXNLI is consistent. Upgrading to 0.4.13 would change eng numbers only. Main's 0.4.12 lacks the fix.

## Validation plan (for any upgrade, run old and new stack side by side)

1. **Freeze baselines on the current stack first.** Save logits for 64 fixed prompts per architecture from one saved checkpoint each: MzansiLM, Mamba-2 (FLA class), GDN, xLSTM.
2. **Kernel parity.**
   - Mamba-2: FLA `Mamba2ForCausalLM` vs transformers `Mamba2ForCausalLM` on the same checkpoint. The difference is expected to be non-zero because of #136. Record it, then patch the grouped norm in a scratch copy and require max |Δlogit| < 1e-3 (fp32).
   - GDN: FLA chunk kernel vs recurrent/naive path, bf16 tolerance about 2e-2.
   - xLSTM: triton chunkwise vs `native_autograd`.
   - Run all of these on the old stack and again on the new one. Cross-stack drift in the saved logits must stay within the same tolerance.
3. **Causality check.** Per architecture, perturb token t+k (k >= 1) and assert logits at positions <= t are bit-identical (fp32) or within 1e-5. Run on both the kernel path and the cache/step path, plus a cached vs uncached greedy decode equality check.
4. **CPU smoke.** `pytest -q` in both trees. Main already has `test_trainer_compatibility.py`, `test_llama_compatibility.py` and `test_lm_eval_integration.py`. Add a test that FLA's cache instantiates under transformers 5.x.
5. **GPU smoke per architecture.** 50-step LoRA fine-tune plus greedy and 5-beam generate on HEX L40S and Kombuys RTX 5090. Assert the fast path is active (mamba_ssm asserted, `FLA fast path available`, xLSTM kernel names logged per #143). Assert PEFT still accepts the protocol targets. Mamba-2 `out_proj` fails on every PEFT version checked.
6. **Paper-number reproduction** (from `results/results.csv` in the paper manuscript repository at commit `03be42d`). Rescore stored checkpoints with the as-run runners on both stacks. Tolerance: identical on the old stack; on the new stack ±0.5 points for chrF/F1/accuracy, and any larger shift explained:
   - GDN T2X xho General chrF 46.7431
   - Mamba T2X xho General chrF 42.1894
   - Transformer T2X xho General chrF 28.9377
   - xLSTM AfriHG zul General chrF 15.9246
   - GDN News eng General F1 89.6271
   - xLSTM SIB-200 xho General F1 56.2800
   - Mamba Belebele zul Base acc_norm 27.4444
   - GDN POS zul General acc 74.7670

## Recommended sequencing

1. **Now to 12 Oct 2026:** change nothing. The rollout (ending ~1 Oct) and the paper (due 12 Oct) stay on the 4.57.3 / FLA 0.5.1 / lm-eval 0.4.9.2 / PEFT 0.18.1 runtime. Do only read-only prep: tag the paper runtime and write the parity and causality scripts.
2. **After 12 Oct:** merge paper into main on a branch. Apply the three transformers-5 code fixes, pin FLA 0.5.2, and keep torch 2.9.1 / triton 3.5.1 / mamba-ssm 2.3.2.post1 / causal-conv1d 1.6.2.post1. Run steps 1-6.
3. **Then, one at a time with parity each time:** transformers 5.17 (brings tokenizers 0.23), lm-eval 0.4.13, xlstm / mlstm_kernels 2.0.6, datasets / trl (together, because trl 1.x needs datasets>=4.7).
4. **Last and separate:** torch >2.10 / triton 3.8. This needs from-source mamba-ssm and causal-conv1d builds on both clusters. Hold huggingface-hub 2.0 until transformers allows it.
5. **Independent of versions:** #136, #137, #143 and the PEFT Mamba `out_proj` block are not fixed by any available release. They need code changes in SALLM.
