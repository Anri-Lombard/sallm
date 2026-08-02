# SALLM advisor meeting update - 2026-06-25

Coverage window: Thursday 2026-06-18 through Wednesday 2026-06-24.

## Short version

This week produced one defensible result family: xLSTM POS improved substantially after the constrained-label HPO path was fixed and evaluated on held-out MasakhaPOS test splits. The broader xLSTM HPO pipeline is now repaired enough to run non-POS sweeps, and selected adapters have validation/staging evidence for NER, SIB, T2X, News, InjongoIntent, and AfriHG, but the final held-out test wave is still pending in the L40S queue.

The headline technical message for the meeting is:

- POS is now a defensible xLSTM result, not the old bugged result.
- Mamba remains clearly bad on corrected POS and diagnostic NER.
- xLSTM HPO helps POS and may help some other tasks, but non-POS claims should wait for the held-out jobs `952247`-`952252`.
- The next architecture direction should stay narrow: finish xLSTM HPO/test closeout first, then prioritize HGRN2/BabyHGRN as the stronger low-resource/sample-efficiency candidate. Gated DeltaNet is interesting, but only as a small feasibility screen, not the next full SALLM pretraining run.

## Defensible POS results

The corrected POS evaluation now uses closed-label token-logprob scoring over MasakhaPOS labels, with tuple prompts and held-out `test` splits for Tswana, Xhosa, and Zulu. This avoids the earlier invalid list-valued target bug that inflated old xLSTM POS numbers.

Final promoted multilingual xLSTM HPO POS result:

- Job: `937140`
- Artifact: `outputs/hex_results/pos/xlstm_pos_hpo_best_test_20260617/xlstm_pos_hpo_best_test.json`
- Protocol: `masakhane/masakhapos`, split `test`, languages `tsn/xho/zul`, closed-label token-logprob, mean label logprob
- Macro token accuracy: `0.7808962457`
- Micro token accuracy: `0.7824370992`
- Exact sequence accuracy: `0.0379814097`
- Tswana token accuracy: `0.7905741410`
- Xhosa token accuracy: `0.7666170889`
- Zulu token accuracy: `0.7854975070`

This improves clearly over the corrected xLSTM multilingual POS baseline:

| Model/result | Tsn | Xho | Zul |
| --- | ---: | ---: | ---: |
| LLaMA corrected POS | `0.8460` | `0.8220` | `0.8409` |
| xLSTM corrected baseline | `0.7245` | `0.6933` | `0.7313` |
| xLSTM HPO-selected multilingual | `0.7906` | `0.7666` | `0.7855` |
| Mamba corrected POS | `0.2340` | `0.0674` | `0.0402` |

Interpretation: xLSTM is still below LLaMA, but the corrected HPO result is real progress. Mamba is not merely slightly worse; on corrected POS it is near-collapse, especially on Xhosa and Zulu.

## POS wave2 monolingual checks

After the multilingual HPO result, selected monolingual adapters were tested on held-out MasakhaPOS.

| Job | Adapter/eval | Token accuracy | Exact sequence | Interpretation |
| --- | --- | ---: | ---: | --- |
| `941184` | Tswana mono | `0.7650477913` | `0.0228405316` | Valid, but below promoted multilingual Tswana `0.7906`; do not promote over multilingual. |
| `941185` | Xhosa mono | `0.7838752693` | `0.0611480865` | Improves over multilingual Xhosa `0.7666`; useful result. |
| `941135` | Zulu mono | `0.8200737047` | `0.1089850250` | Best xLSTM Zulu POS result so far; closer to LLaMA Zulu `0.8409`. |
| `941186` | Wave2 multilingual | macro `0.7676006651`, micro `0.7638425375` | `0.0420093956` | Valid but lower than promoted multilingual HPO macro `0.7809`; record as not promoted. |

This gives a nuanced answer to advisors: monolingual HPO helped Xhosa and Zulu, but not Tswana. The best POS path is not uniformly monolingual or multilingual; it depends on language/task data.

## POS generalization attempt

The broader/general POS HPO path did not produce a better result.

- Initial general POS HPO job `937356` hit xLSTM chunking problems: sequence length not divisible by chunk size `64`.
- Repaired general POS HPO job `941187` later timed out after `1-12:00:19`, reaching about `873/1023` steps in epoch `2.55`.
- Existing validation artifact was weak: macro/micro token accuracy around `0.6729`/`0.6721`, exact around `0.0078`.

Interpretation: this is not a promotable result. It is useful evidence that general/mixed POS is harder and that the cleanest evidence remains the constrained multilingual and selected monolingual POS runs.

## HPO and scorer engineering

Two important engineering repairs landed this week.

First, constrained label scoring was sped up:

- File: `scripts/constrained_label_scoring.py`
- Smoke check: `scripts/check_constrained_label_scoring_fastpath.py`
- Behavior: if every candidate label is one tokenizer token, score all labels from a single next-token logits vector; fall back to the slower multi-token path when needed.

This matters because POS closed-label evaluation became fast enough to run more reliably across languages and prompts.

Second, generic xLSTM HPO was repaired in `src/main/sallm/hpo/trial.py`:

- Adds/defaults `training.pad_to_multiple_of=64` for xLSTM generic sweeps.
- Defaults `training.show_completions=false` when omitted, avoiding generation-time chunk issues during training callbacks.
- Saves final adapters for HPO runs via `SALLM_SAVE_FINAL_ADAPTER_FOR_HPO=1`.
- Disables Hub pushes for trial configs.
- Accepts OmegaConf `DictConfig` for run-id output/log path handling.

Before that repair, non-POS smoke jobs such as `941190` and `941191` could finish SLURM as `COMPLETED 0:0` while still producing malformed/no-result runs because of xLSTM chunk assertions like:

- `Sequence length 255 is not divisible by chunk size 64`
- `Sequence length 276 is not divisible by chunk size 64`

After the repair, the generic HPO path is usable enough for selected-adapter experiments, but it still needs held-out test confirmation before we make final task-level claims.

## Non-POS xLSTM HPO status

The non-POS HPO wave made operational progress, but most results are not yet final claims.

Completed HPO / selected-adapter evidence:

- NER all-language HPO job `944325` completed cleanly in `09:51:47`.
  - Best observed validation loss came from trial/run `4vfuoadl`.
  - Validation losses observed included `1.4910610156431516`, `1.5955722340863876`, `1.5314080334064242`, `1.5314886383850779`, `1.5260831981786565`, `1.5691551860823507`.
  - Selected/eval staging showed best F1: `tn 0.2510`, `xh 0.2072`, `zu 0.2443`.
  - Caveat: the synced NER artifacts were validation tasks, so this is not yet a defensible held-out NER result.

- SIB all-language HPO job `944326` completed cleanly in `07:18:48`.
  - Best observed validation loss came from trial/run `k6ob2x17`.
  - Validation losses observed included `1.988075066254998`, `2.094855229541509`, `2.1336396599458123`, `2.2179170036958125`.
  - Selected/eval staging best F1 by language: `afr 0.1255`, `eng 0.1497`, `nso 0.0754`, `sot 0.1043`, `xho 0.1144`, `zul 0.0879`.
  - These are low absolute F1s, so even if held-out confirms them, SIB remains weak.

- T2X Xhosa selected-adapter staging found candidate `hzumyec4` as best among the tested HPO candidates.
  - Candidate chrF scores: `qu2h640x 29.0187`, `py4owm99 27.0189`, `hzumyec4 30.6990`, `2cfkgy1o 27.7861`.
  - Caveat: this is below the prior official xLSTM T2X Xhosa test result around chrF `34.1600`, and far below LLaMA-level performance. Final held-out confirmation is still pending.

- InjongoIntent HPO job `948169` completed cleanly in `16:24:12`.
- MasakhaNews HPO job `948170` completed cleanly in `12:27:18`.
- AfriHG HPO job `948171` completed cleanly after a long run of `1-17:09:06`.
  - AfriHG logs showed repeated context truncation warnings and automatic batch-size/OOM fallback from `64` to `32` during generation metrics.
  - SLURM still completed cleanly, but this is a warning sign to inspect output quality carefully.

Local compact eval output was synced to:

- `outputs/eval/xlstm_hpo_selected_20260622/`

## Held-out test wave pending

The key next evidence is already queued. These jobs were submitted on L40S with explicit `--account=l40sfree` after an initial account/partition mismatch.

Manifest:

- `/home/lmbanr001/masters/sallm/outputs/final_submissions/xlstm_hpo_test_20260624.csv`

Eval output root:

- `/scratch/lmbanr001/masters/sallm/results/eval/xlstm_hpo_test_20260624`

Queued held-out jobs:

| Job | Task | Selected adapter/run |
| --- | --- | --- |
| `952247` | SIB all | `k6ob2x17` |
| `952248` | MasakhaNews all | `7mq0snuc` |
| `952249` | InjongoIntent all | `imrufs4l` |
| `952250` | AfriHG all | `dbi2wg9w` |
| `952251` | T2X Xhosa | `hzumyec4` |
| `952252` | MasakhaNER all | `4vfuoadl` |

As of the latest check on 2026-06-24 evening, all six were still `PENDING` for `Priority`; no logs or artifacts yet.

Important NER defensibility fix: the stock `masakhaner_all.yaml` points at validation-style tasks. For this final wave, temporary scratch lm-eval YAMLs were generated under:

- `/scratch/lmbanr001/masters/sallm/tmp_eval_tasks/masakhaner_test_20260624`

Those copied the prompt definitions but changed the split/task suffix to use held-out `*_test` tasks. This should let us answer advisors cleanly if the NER result is still bad: we are not only reporting validation behavior or a broken task split.

## NER and Mamba interpretation

The prior diagnostic NER conclusion still stands unless the pending held-out xLSTM HPO NER job changes it.

Diagnostic constrained BIO tag-sequence MasakhaNER did not rescue any architecture:

- Mamba macro entity F1: `0.0119`
- LLaMA macro entity F1: `0.0353`
- xLSTM macro entity F1: `0.0367`

Failure signatures:

- Mamba overpredicts date labels.
- LLaMA overpredicts `b-per`.
- xLSTM overpredicts repeated inside labels such as `i-loc`, `i-per`, and `i-org`.

Interpretation for advisors: NER is not just failing because of one output format. We simplified the task to constrained BIO label sequences and it still performed badly. That makes the negative result more defensible, but the final answer should wait for pending held-out HPO NER job `952252`.

## Architecture research

Two strategy notes were created this week:

- `sallm_memory/gated_deltanet_research_2026-06-21.md`
- `sallm_memory/latest_architecture_research_2026-06-21.html`

Main conclusion:

Do not spend the next full SALLM run on Gated DeltaNet yet. Gated DeltaNet is a reasonable small feasibility screen because it combines Mamba2-style decay with DeltaNet-style targeted memory updates, and recent hybrid linear-attention work is moving in that direction. But there is no direct South African-language or low-resource morphology evidence yet, and the strongest sample-efficiency candidate still looks like HGRN2/BabyHGRN.

Practical recommendation:

- Finish xLSTM HPO/test closeout first.
- If adding one full architecture next, prioritize HGRN2/BabyHGRN.
- If testing GDN, keep it small and FLA-based, with a clear stop condition.

## Operational notes

Scratch quota was a real risk this week.

- Scratch reached about `95.4%`, later up to about `97.6%`.
- Cleanup removed only re-creatable/non-evidence paths such as HF cache, Triton cache, tmp/cache directories, old local env/cache material, and a duplicate scratch workspace.
- Checkpoints and results were intentionally not deleted.
- Scratch later sat around `93.7%` after the new held-out wave.

There were also repeated VPN/HEX SSH visibility gaps. These caused monitoring delays but are not evidence of experiment failure.

## What to ask advisors

1. Is the corrected POS story strong enough to include as a main result: xLSTM HPO improves corrected POS, especially Xhosa/Zulu monolingual, but LLaMA remains stronger?
2. If the pending held-out HPO tests remain weak on NER/SIB/T2X, should we frame this as an xLSTM limitation on structured generation/sequence-labeling, or as evidence that the task formulation/eval setup still needs more work?
3. For the next architecture, should we spend the next full run on HGRN2/BabyHGRN and only do a small GDN screen, or should GDN move higher despite weaker low-resource evidence?
4. How aggressive should scratch cleanup be before launching another architecture wave?

## Immediate next steps

1. Monitor jobs `952247`-`952252`.
2. Sync artifacts from `/scratch/lmbanr001/masters/sallm/results/eval/xlstm_hpo_test_20260624` when jobs complete.
3. Confirm every result uses held-out test splits and the selected adapter listed in the manifest.
4. Update Google Sheet and `sallm_memory/sallm_progress.md` only for final, defensible held-out results.
5. Prepare side-by-side examples for AfriHG, T2X, NER, and POS if the jobs complete before the meeting.
