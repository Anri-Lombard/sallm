# Making the Qwen3.5 isiZulu model substantially better

## Abstract

The current continual-pretraining recipe is a good control, but it should not become the default for every later run. The matched 2B and 4B experiments show that more MzansiText exposure keeps lowering isiZulu validation perplexity after several downstream tasks have flattened or worsened. The live 9B run must therefore be selected by a checkpoint sweep, not by its final loss.

The next useful pretraining experiment is a controlled data-admission study on the 4B model. HPLT 3.0 and FineWeb2 contain enough isiZulu to matter, but both overlap heavily with the Common Crawl sources already represented in MzansiText. Each source should first be cross-deduplicated against MzansiText and the evaluation corpora. Three equal-token continuations can then compare current MzansiText, high-quality residual HPLT, and filtered residual FineWeb2. A source that passes at 10M tokens should continue to 50M before it earns a larger run. This design costs hours, not days, and directly tests whether new text is better than another pass over the old text.

The evidence does not support tokenizer replacement, LoRA as the main CPT method, a longer context window, a large replay grid, or a blind machine-translated data dump. A small learning-rate check is worthwhile after the source mixture is chosen. Instruction tuning should follow checkpoint selection and remain a separate comparison. The base CPT checkpoint is the scientific result; the instruction-tuned checkpoint is a derivative model.

## 1. Questions

This review answers three questions.

1. What additional data or mixture is most likely to improve isiZulu without needless duplication or general-capability loss?
2. Which training changes are worth GPU time for the Qwen3.5 2B, 4B, and 9B line?
3. What evidence should be complete before reporting the model to Jan or starting instruction tuning?

The target is an isiZulu-first model. English replay is a retention control, not a co-equal objective. Broader reasoning data is optional unless the project scope expands beyond language competence.

## 2. Method

The review combines four evidence types.

- Project evidence: pinned training reports, manifests, validation JSON files, checkpoint sweeps, and the live AMD-host state.
- Direct model-family evidence: Qwen3.5-4B Base results in AfriqueLLM [1].
- Continual-pretraining evidence: replay, rewarming, repeated data, low-rank adaptation, mixture selection, and tokenizer adaptation studies [2-8].
- African-language evidence: official corpus cards and benchmark papers [9-29].

Primary papers and official dataset metadata were checked through 30 August 2026. Dataset counts retain the unit used by the source because HPLT, FineWeb2, MzansiText, and MADLAD use different tokenizers or report words, characters, or documents. Any candidate entering training must be retokenized with the pinned Qwen tokenizer.

The recommendations use these confidence labels.

- High: direct project evidence or a closely matched Qwen3.5 result.
- Moderate: repeated results across related decoder models, with a material scale or language difference.
- Proposed: a controlled experiment designed for this project, not a settled literature result.

## 3. What the current experiments already establish

### 3.1 The recipe and provenance are sound

The completed 2B and 4B runs used the same immutable block order and tokenizer, with exactly 436,409,318 isiZulu tokens and 48,489,924 decontaminated English replay tokens. The total dose was 484,899,242 tokens. Both used full-parameter training, sequence length 2,048, an effective sequence batch of eight, learning rate `1e-5`, cosine decay, 3% warmup, AdamW, and Liger kernels. The 9B run changes only the model and the physical microbatch needed to fit memory. Held-out tests remain untouched.

This is a strong matched scaling study. Changing the active 9B run would weaken it and waste the completed half-run.

### 3.2 More exposure is not the same as a better checkpoint

The 4B checkpoint curve is the clearest project result.

| Training dose | isiZulu PPL | isiZulu SIB-200 | WikiText PPL | English SIB-200 | AfriXNLI |
|---:|---:|---:|---:|---:|---:|
| 50M | 14.39 | 49.90% | 10.82 | 57.17% | 37.33% |
| 150M | 12.56 | 49.90% | 10.86 | 57.37% | 37.33% |
| 250M | 12.24 | 49.09% | 10.93 | 54.75% | 37.33% |
| 450M | 12.18 | 48.08% | 10.93 | 55.76% | 36.89% |
| 484.9M | 12.18 | 47.68% | 10.93 | 56.16% | 36.67% |

Perplexity improves sharply through about 150M to 250M tokens, then changes very little. The downstream measures do not follow it. This is not evidence that later training is useless. It is evidence that the model is learning the text distribution after the measured transfer gains have saturated.

Perplexity can track in-context task performance within a fixed model and training trajectory, but it can also conceal sequence-level behavior [28]. The project results are a direct example of that limit.

The corrected broader validation supports the 150M checkpoint as the present balanced 4B choice.

| 4B model | AfriQA EM | AfriQA F1 | MasakhaNER mean F1 | MasakhaPOS accuracy |
|---|---:|---:|---:|---:|
| Original base | 41.80% | 52.32% | 20.95% | 64.67% |
| CPT 150M | 52.46% | 65.89% | 17.61% | 70.07% |
| CPT 250M | 49.59% | 63.09% | 16.91% | 69.49% |
| CPT 450M | 49.59% | 62.99% | 17.17% | 69.53% |

The result is mixed. AfriQA and POS improve, while NER falls. That pattern argues for multi-axis selection and later task adaptation, not for selecting the last checkpoint.

### 3.3 Model size helps, but does not remove the need for selection

At the matched 150M checkpoint, 4B is much stronger than 2B on most downstream tasks. The 2B run still improved target-language modelling, SIB-200, AfriMMLU, InjongoIntent, and POS, but AfriQA and NER fell. Its isiZulu PPL moved from 38.60 to 15.76 and SIB-200 from 35.76% to 41.62%. The 4B model reached PPL 12.56 and stronger absolute downstream results at the same dose.

The 9B 10M pilot also warned against assuming that size settles the question. Its isiZulu PPL improved from 21.58 to 16.46, but SIB-200 fell from 53.13% to 51.31% and AfriMMLU fell from 48.19% to 46.99%. InjongoIntent and AfriQA improved. The full 9B curve may differ, so the pilot is neither a rejection nor proof of superiority.

### 3.4 Live 9B state

At 13:46 SAST on 30 August 2026, the full 9B run was healthy at step 14,737 of 29,596 and 241,449,386 of 484,899,242 tokens. Sustained throughput was 4,472.9 tokens per second. The GPU was 99% busy and used 223.5 GB of 308.8 GB VRAM. Losses were finite, four checkpoints existed, and no final report or error was present. The measured remaining training time was about 15.1 hours.

The 9B run should finish unchanged. Its validation sweep must compare the original base, intermediate checkpoints, and final checkpoint under the same prompts and evaluator.

## 4. What the closest published work changes

AfriqueLLM is the closest direct precedent because it includes Qwen3.5-4B Base [1]. It reports an African-language mean increase from 46.01 to 57.12 after full CPT on a 25.2B-token mixture. That mixture contained about 22.8B multilingual text tokens, roughly 1B code tokens, 1B math tokens, and 324M translated synthetic tokens. The raw Zulu allocation was about 350M tokens and was repeated to about 1.07B effective tokens. An extended code and math version scored 58.26.

This is useful evidence for three decisions.

1. Full-parameter CPT on Qwen3.5-4B is viable.
2. Data composition matters enough to test directly.
3. Repeated low-resource text can still help when the mixture and evaluation are broad.

It does not prove that this project should copy the recipe. AfriqueLLM trained for about 25B tokens with a global batch near 4M tokens and much broader multilingual objectives. Our run uses a 16,384-token effective batch and an isiZulu-first objective. Its code and math extensions also add tokens, so token count and composition are partly confounded. The paper's selected `5e-5` learning rate is a candidate for a small pilot, not a safe replacement for the current rate.

Other CPT results sharpen the boundary.

- Modest replay repeatedly reduces forgetting. One study found useful effects from 1% to 10% replay and adaptation loss at 50% replay [2]. The present 10% English mixture is defensible.
- Rewarming and decay can support distribution shifts, but the best peak rate depends on scale and training horizon [2, 3].
- Repeating constrained data for several epochs can remain useful for language-model loss [4]. Our downstream curve shows why this cannot be treated as proof of task improvement.
- Standard LoRA learns less of the new distribution than full finetuning, although it also forgets less [5]. Full CPT should remain the main line.
- Bilingual and code mixtures can improve classification while increasing language mixing or harming generation [6]. Generation checks are required before adding broad capability data.
- Data selection based on similarity to a clean target distribution can beat uniform sampling [7]. This supports quality weighting of supplemental isiZulu rather than raw concatenation.
- Vocabulary replacement helps when tokenization is badly mismatched, but it changes the embedding space and breaks the current model-family comparison [8]. It should be considered only after a token-fertility audit shows a real problem.

## 5. The data decision

### 5.1 MzansiText remains the anchor, not the whole answer

MzansiText already combines WURA, mC4, CulturaX, Glot500-c, Inkuba-Mono, CC100, ParaCrawl, NCHLT, and other material [9]. Adding those releases separately would mostly test deduplication. The current artifact contains 436.4M Qwen tokens after project-specific preparation and benchmark decontamination. The original card's smaller token figure uses a different tokenizer, so the counts are not contradictory.

Exact deduplication is necessary but not enough. Public MzansiText samples include noisy and wrong-language rows, and its upstream sources have different quality profiles. A quality-weighted MzansiText subset may be more useful than another complete pass.

### 5.2 HPLT is the first serious challenger

HPLT 3.0 reports 336,440 `zul_Latn` documents, 410.54M Gemma-3 tokens, 8.02M segments, and 1.12B characters [10]. Its crawls span 2012 to 2024. It provides document and segment language predictions, web-register labels, provenance, global per-language deduplication, MinHash cluster sizes, and Web Docs Scorer quality bins.

Its headline count is not a count of new tokens. HPLT draws from Common Crawl and Internet Archive, while MzansiText already contains several Common Crawl derivatives. HPLT earns a training allocation only from high-quality residual documents left after union-level deduplication.

### 5.3 FineWeb2 is a useful independent challenger

The official viewer currently reports 127,335 filtered `zul_Latn` training documents and 762 test documents [11]. FineWeb2 uses GlotLID, per-language filtering, global MinHash deduplication, and metadata across 96 Common Crawl snapshots. The test split and all benchmark-like material stay out of training.

FineWeb2 removed far more Zulu documents than it retained. Its own work also shows that aggressive multilingual filtering can discard useful low-resource text [11]. Use the filtered train slice first. Do not admit the removed slice without fluent auditing.

### 5.4 Small curated South African data may have high marginal value

The new SA-Knowledge release contains government-domain text for isiZulu, isiXhosa, Sepedi, Sesotho, and English [12]. Its isiZulu monolingual file is 5,034,076 bytes over 27,987 lines. It is small, but it has formal South African vocabulary, document-level language checks, and clearer provenance than generic web text. The repository uses mixed licensing. The monolingual and parallel government text is CC BY 4.0 subject to source terms.

This source should be audited for overlap with NCHLT and government text already present in MzansiText. Its CommonsenseQA, OpenBookQA, hallucination, and generic-probe subsets are evaluation data and must not enter CPT.

### 5.5 MADLAD is a residual source, not a default ingredient

MADLAD-400 clean reports about 53.8K isiZulu documents, 1.2M sentences, and 27.2M whitespace tokens [13]. It is another Common Crawl view and predates some HPLT and FineWeb2 material. Only a clean residual that survives cross-source deduplication is worth considering.

### 5.6 Conversational transcripts belong later

The Way With Words isiZulu dataset contains 50 hours of simulated call-centre speech, 202 calls, and 63 speakers across retail, debt collection, insurance, and travel [14]. It is a useful lead for conversational register and may be the corpus Jan heard about, but that is unconfirmed. It has a custom gated licence and no predefined split. It is also transcript text, not instruction-response data.

The sensible path is to ask Jan for the exact dataset and licence. If it is Way With Words, keep speakers and calls disjoint across splits, confirm that language-model training is permitted, and treat it as a small later CPT or dialogue-SFT component. It should not delay the current 9B run.

## 6. The next pretraining experiment

### 6.1 First build a union-level admission manifest

For MzansiText, HPLT, FineWeb2, MADLAD, and SA-Knowledge, record:

- pinned revision and licence metadata;
- input documents, characters, and Qwen tokens;
- exact document and normalized-URL overlap;
- MinHash or character n-gram near-duplicate clusters;
- document-level and sentence-level language scores;
- source, host, crawl date, register, and quality-bin distributions;
- benchmark overlap in isiZulu and source English;
- retained unique Qwen tokens after every filter.

The process must keep duplicate-cluster provenance so that the best copy survives. Document-level language identification should precede sentence filtering. Short Nguni sentences are easy to misclassify as isiXhosa or siSwati, and a sentence-only filter can remove valid isiZulu [12].

Deduplication is not administrative work. Near-duplicate removal has reduced memorized output, evaluation leakage, and wasted training in large language-model corpora [15].

### 6.2 Run a cheap reject-first continuation

Branch three 10M-token continuations from the same selected 4B CPT checkpoint. Use the same seed, optimizer, schedule, English replay, evaluation set, and total dose.

| Arm | isiZulu allocation | English replay | Purpose |
|---|---:|---:|---|
| Control | 9M current MzansiText | 1M | Measure the value of another old-data dose |
| HPLT | 4.5M MzansiText + 4.5M high-WDS residual HPLT | 1M | Test newer, quality-ranked web text |
| FineWeb2 | 4.5M MzansiText + 4.5M filtered residual FineWeb2 | 1M | Test a different filtering pipeline |

At the measured 4B throughput of 5,808 tokens per second, each 10M arm needs about 29 minutes of active training. All three require about 1.4 GPU hours before evaluation.

The 10M stage is a rejection gate. It is too small to prove that a source wins at scale. If one or both challengers beat the control on the preregistered validation scorecard without a material retention loss, continue the control and best challenger to 50M. Continuing two arms from 10M to 50M adds about 3.8 GPU hours. A persistent winner can then continue to 150M.

The earlier 2M-HPLT pilot does not settle this question. It used a small HPLT fraction and did not test a high-WDS, cross-deduplicated residual against a source-matched control.

### 6.3 Select on a scorecard, not one average

The admission gate should show each dimension separately.

- isiZulu modelling: clean validation perplexity and bits per byte.
- Understanding: SIB-200, AfriXNLI, AfriMMLU, and InjongoIntent.
- Structured language: MasakhaPOS and MasakhaNER with format-validity rates.
- Generative use: corrected AfriQA and bidirectional translation.
- Retention: WikiText perplexity and English SIB-200.
- Fluency: language fidelity, code-switching, repetition, and a later blind native comparison.

Use paired confidence intervals where the dataset size permits. For prompted tasks, report the mean and range across fixed templates rather than the best prompt. SIB-200 and FLORES share source material, so they should not be presented as two independent wins [16]. AfriMMLU is translated international knowledge, not South African cultural knowledge [17]. AfriQA must name whether it is retrieval, reader-only, or closed-book evaluation [18].

The held-out test remains sealed until the data recipe, checkpoint, prompts, parsers, and decoding settings are frozen.

## 7. Training changes worth testing

### 7.1 Keep the current mainline

The main control should remain full-parameter CPT with 10% English replay. The active 9B GPU is already near full utilization, and the 4B and 2B runs were stable. This is not the place to introduce a new optimizer stack or a parameter-efficient method.

### 7.2 Do a small learning-rate check after the data choice

AfriqueLLM selected `5e-5` in its setting [1], while the present recipe uses `1e-5`. Rewarming work shows a real plasticity-retention tradeoff at higher rates [2, 3]. Test `1e-5`, `3e-5`, and `5e-5` for 10M tokens on the best source mixture. Stop an arm immediately for non-finite loss, unstable loss, or clear retention failure. Promote at most the best two rates to the 50M gate.

This is a calibration, not a broad HPO programme. The model, mixture, dose, and schedule stay fixed. If `1e-5` remains on the Pareto frontier, keep it.

### 7.3 Do not raise English replay without evidence

Ten percent is within the best-supported modest replay range [2]. A single 20% arm is reasonable only if the selected 9B checkpoint has an unacceptable English-retention loss. Keep total tokens fixed, so more English necessarily means less isiZulu. There is no reason to run a 5%, 10%, 20%, 30% grid now.

### 7.4 Audit the tokenizer, then leave it alone unless it fails

Measure Qwen tokens per word, tokens per character, byte fallback, and long-tail morpheme fragmentation on matched isiZulu and English samples. If isiZulu is badly fragmented, a separate vocabulary experiment may be justified [8, 19]. Otherwise, tokenizer surgery would add risk and break the clean 2B, 4B, and 9B comparison.

### 7.5 Keep sequence length at 2,048 for the language run

AfriqueLLM found benefits from longer context on some reasoning tasks [1]. That does not establish a fluency gain, and most candidate documents are short. Longer context raises memory and compute cost. Revisit it only if long-document summarisation or retrieval becomes a declared target.

### 7.6 Treat code and math as an optional model variant

AfriqueLLM provides a plausible reason to test a small decontaminated code and math allocation [1]. Rethinking Multilingual CPT warns that such data can help classification while hurting generation or language fidelity [6]. Since isiZulu is the primary target, code and math should not displace the first source-quality experiment.

If Jan wants a broader reasoning model, replace about 8% to 10% of the fixed token budget with high-quality code and math and compare it with the isiZulu-first model. Do not append extra tokens and call the difference a mixture effect.

## 8. Instruction tuning after CPT

Instruction tuning should start only after the CPT checkpoint is selected and its base-model scorecard is frozen. This preserves the causal story: CPT changes language and domain modelling, while SFT changes how the model exposes those capabilities.

The existing Aya experiment already proves that a small human isiZulu set can change behavior. After filtering, it retained 1,659 human rows. On the 150M 4B checkpoint, SFT raised AfriMMLU from 36.14% to 39.76%, InjongoIntent from 37.19% to 40.31%, and the limited NER F1 from 4.74% to 19.37%. AfriQA F1 fell from 65.89% to 63.99% and POS fell from 69.73% to 68.13%. SFT helps some interfaces and hurts others. It needs a control and a retention gate.

The clean comparison is a 2 by 2 design.

| Base weights | No SFT | Same SFT recipe |
|---|---|---|
| Original model | Original base | Original base plus SFT |
| Selected CPT model | CPT base | CPT plus SFT |

The instruction corpus should be compact, native-written where possible, benchmark-clean, and varied across speakers and tasks. Aya is a sensible seed [20], but its isiZulu human annotations have a narrow contributor base. The large Aya Collection and community Alpaca translations are useful only as audited supplements, not wholesale imports. AfriInstruct and Inkuba-Instruct contain several of the same task datasets used for evaluation, so their full mixtures would compromise this protocol [21].

The conversational data Jan mentioned is likely the highest-value missing input. It should be checked for native authorship, licence, turns, speaker diversity, task coverage, benchmark overlap, and train-validation separation before it is used.

Preference tuning can wait. It needs enough native-speaker comparisons to define a preference target. A small blind evaluation is more useful first.

## 9. Minimum defensible validation before release

For the selected 9B checkpoint and the final 4B candidate, compare the untouched base, intermediate CPT, selected CPT, and final CPT under the same evaluator.

1. Clean isiZulu and English perplexity.
2. SIB-200 with accuracy, macro-F1, prompt range, and prediction distribution [16].
3. AfriXNLI with accuracy, macro-F1, confusion matrix, and neutral-collapse rate.
4. AfriMMLU by subject, with uncertainty [17].
5. Injongo intent accuracy and slot F1 [22].
6. MasakhaNER and MasakhaPOS with task metric and output-validity rate [23, 24].
7. AfriQA with the evaluation condition stated [18].
8. English-to-isiZulu and isiZulu-to-English translation using corrected isiZulu FLORES or a clean Autshumato split [25].
9. A small private generation set covering explanation, summarisation, rewriting, factual QA, functional writing, and dialogue.

NGLUEni provides useful Nguni generation tasks, but its isiZulu generation coverage is too small to replace a fresh set designed for this model [29].

Prompted classification and free generation are different interfaces. AfroBench uses both and reports prompt variation [26]. A base model can hold useful representations while failing an output format. For NER, POS, and intent, a fixed small task-adapted probe would help separate representation quality from prompting failure.

Human review is expensive, so use it only after automatic evaluation narrows the field to two candidates. A 50-prompt blind pairwise comparison can score grammaticality, naturalness, semantic adequacy, unwanted code-switching, and instruction following. Use at least two fluent reviewers and adjudicate disagreements. An LLM judge should not decide isiZulu fluency because African-language metric correlations remain uneven [27].

## 10. Ranked work plan

| Priority | Action | GPU cost | Decision produced |
|---:|---|---:|---|
| 1 | Finish the active 9B run unchanged | About 15.1 hours remaining at the live check | Complete matched size experiment |
| 2 | Run a staged 9B checkpoint sweep | Evaluation only | Select 9B by validation, not final loss |
| 3 | Build the cross-source admission manifest | Mostly CPU and storage | Unique, legal, benchmark-clean candidate pool |
| 4 | Run 10M Mzansi, HPLT, and FineWeb2 branches on 4B | About 1.4 training hours total | Reject weak supplemental sources cheaply |
| 5 | Continue control and winner to 50M | About 3.8 additional training hours | Check whether the source ranking persists |
| 6 | Calibrate `1e-5`, `3e-5`, and `5e-5` on the winning mixture | About 1.4 training hours for 10M arms | Small LR decision |
| 7 | Run the 2 by 2 SFT comparison | Small relative to CPT | Separate language learning from instruction behavior |
| 8 | Native blind review of the final two models | Human time | Direct fluency evidence |

The first five items can finish without touching held-out tests. A second full 485M-token MzansiText pass is not recommended. If the source challengers fail, stop the web-data search and invest in native conversational and instruction data.

## 11. What not to do

- Do not interrupt or reconfigure the active 9B run.
- Do not choose the final checkpoint because it has the lowest loss.
- Do not merge MzansiText, HPLT, FineWeb2, and MADLAD without union-level deduplication.
- Do not assume that 410M HPLT tokens are 410M new tokens.
- Do not add more English unless retention evidence calls for it.
- Do not use LoRA as the main maximum-acquisition CPT method.
- Do not replace the tokenizer without measured fragmentation.
- Do not use evaluation-derived instruction corpora such as the full AfriInstruct or Inkuba-Instruct mixtures.
- Do not use the held-out test to choose data, learning rate, prompt, checkpoint, or SFT recipe.
- Do not claim fluency from perplexity or classification alone.

## 12. Strongest objections and remaining uncertainty

### A 10M source test may mis-rank the long run

Correct. The 10M stage is allowed to reject a clearly weak source, not to certify a winner. A passing source must survive 50M before it earns a larger allocation.

### HPLT and FineWeb2 may add almost no unique text

That is plausible. If cross-deduplication leaves little clean residual text, the audit has answered the question without spending GPU time. SA-Knowledge and conversational data then become more valuable because they add register and provenance rather than bulk.

### The 9B model may fall outside the project limit

Jan cited a 1B to 7B target range and said the actual restriction needed clarification. The 9B study is still useful science, but 4B should remain the release candidate until Jan confirms 9B eligibility.

### The current validation tasks are too prompt-sensitive

Some are. The corrected AfriQA result and NER format-validity swings show this directly. Constrained scoring, prompt ranges, fixed probes, translation, and native review reduce this risk. They do not make every metric commensurate, so the report should keep dimensions separate.

### The literature mostly studies multilingual rather than isiZulu-only CPT

Yes. AfriqueLLM is the closest Qwen3.5 result, but its objective and scale differ. That is why this plan uses its findings to choose small controlled tests rather than copying its token mixture.

## 13. Recommended update to Jan

The fullest update should wait for the 9B checkpoint sweep. If a progress note is useful now, this is accurate:

> Hi Jan, a quick update: the matched 2B and 4B base-model runs are complete on the same 436.4M isiZulu plus 48.5M English replay tokens. The 4B model is clearly stronger overall, but the best balanced checkpoint is around 150M tokens rather than the final checkpoint. At 150M, isiZulu validation perplexity improves from 27.21 to 12.56, AfriQA F1 from 52.3% to 65.9%, InjongoIntent accuracy from 21.9% to 37.2%, and POS accuracy from 64.7% to 70.1%. SIB-200 is roughly flat and NER decreases, so the result is useful but not uniformly positive. The matched 9B full run is about halfway through and should finish in roughly 15 hours, after which I will run the same checkpoint sweep. All of this is still validation-only and the held-out tests remain untouched.
>
> I have also checked the next data options. Rather than merging more Common Crawl sources blindly, I plan to cross-deduplicate high-quality HPLT and FineWeb2 isiZulu against MzansiText and run a small equal-token admission test. Could you send me the details and licence for the conversational isiZulu data you mentioned? Also, can you confirm whether the 9B model is acceptable, given the original 1B to 7B target range?

After the 9B sweep, replace the progress sentence with a small table containing original, selected CPT, and final CPT results for 2B, 4B, and 9B. Keep long-run token dose, model-size scaling, and SFT as separate comparisons.

## 14. Answers to the research questions

1. The best next data candidate is not a blind larger corpus. It is the unique, quality-ranked residual of HPLT, with filtered FineWeb2 as the matched challenger. SA-Knowledge is a small formal-domain supplement. Conversational transcripts and native instructions belong in a later, licensed experiment.
2. Keep full-parameter CPT, 10% English replay, sequence length 2,048, and the active 9B run. Select checkpoints on validation. Use 4B for the source and learning-rate study because it is strong and economical.
3. Before calling the model ready, finish the 9B curve, add translation and generation evidence, freeze a CPT checkpoint, and only then run a benchmark-clean 2 by 2 SFT comparison. Report validation-only results honestly and keep held-out tests sealed.

The likely path to a better isiZulu model is better marginal data and better checkpoint selection. More of the same MzansiText is now a weaker bet.

## References

[1] Yu, H., et al. 2026. *AfriqueLLM: How Data Mixing and Model Architecture Impact Continued Pre-training for African Languages*. ACL 2026, long paper 267.

[2] Ibrahim, A., et al. 2024. *Simple and Scalable Strategies to Continually Pre-train Large Language Models*. TMLR; arXiv:2403.08763.

[3] Gupta, K., et al. 2023. *How to Rewarm Your Model?*. arXiv:2308.04014.

[4] Muennighoff, N., et al. 2023. *Scaling Data-Constrained Language Models*. NeurIPS 2023.

[5] Biderman, D., et al. 2024. *LoRA Learns Less and Forgets Less*. TMLR; arXiv:2405.09673.

[6] Li, Z., et al. 2025. *Rethinking Multilingual Continual Pretraining: Data, Model, and Scale*. COLM 2025; arXiv:2504.04152.

[7] Xie, S. M., et al. 2023. *Data Selection for Language Models via Importance Resampling*. NeurIPS 2023.

[8] Minixhofer, B., et al. 2022. *WECHSEL: Effective Initialization of Subword Embeddings for Cross-Lingual Transfer of Monolingual Language Models*. NAACL 2022.

[9] Lombard, A., et al. 2026. *MzansiText and MzansiLM: An Open Corpus and Decoder-Only Language Model for South African Languages*. LREC 2026; arXiv:2603.20732.

[10] HPLT Project. 2025. *HPLT Monolingual Datasets 3.0*. Official release card and manifest.

[11] Penedo, G., et al. 2025. *FineWeb2: One Pipeline to Scale Them All, Adapting Pre-Training Data Processing to Every Language*. arXiv:2506.20920. Official release 2.1.1 metadata.

[12] Ralethe, S. 2026. *SA-Knowledge*. Official dataset release, University of Cape Town.

[13] Kudugunta, S., et al. 2023. *MADLAD-400: A Multilingual And Document-Level Large Audited Dataset*. arXiv:2309.04662.

[14] Way With Words. 2026. *South African isiZulu Simulated Call Centre Speech Dataset*. Official dataset card and Speech Collection Dataset Licence Agreement.

[15] Lee, K., et al. 2022. *Deduplicating Training Data Makes Language Models Better*. ACL 2022.

[16] Adelani, D. I., et al. 2024. *SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects*. EACL 2024.

[17] Adelani, D. I., et al. 2025. *IrokoBench: A New Benchmark for African Languages in the Age of Large Language Models*. NAACL 2025.

[18] Ogundepo, O., et al. 2023. *AfriQA: Cross-lingual Open-Retrieval Question Answering for African Languages*. Findings of EMNLP 2023.

[19] Nag, A., et al. 2025. *Language Adaptation for Large Language Models on a Tight Academic Budget*. NAACL 2025 Industry Track.

[20] Singh, S., et al. 2024. *Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning*. ACL 2024.

[21] Uemura, K., et al. 2024. *AfriInstruct: Instruction Tuning of African Languages for Diverse Tasks*. Findings of EMNLP 2024.

[22] Adebara, I., et al. 2025. *INJONGO: A Multicultural Intent Detection and Slot-Filling Dataset for African Languages*. ACL 2025.

[23] Adelani, D. I., et al. 2022. *MasakhaNER 2.0: Africa-Centric Transfer Learning for Named Entity Recognition*. EMNLP 2022.

[24] Dione, C. M. B., et al. 2023. *MasakhaPOS: Part-of-Speech Tagging for Typologically Diverse African Languages*. ACL 2023.

[25] de Wet, F., et al. 2024. *Correcting FLORES Evaluation Data for Four African Languages*. WMT 2024.

[26] Ojo, J., et al. 2025. *AfroBench: How Good are Large Language Models on African Languages?* Findings of ACL 2025.

[27] WMT 2025 Shared Task Organizers. 2025. *Evaluation of Automatic Metrics for African Machine Translation*. WMT 2025.

[28] Xia, M., et al. 2023. *Training Trajectories of Language Models Across Scales*. ACL 2023.

[29] Mabuya, R., et al. 2024. *NGLUEni: A Benchmark and Dataset Suite for Natural Language Generation in Nguni Languages*. LREC-COLING 2024.
