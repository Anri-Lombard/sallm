# Data and pilot strategy for isiZulu continual pretraining of Qwen3.5-4B

## Abstract

The machine can be prepared now, but the first training run should wait for a short data audit. The recommended corpus is not a blind union of MzansiText, HPLT 3.0, FineWeb2, and MADLAD-400. All four contain web data, three draw heavily from overlapping Common Crawl periods, and MzansiText already includes several older African and multilingual corpora. The defensible starting point is `uctnlp/mzansi-text-deduplicated`, followed by HPLT 3.0, FineWeb2, and MADLAD clean in that order. Each supplement earns inclusion only through unique retained Qwen tokens, acceptable native-reviewed quality, resolved usage terms, and no evaluation contamination after union-level exact and near deduplication.

The model decision also remains empirical. African-language work usually continues from base checkpoints, but Jan's concern is reasonable: without enough isiZulu instruction data, a base model may be hard to turn into a useful assistant. Training an instruction model on raw text can, however, damage instruction following, language choice, formatting, and reliability. The first experiment should therefore compare `Qwen/Qwen3.5-4B-Base` and `Qwen/Qwen3.5-4B` under the same small token budgets. The held-out test set must remain sealed until the corpus, checkpoint, prompt, decoding, and evaluation rules are frozen.

## 1. Decision

Proceed in two tracks:

1. Prepare and verify the ROCm runtime, storage layout, model loading, tokenizer round-trip, forward and backward pass, checkpoint save, and checkpoint reload.
2. Hold full dataset ingestion and continual pretraining until the source audit reports the unique contribution and quality of each candidate corpus.

This is a go for setup and a hold for training. The setup should remain model-neutral until both Qwen checkpoints load through the same text-only path.

## 2. Research questions

This review addresses three questions.

1. Which public isiZulu corpora are usable, legally understandable, and materially distinct?
2. Which cleaning, deduplication, decontamination, and replay controls are supported for low-resource continual pretraining?
3. What is the smallest experiment that can choose between the base and post-trained Qwen3.5-4B checkpoints without spending the full budget?

## 3. Methodology

The search covered primary papers, peer-reviewed proceedings, official model and dataset cards, official pipeline code, and the local SALLM training path. Sources were checked live on 21 August 2026. Dataset sizes were not normalized across tokenizers because MzansiText, HPLT, FineWeb2, and MADLAD report different units. Counts are reported in their published unit, and the proposed audit retokenizes all retained text with the Qwen tokenizer.

Evidence labels in this report are:

- A: peer-reviewed paper, official release metadata, or normative standard.
- B: primary preprint or official implementation without peer review.
- C: inference from multiple sources or a proposed experimental choice.

The local repository was inspected separately. That inspection is implementation evidence, not a claim about the published datasets.

## 4. Corpus taxonomy

The candidates fall into four practical groups.

1. A South African anchor: MzansiText combines local and multilingual sources and has a reproducible pipeline [1].
2. Large multilingual web supplements: HPLT 3.0, FineWeb2, and MADLAD-400 [3-7].
3. General replay: a small sample from the base model's broader language and domain distribution, used to test retention rather than to increase isiZulu volume [12, 13].
4. Evaluation-only material: NGLUEni, corrected FLORES where relevant, instruction-retention prompts, and a fresh native-reviewed set [16-18]. These records must never enter the training corpus.

## 5. Candidate data sources

| Source | Published isiZulu quantity | Existing processing | Terms and access | Recommendation |
|---|---:|---|---|---|
| `uctnlp/mzansi-text-deduplicated`, language `zul` | The original release reports 320,224,015 isiZulu train tokens under its own 65,536-token vocabulary [1]. The new release removed 19,156 exact isiZulu duplicate rows before splitting [2]. | NFC normalization, collapsed whitespace, SHA-256 exact deduplication keyed by language, then deterministic splitting [2]. The older local cleaning path also applies repetition and generic quality filters. | Dataset card states Apache-2.0. Upstream-source obligations still need to remain in the manifest. | Include first as the anchor. Use the deduplicated release, not the older split. |
| HPLT 3.0, `zul_Latn` | 336,440 documents, about 410.54 million Gemma-3 tokens, 8.02 million segments, and 1.12 billion characters [3]. | Trafilatura extraction, OpenLID 2.0, Monotextor filtering and annotations, quality ordering, and global language-level deduplication for isiZulu [3, 4]. | HPLT packages the release under CC0 but says it does not own the underlying web text. Copyright and data-protection review remains the user's responsibility. | Audit second. It has the largest clearly reported recent isiZulu web slice and useful quality metadata. |
| FineWeb2, `zul_Latn` | The live viewer exposes 127,335 train rows and 762 test rows in the current configuration. The card does not give a canonical current Qwen-token count [5]. | GlotLID, language-dependent thresholds, global per-language MinHash deduplication, quality filtering, and PII handling across 96 Common Crawl snapshots through April 2024 [5, 6]. | ODC-By 1.0 plus Common Crawl terms. | Audit third. Use filtered train only. Exclude its test and removed subsets. |
| MADLAD-400, `zu` | 53,809 clean documents and about 257.4 million clean characters. The noisy slice has about 372,300 documents and 1.2 billion characters [7]. | Language identification, document filters, repeated three-sentence-span removal, questionable-content rules, and manual language audit [7]. | Current card metadata states ODC-By. Common Crawl terms also apply. The prose in older card material has shown inconsistent license wording, so pin and archive the exact metadata used. | Audit clean fourth. Do not use noisy by default. |

Standalone Inkuba-Mono should not enter the initial candidate set. MzansiText already lists it as an input, while the standalone release is gated and marked non-commercial [1, 8]. WURA, mC4, CC100, CulturaX, Glot500-c, NCHLT, and ParaCrawl also appear in MzansiText's provenance. Adding them again would mainly test the deduplicator.

### 5.1 What is confirmed

- The new MzansiText release removes normalized exact duplicates before splitting and reports no remaining exact duplicates under that rule across its splits [2]. Evidence A.
- HPLT, FineWeb2, and MADLAD each perform some internal deduplication [3-7]. Evidence A or B, depending on the component.
- Internal deduplication does not establish that two separately released corpora are distinct. No pairwise overlap matrix was found for these isiZulu slices. Evidence C.
- HPLT, FineWeb2, and MADLAD share Common Crawl provenance and overlapping crawl years. Substantial overlap is therefore likely, though its size is unknown. Evidence C.

### 5.2 Why the largest published count is not the answer

The published counts use incompatible tokenizers and units. More importantly, total rows or tokens do not measure unique, useful isiZulu after cross-source deduplication. A source with fewer retained tokens may still add better domains, registers, or newer material. The audit should rank sources by retained Qwen tokens and reviewed quality, not by their card headline.

## 6. Cleaning and deduplication strategy

### 6.1 Preserve raw text and provenance

Each processed record should keep the source, URL or stable source identifier, crawl date where available, hostname, original language scores, normalization version, quality scores, and duplicate-cluster ID. Raw text stays immutable. This allows a better copy to be chosen when several corpora contain the same page.

### 6.2 Normalize conservatively

Apply Unicode NFC, decode valid UTF-8, remove control and zero-width artifacts, repair malformed markup where unambiguous, and normalize pathological whitespace [2, 14]. Keep punctuation, case, diacritics, and the original text. Do not apply NFKC, transliteration, broad case folding, or diacritic stripping without an isiZulu-specific experiment. NFC preserves canonical equivalence, while NFKC can erase compatibility distinctions [14].

### 6.3 Calibrate language identification on real web text

Run language identification at line and document level. Build a small native-reviewed calibration set that contains isiZulu, neighbouring Nguni languages, English and Afrikaans code-switching, boilerplate, and short documents. Choose the threshold from measured precision and recall. FineWeb2 found that language-specific settings were necessary and that aggressive general filters could hurt lower-resource data [5, 6]. Its Swahili settings are not an isiZulu threshold.

### 6.4 Use isiZulu-aware quality controls

Reject non-linguistic pages, navigation and boilerplate, severe repetition, malformed documents, obvious spam, PII, and extreme script anomalies. Audit every word-count, word-length, stopword, punctuation, and repetition rule before applying it broadly. isiZulu's conjunctive orthography and rich morphology make English word heuristics unsafe [15].

This matters for the current SALLM code. `data/cleaning/clean_mzansi_text.py` applies Gopher, C4, and FineWeb quality filters, including minimum word and sentence rules. It handles within-document repetition, but it does not perform union-level cross-document near deduplication. Those filters should not be reused unchanged for the new union.

### 6.5 Deduplicate before splitting

Use the following order:

1. Reserve every evaluation, validation, and test record, then build exact and n-gram fingerprints for contamination checks.
2. Deduplicate exact URLs and stable source IDs within each source.
3. Apply NFC plus whitespace normalization and SHA-256 exact-text deduplication across sources.
4. Remove exact repeated sentence or paragraph spans, with a separate audit of short documents.
5. Run 5-word-shingle MinHash near deduplication. FineWeb and DataTrove use a practical 14-band by 8-hash configuration [5, 9]. Treat it as a starting point, not an isiZulu optimum.
6. Choose one canonical keeper per cluster using provenance, language confidence, quality, formatting, and domain value.
7. Recheck benchmark contamination after all normalization and deduplication.
8. Create train and development splits by document and duplicate cluster. Seal the held-out test split last.

Exact and approximate deduplication can reduce memorization and train-evaluation overlap while preserving or improving language-model accuracy [24]. The order is important. The local `data/prepare_datasets.py` currently shuffles documents and assigns validation and test before any union-level duplicate clustering. Near duplicates could therefore cross split boundaries if that path were reused unchanged.

### 6.6 Keeper priority

Use source-aware keeper selection. A reasonable initial order is:

1. verified local, government, educational, reference, and audited news material;
2. the MzansiText anchor;
3. HPLT records with strong language and quality metadata;
4. FineWeb2 filtered records;
5. MADLAD clean records;
6. parallel-crawl or suspected machine-translated material, kept as a separate ablation.

This ordering is a proposed rule, not an established isiZulu ranking. Record all discarded cluster members so the choice can be audited. Random global keeper selection risks preserving a poor mirror of a good document [5, 9].

## 7. Minimum data audit before training

The audit should produce one manifest and one decision table. It does not need a new data platform.

### 7.1 Required measurements

For each source, report:

- input documents, characters, and Qwen tokens;
- exact duplicates within the source and against higher-priority sources;
- near-duplicate clusters and unique retained tokens;
- language-ID score distributions and a native-reviewed confusion table;
- domain and hostname concentration;
- quality-filter removal counts by rule;
- benchmark contamination matches;
- the pinned dataset revision, terms, and upstream provenance.

For native review, draw a stratified sample across sources, language scores, quality bins, and document lengths. The sample size and acceptance threshold should be agreed with Jan or another isiZulu reviewer before labels are opened. A fixed universal threshold would pretend the literature has settled something it has not.

### 7.2 Inclusion rule

A supplement enters the pilot only if it adds a material amount of unique Qwen-tokenized isiZulu, its reviewed quality is acceptable, its terms are recorded, and it passes decontamination. If HPLT contributes most of the clean unique increment, stop there. FineWeb2 and MADLAD do not need to be included merely because they are available.

### 7.3 Proposed source ablation

Hold total training tokens fixed and compare:

- MzansiText anchor only;
- anchor plus HPLT;
- anchor plus the best unique supplement after audit.

Do not begin with a four-source mixture. That would confound source quality, overlap, and checkpoint choice in one experiment.

## 8. Base versus post-trained Qwen3.5

Jan's reasoning makes sense, but it does not settle the checkpoint choice.

African-language precedents such as AfriInstruct and AfriqueLLM largely continue from base checkpoints, then apply instruction tuning where data is available [10, 11]. A matched study of Llama and Qwen families reports that raw continual pretraining of instruction checkpoints can reduce instruction-following scores as token exposure grows [12]. Traditional Chinese adaptation of a chat checkpoint also reported changes in output language, repetition, and reliability [19]. Counterevidence shows that forgetting can be moderate and sensitive to scale rather than inevitable [20].

The practical conclusion is a matched pilot. A base-only run risks leaving us with a model that knows more isiZulu but cannot be made useful with the available instruction data. An instruction-only run risks damaging the behaviour we hoped to preserve.

### 8.1 Four required arms

1. Untouched `Qwen/Qwen3.5-4B-Base` control.
2. Untouched `Qwen/Qwen3.5-4B` post-trained control.
3. Base CPT on the frozen isiZulu batches.
4. Post-trained CPT on exactly the same batches, token count, sequence length, optimizer, schedule, and seed.

The controls cost evaluation time, not training time. Replay should be a second-stage arm only if either CPT run improves isiZulu while failing retention.

### 8.2 Budget ladder

Use one seed at 10 million and 30 million tokens to catch obvious failure, then advance both checkpoint arms to 90 million tokens only if they remain plausible. Run multiple seeds at the selected budget. AfriqueLLM reports a useful 90-million-token threshold in its own setting, but that is precedent rather than a universal minimum [11]. If the cleaned corpus has fewer than 90 million unique tokens, report epochs and repeated exposure explicitly.

Pre-register a small development-only learning-rate calibration, for example `1e-5`, `3e-5`, and `5e-5`, then lock one rate for the matched comparison. Never use held-out test results to select the learning rate, token budget, source mix, checkpoint, prompt, or decoding settings.

### 8.3 Replay

General replay can reduce forgetting, but there is no isiZulu-specific best ratio. Literature supports testing small fractions such as 5%, 10%, and 25%, with diminishing returns possible at larger ratios [13, 21]. Replay dilutes isiZulu exposure, so compare arms at equal total tokens and choose on a development-set Pareto rule.

## 9. Evaluation and selection rule

### 9.1 Adaptation

Measure raw isiZulu development perplexity and bits per byte, plus fixed isiZulu tasks. NGLUEni provides six tasks across eleven datasets for isiZulu and related Nguni languages [16]. Use corrected FLORES data if translation is included because native review has identified errors in the original isiZulu development and test material [17].

### 9.2 Retention

For the post-trained arm, measure exact-format compliance, language choice, repetition, refusal and safety behaviour, answer validity, and English plus general task regressions. For the base arm, measure whether a small fixed instruction-recovery step is sufficient, but do not use held-out results to design that step.

### 9.3 Decision rule

Select on development data using a pre-registered Pareto rule:

- isiZulu language modelling and task performance must improve;
- instruction and general capabilities must stay within agreed non-inferiority limits;
- the result must be stable enough across seeds to justify a full run;
- native review must not show a material increase in wrong-language output, repetition, or broken isiZulu.

If neither checkpoint clears the gate, the correct result is no-go. Open the held-out test set once after the entire recipe is frozen.

## 10. Implementation readiness

The AMD host has enough memory for this scale of model, and a runtime wizard has already been syntax-checked for storage creation, ROCm container pull, and a real GPU matrix multiplication smoke test. It does not start training.

Three software checks remain before calling the environment training-ready:

1. The local environment has Transformers 4.57.3, but its auto-configuration mapping does not contain `qwen3_5`. Pin a tested Transformers revision that contains `Qwen3_5ForCausalLM` inside the container.
2. `src/main/sallm/models/factory.py` forcibly replaces every tokenizer decoder with `ByteLevel`. Do not run Qwen through this path until a tokenizer round-trip proves that mutation is valid, or the mutation is removed for pretrained tokenizers.
3. The current model registry does not expose Qwen3.5. The smoke test should first use the official text-only class directly. Add the smallest repository integration only after model load, one backward step, save, and reload work on ROCm.

The official Qwen cards describe a causal language model with a vision encoder and distinguish pre-trained-only and post-trained checkpoints [22, 23]. Use the text-only language-model path for this project and leave the vision encoder out of the optimizer.

## 11. Self-adversarial review

### 11.1 Strongest objection

The proposed audit may delay a cheap pilot. MzansiText alone already contains enough published isiZulu tokens for a 90-million-token run, and the matched pilot could start without HPLT, FineWeb2, or MADLAD.

That objection is partly correct. A small anchor-only runtime and training smoke test is safe once evaluation exclusions have been applied and split integrity is checked. It does not justify a full run or a four-source mixture. The source-expansion audit can run alongside the 10-million-token anchor-only pilot; benchmark decontamination cannot.

### 11.2 Evidence gaps

- No published pairwise overlap matrix exists for these isiZulu releases.
- No isiZulu-specific GlotLID threshold, MinHash threshold, replay fraction, or source-quality ordering was found.
- The checkpoint cards do not prove that every corresponding base and post-trained text tensor has an exact ancestor relationship.
- Published token counts are not comparable without retokenization.
- Public benchmarks may overlap web training data even after exact matching. A fresh native-reviewed set remains necessary.

### 11.3 Claims that should not be made

- "All four corpora provide independent isiZulu data."
- "The largest corpus will train the best model."
- "An instruct checkpoint will preserve instruction following during raw CPT."
- "A base checkpoint can be recovered later with the instruction data currently available."
- "Internal source deduplication prevents train-test leakage after the sources are merged."

## 12. Recommended sequence

1. Run the existing machine-setup wizard and record the image digest and GPU smoke result.
2. Prove direct text-only load, tokenizer round-trip, one forward and backward step, checkpoint save, and reload for both Qwen checkpoints.
3. Pin `uctnlp/mzansi-text-deduplicated`, freeze the evaluation exclusion manifest, and apply exact and near-match exclusions.
4. Start an anchor-only 10-million-token matched smoke pilot only after that decontamination gate passes.
5. In parallel, audit HPLT, FineWeb2, and MADLAD clean for unique retained Qwen tokens and native-reviewed quality.
6. Freeze the source mixture and development selection rule.
7. Run the 30-million and conditional 90-million-token matched pilots.
8. Add a replay arm only if adaptation succeeds but retention fails.
9. Freeze the winning recipe, then open the held-out test once.

## 13. Conclusion

Setup should proceed now. Full training should not. Use the deduplicated MzansiText release as the initial corpus and allow HPLT, FineWeb2, or MADLAD clean into the mixture only when the audit shows useful unique data. Jan's post-trained-checkpoint proposal is plausible enough to test and risky enough not to assume. A matched base-versus-post-trained pilot on identical batches is the shortest defensible route to the decision.

## References

[1] Lombard, A., et al. 2026. *MzansiText and MzansiLM: An Open Corpus and Decoder-Only Language Model for South African Languages*. LREC 2026 and arXiv:2603.20732.

[2] UCTNLP. 2026. *mzansi-text-deduplicated dataset card*. Hugging Face, revision accessed 21 August 2026.

[3] HPLT Project. 2025. *HPLT Monolingual Datasets 3.0 dataset card and manifest*. Release 3.0.

[4] de Gibert, O., et al. 2024. *A New Massive Multilingual Dataset for High-Performance Language Technologies*. LREC-COLING 2024.

[5] Hugging Face. 2026. *FineWeb2 dataset card and dataset viewer*. Version 2.1.1, accessed 21 August 2026.

[6] Penedo, G., et al. 2025. *FineWeb2: One Pipeline to Scale Them All, Adapting Pre-Training Data Processing to Every Language*. arXiv:2506.20920.

[7] Kudugunta, S., et al. 2023. *MADLAD-400: A Multilingual And Document-Level Large Audited Dataset*. arXiv:2309.04662.

[8] Dossou, B. F. P., et al. 2024. *InkubaLM: A Small Language Model for Low-Resource African Languages*. arXiv:2408.17024.

[9] Penedo, G., et al. 2024. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*. NeurIPS 2024 Datasets and Benchmarks Track.

[10] Uemura, K., et al. 2024. *AfriInstruct: Instruction Tuning of African Languages for Diverse Tasks*. Findings of EMNLP 2024.

[11] Yu, H., et al. 2026. *AfriqueLLM: How Data Mixing and Model Architecture Impact Continued Pre-training for African Languages*. ACL 2026 and arXiv:2601.06395.

[12] Jindal, S., et al. 2024. *Balancing Continuous Pre-Training and Instruction Fine-Tuning: Optimizing Instruction-Following in LLMs*. arXiv:2410.10739.

[13] Abbes, I., et al. 2025. *Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models*. arXiv:2508.01908.

[14] Unicode Consortium. 2025. *Unicode Standard Annex #15: Unicode Normalization Forms*. Revision 57.

[15] Eiselen, R. and Gaustad, T. 2023. *Deep Learning and Low-Resource Languages: How Much Data Is Enough? A Case Study of Three Linguistically Distinct South African Languages*. Proceedings of the Fourth Workshop on Resources for African Indigenous Languages.

[16] Meyer, F., et al. 2024. *NGLUEni: Benchmarking and Adapting Pretrained Language Models for Nguni Languages*. LREC-COLING 2024.

[17] Abdulmumin, I., et al. 2024. *Correcting FLORES Evaluation Dataset for Four African Languages*. arXiv:2409.00626.

[18] Sainz, O., et al. 2023. *NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for Each Benchmark*. Findings of EMNLP 2023.

[19] Li, C. and Lee, H. 2024. *Examining Forgetting in Continual Pre-training of Aligned Large Language Models*. arXiv:2401.03129.

[20] Harmon, J., et al. 2025. *Mapping Post-Training Forgetting in Language Models at Scale*. arXiv:2510.17776.

[21] Ibrahim, A., et al. 2024. *Simple and Scalable Strategies to Continually Pre-train Large Language Models*. Transactions on Machine Learning Research.

[22] Qwen Team. 2026. *Qwen3.5-4B-Base model card*. Hugging Face, accessed 21 August 2026.

[23] Qwen Team. 2026. *Qwen3.5-4B model card*. Hugging Face, accessed 21 August 2026.

[24] Lee, K., et al. 2022. *Deduplicating Training Data Makes Language Models Better*. Proceedings of ACL 2022.
