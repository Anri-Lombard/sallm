# Evaluation Investigation Log

**Created:** 2025-12-28
**Status:** In Progress

## Current Monolingual Results (Dec 23, 2024)

### Classification (Accuracy)

| Task | afr | eng | nso | sot | xho | zul |
|------|-----|-----|-----|-----|-----|-----|
| **SIB** | 11.27% | 17.65% | 16.67% | 12.25% | 9.31% | 13.24% |
| **InjongoIntent** | - | 2.57% | - | 4.22% | 2.66% | 3.44% |
| **MasakhaNEWS** | - | 13.82% | - | - | 13.47% | - |

### NER (F1)

| Task | tsn | xho | zul |
|------|-----|-----|-----|
| **MasakhaNER** | 0% | 0% | 0% |

### Generation

| Task | Metric | xho | zul |
|------|--------|-----|-----|
| **AfriHG** | chrF | 7.52 | 2.42 |
| **AfriHG** | ROUGE-L | 0.58% | 0.69% |
| **T2X** | chrF | 23.96 | - |
| **T2X** | ROUGE-L | 22.37% | - |

---

## Missing Evaluations (Need to Run)

- [x] _all multilingual models (sib_all, news_all, ner_all, injongointent_all, afrihg_all) - SUBMITTED
- [ ] pos_all (skipped - broken)
- [ ] sa_general_all (waiting for HPO to complete)

---

## Investigation 1: NER 0% F1

**Problem:** All NER evaluations return 0% F1 for tsn, xho, zul

### Hypothesis Checklist
- [ ] Model not generating in expected NER format
- [ ] Evaluation metric mismatch (flexible-extract not matching output)
- [ ] Template formatting issue during fine-tuning vs evaluation
- [ ] Model generating empty or garbage output

### Investigation Steps
1. Check what the model is actually generating during eval
2. Compare eval template format with training template format
3. Check if lm-eval NER task expects specific output format
4. Look at raw predictions vs expected outputs

### Findings

**ROOT CAUSE IDENTIFIED: Corrupted model merge for lm-eval**

1. **Training DID work** (verified via wandb run `glfrb75h` Dec 17):
   - `eval/xho_rougeL: 0.2495` (25%)
   - `eval/xho_chrf: 34.24`

2. **The merged model saved on cluster is CORRUPTED**:
   - Pre-merged model generates: `"uhlaza qaku qaku qaku..."` (broken)
   - Properly merged model generates: `"DATE: kulo kule nyanga..."` (NER format)

3. **The corruption happened during lm-eval's merge process**:
   - Located at: `/scratch/.../results/eval/ft_mamba_125m_ner_xho/_lm_eval/merged_model`
   - The model always outputs token ID 21168 (`uhlaza`) then 16910 (`qaku`) repeating
   - This is not the trained model behavior

4. **Vocab size mismatch was handled incorrectly**:
   - Base model: 65536 tokens
   - Adapter tokenizer: 65539 tokens (added `<|system|>`, `<|user|>`, `<|assistant|>`)
   - When resized + merged incorrectly, embeddings for new tokens got corrupted

### Fixes Applied (Dec 28, 2024)

1. **Use PEFT adapter directly** (`lm_eval_runner.py`):
   - Changed `_materialize_model_for_lm_eval` → `_prepare_model_for_harness`
   - Now passes adapter path to lm-eval's native `peft=` argument
   - Avoids merging entirely, preventing corruption

2. **Use adapter's tokenizer** (`lm_eval_runner.py`):
   - `tokenizer_source = peft_adapter if peft_adapter else pretrained_path`
   - Ensures correct vocab size (65539) and chat template

3. **Disable weight tying for Mamba** (`config.py` + yaml configs):
   - Added `extra_model_args` field to `ModelEvalConfig`
   - All mamba configs updated with `extra_model_args: {tie_word_embeddings: false}`
   - Fixes `KeyError: "attribute 'weight' already exists"` crash

4. **Config path fix**:
   - Python code at `/home/` uses configs from `/home/.../src/conf/`
   - Was incorrectly SCPing to `/scratch/` - configs not being read
   - Now properly syncing to `/home/lmbanr001/masters/sallm/src/conf/`

### Re-evaluation Jobs Submitted

All monolingual mamba evals resubmitted (Jobs 411549-411571):
- NER: tsn, xho, zul, all
- NEWS: eng, xho, all
- SIB: afr, eng, nso, sot, xho, zul, all
- InjongoIntent: eng, sot, xho, zul, all
- AfriHG: xho, zul, all
- T2X: xho

---

## Investigation 2: NEWS Low Accuracy (~13-14%)

**Problem:** MasakhaNEWS accuracy is only 13-14% for both eng and xho

### Hypothesis Checklist
- [ ] Random baseline would be ~14% (7 classes) - model not learning
- [ ] Label format mismatch between training and evaluation
- [ ] Wrong checkpoint being evaluated
- [ ] Evaluation prompts differ from training prompts

### Investigation Steps
1. Check number of classes (what's random baseline?)
2. Compare training template with eval template
3. Look at model predictions vs expected labels
4. Verify correct fine-tuned checkpoint is loaded

### Findings
<!-- To be filled in -->

---

## Investigation 3: Other Tasks

### SIB (~9-17%)
- Random baseline for 7-way classification would be ~14%
- Results are near random - needs investigation

### InjongoIntent (~2-4%)
- Very low accuracy
- Need to check number of intent classes

### AfriHG (chrF 2-7)
- Very low chrF scores
- Need to check if model is generating headlines at all

---

## Action Items

1. [x] Debug NER evaluation - ROOT CAUSE: corrupted merge, FIXED
2. [ ] Debug NEWS evaluation - re-running with fix, check results
3. [x] Re-run evaluations after fixes - Jobs 411549-411571 submitted
4. [x] Run _all multilingual evaluations - included in job submission
5. [ ] Update results spreadsheet - pending job completion
6. [ ] Evaluate sa_general_all model after HPO completes (Job 411539)
