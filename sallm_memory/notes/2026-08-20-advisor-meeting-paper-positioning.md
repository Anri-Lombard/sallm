# Advisor meeting: paper positioning and preliminary results — 2026-08-20

Source: [Granola meeting notes and transcript chat](https://notes.granola.ai/t/d5a009ce-3463-47a4-b35e-cc90e53c7876)

Status: meeting notes, not canonical experimental evidence. All numerical
claims below are preliminary recollections and must be reconciled against the
frozen artifacts before entering the paper, result sheet, or architecture
ranking.

## Paper Positioning and Results

- Wait for full results before finalizing the framing
- Frame more generally: test case for small language models trained on limited data
  - Findings shouldn’t be language-specific; more a function of data size and model size
  - Someone doing this in another language set at 125M parameters should expect similar results

## Preliminary GDN Results

- NER: 0.65 to 0.68 across languages, on par with Transformer (~0.70)
- Parts of speech: ~80, also on par with Transformer
- Generation tasks (T2X): 51, above Llama; Zulu: 23 and 26 respectively
  - GDN performing better than expected on generation; needs sanity checks
- Pre-training loss ranking: Llama (2.5) > xLSTM (2.97) > GDN (3.42) > Mamba (3.99)
  - Potential inconsistency between loss ranking and downstream eval results
  - Mamba is the outlier: worst performance and the only model that started overfitting
- HPO done for xLSTM and GDN; Mamba and Llama still on old HPO approach
  - Risk that their results aren’t as strong as they could be
- GDN has no optimized kernels, making it the slowest to fine-tune; others will go faster once it’s done

## Timeline, Compute, and Side Project

- Target: first draft in September, submit paper early October (CoNLL or similar, TBC)
  - Plan to send draft to advisors first to validate positioning, knowing results may still change
  - October is bad for both advisors, so September draft is the priority
- Compute options explored: additional GPU partitions after the Honest project deadline (early September), cloud credits, AMD GPUs
  - AMD GPUs available now but better suited to continual pre-training than scratch training
- Quinn/hybrid model: not a priority until full results are in; could strengthen the paper if GDN outperforms, but risks reviewer questions about incomplete hybrid coverage
- Open-source Xhosa LLM consortium project (with MTN and others)
  - Clayton handling speech recognition; next phase is continual pre-training of an existing model (Llama or Gemma, up to 70B) on Xhosa data, then instruction fine-tuning
  - Data quality is a known risk: audio data from the consortium arrived as raw call-center recordings with no transcriptions
  - Anri expressed interest in joining; will send SSH public key (via Bitbucket) to get access to the AMD GPU server

## Next Steps

- **Complete HPO and compile full GDN results**
- **Write first paper draft in September**
- **Send SSH public key via Bitbucket for AMD GPU access** (Anri)

## Transcript

Retrieval pending. The public Granola link exposes the generated meeting notes
and an “Ask anything” transcript chat, but does not expose a raw transcript or
download control directly. The transcript will be appended here after querying
the Granola transcript chat.
