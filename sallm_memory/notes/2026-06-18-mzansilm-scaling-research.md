# MzansiLM scaling research and later-training TODO

## Research conclusion

- Train a `~250M` MzansiLM next.
- Do not train `400M`-`500M` yet unless MzansiText v2 reaches roughly
  `8B`-`10B` deduped useful tokens with better non-English/non-Afrikaans
  coverage.
- Current MzansiText train split is about `3.81B` tokens.
- Existing local LLaMA configs:
  - `125M`: `125,008,384` actual parameters.
  - `400M`: `397,863,936` actual parameters.
- Chinchilla-style target is roughly `20` tokens per parameter:
  - `125M`: `2.5B` target tokens.
  - `250M`: `5.0B` target tokens.
  - `400M`: `8.0B` target tokens.
  - `500M`: `10.0B` target tokens.

## MzansiText v2 direction

- Treat v2 as a deduped delta over v1, not a blind repack of web corpora.
- Cross-dedup against v1 and against held-out downstream/eval material.
- Keep source, URL/date/domain, language, and license/provenance metadata.
- Cap dominant languages in sampling:
  - Afrikaans roughly `35%`-`40%`.
  - English roughly `12%`-`15%`.
- Use sampling weights for tiny languages rather than physically duplicating
  text until they become memorization tests.
- Add source/date/domain-disjoint validation for v2.

## Candidate data sources

- FineWeb2: first web-delta candidate; has SA language subsets, but likely
  overlaps heavily with v1 web sources.
- HPLT v2 cleaned: compare as a second web delta; useful if dedup leaves real
  new `zul`, `xho`, `sot`, `nso`, `tsn`, `tso`, `ssw`, and `af` text.
- SADiLaR / Autshumato / NCHLT refresh / Vuk'uzenzele / ZA government:
  smaller but high-value curated sources, especially for low-resource SA
  languages.
- MADLAD-400: not a priority for low-resource SA language coverage; only use
  after audit if it provides clean, deduped, useful delta text.
- Do not blindly re-add CulturaX, mC4, CC100, Glot500, WURA, or Inkuba because
  v1 already used them.

## Later training TODO

1. Build a MzansiText v2 audit manifest.
   - List v1 sources and source identifiers.
   - Add candidate v2 source identifiers.
   - Record license/provenance and expected language coverage.

2. Prototype a v2 dedup pipeline on a small sample.
   - Exact dedup first.
   - Then MinHash/SimHash-style near-dedup across v1 + v2.
   - Fuzzy-dedup against downstream benchmark train/validation/test material.

3. Produce a post-dedup token table.
   - Per source.
   - Per language.
   - Raw tokens versus kept tokens.
   - Low-resource language gain after dedup.

4. Decide the next base size.
   - If v2 reaches about `5B` clean useful tokens: train `~250M`.
   - If v2 reaches about `8B`-`10B` clean useful tokens: revisit `400M`-`500M`.
   - If v2 remains near `3.8B`-`4.5B`: stay with `125M`/`250M`, not larger.

5. Prepare a `250M` config only after the v2 token table exists.
   - Keep parameter-count audit next to the config.
   - Match tokenizer/vocab assumptions to the existing MzansiLM line.
   - Add a cheap loss-audit gate before any downstream wave.

6. After a successful `250M` base, rerun the same downstream suite.
   - POS with constrained closed-label token scoring.
   - NER official span rows plus diagnostic tag-sequence rows kept separate.
   - T2X, AfriHG, InjongoIntent, SIB/news as already defined.
   - Promote only final held-out test rows to the Google Sheet.

## Open decision

Do the v2 data audit before writing more large-model configs. The next useful
work is data accounting, not another speculative architecture file.
