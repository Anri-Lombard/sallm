# Pure-GDN base evaluation closeout — 2026-08-07

- All 16 frozen zero-shot base-evaluation lanes are complete and their rows
  are verified in `Pure GatedDeltaNet Results` (`sheetId=202608060`). No
  held-out metric was used to change a checkpoint, prompt, recipe, or rerun.
- Final lane `1185915_15` completed `0:0` on one A100-40GB
  `gpu:ampere` at 00:24:44 SAST after `09:33:50`. No owned GPU job remained
  in `squeue` at the 2026-08-07 verification pass.
- Final AfriHG result root is
  `/scratch/lmbanr001/masters/sallm/results/eval/pure_gdn_125m_base_0shot_afrihg_all_r1`;
  combined summary SHA-256 is
  `089800aba8602dc7bec9795176e0a38aa1f7f17fcbb508c7903c210d241bad78`.
- Xhosa: chrF `4.079834809867061`, ROUGE-1/ROUGE-L
  `0.00012029488297117726`, ROUGE-2/BLEU `0.0`; metrics SHA-256
  `9cbb9f7572794838c9cb2b53104ff814bd688bdf225d1b22ed7897b15001f198`.
- Zulu: chrF `4.125909737887804`, ROUGE-1/ROUGE-L
  `0.00012025427680584782`, ROUGE-2/BLEU `0.0`; metrics SHA-256
  `084bb52800d659ac81e71e43863130c62092c708c8422b9a99bf691d59c5cab5`.
- English is structurally inapplicable for AfriHG and was explicitly recorded
  as not applicable; no English metric was run or inferred. Rows 42--44 were
  re-read after publication with job, paths, hashes, notes, dates, and wrapping
  intact. Historical hybrid tabs remain untouched.
- Fresh resource state: HEX `/scratch` `100/300 GB` (`33.5%`); Kombuys was
  read-only and idle (RTX 5090 `10 MiB`, `0%`; RTX 3080 Ti `1 MiB`, `0%`;
  `/scratch` `61%`).
- Fine-tuning remains paused. The next authorized action is protocol inspection
  and preregistration for validation-selected Monolingual, Multilingual, and
  General pure-GDN variants. No training or held-out adapter evaluation may be
  submitted before that preregistration is complete.
- Operational deadline remains 31 August 2026 for all experiments and
  evaluations. Paper drafting is deferred; first advisor draft is targeted for
  mid-September.

