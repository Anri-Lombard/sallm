# Pure-GDN News held-out recovery results

Date: 2026-09-03

Recovery job `1291841` completed `0:0` in `00:13:12`. All four frozen
News arms passed structural verification before metrics were opened:

- Multi English: `4,740` rows over five prompts; mean F1
  `0.3841577509141925`, range `0.3697051330504381` to
  `0.40960271228156625`.
- Multi Xhosa: `1,485` rows over five prompts; mean F1
  `0.43592257295697856`, range `0.4282394076511724` to
  `0.44502240066277915`.
- Mono English: `4,740` rows over five prompts; mean F1
  `0.2591946070862873`, range `0.2461169243567566` to
  `0.27172156979179996`.
- Mono Xhosa: `1,485` rows over five prompts; mean F1
  `0.33770995918666835`, range `0.22386963478063113` to
  `0.3857687468563473`.

These are descriptive means over every frozen held-out prompt, not prompt
selection. The exact source/config files used by lm-eval matched the immutable
snapshot. The source-equivalence manifest SHA-256 is
`288a40d9821a93287f002c54fb61b28851d591cd67dcef2ce99aedbb0abedd7c`.
The sealed result-tree manifest SHA-256 is
`1b34d001da4abd04dbca5b950095e513dff717210f1380de79514b182045bf5f`.

The values remain pending Sheet write until POS and AfriHG recovery bundles
also complete and verify.
