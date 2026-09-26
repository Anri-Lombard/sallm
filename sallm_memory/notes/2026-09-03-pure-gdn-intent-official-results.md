# Pure-GDN Intent official results — 3 September 2026

## Status

- Corrected official job: `1290458`, completed `0:0` in `00:52:31`.
- Supporting jobs: manifest `1290455`, preflight `1290456`, seal `1290457`;
  all completed `0:0`.
- Result root:
  `/scratch/lmbanr001/masters/sallm/results/official_test/familywise_20260901/intent`.
- Sealed result-tree manifest SHA-256:
  `44a89fde7b63361e586a2135e1dc46a756032a6fad9b8613ac55174832534576`.
- Coverage: eight of eight Mono/Multi arms, 32 files, five prompts per arm;
  `3,110` English expanded rows and `3,200` per Xhosa, Zulu, and Southern
  Sotho arm.
- Every structural verifier passed with `metric_values_included=false`.

## Report-only held-out F1

| Mode | Language | Best prompt | Best | Prompt mean | Range |
|---|---|---:|---:|---:|---:|
| Mono | Eng | P3 | 0.003677697685 | 0.002600230386 | 0.001296300985–0.003677697685 |
| Mono | Xho | P2 | 0.002106732455 | 0.001645406342 | 0.001164725458–0.002106732455 |
| Mono | Zul | P5 | 0.004812623274 | 0.003035688504 | 0.001277955272–0.004812623274 |
| Mono | Sot | P4 | 0.004265373006 | 0.003178181500 | 0.001246105919–0.004265373006 |
| Multi | Eng | tied | 0.000990919938 | 0.000990919938 | 0.000990919938–0.000990919938 |
| Multi | Xho | P4 | 0.001232665639 | 0.001212689995 | 0.001160990712–0.001232665639 |
| Multi | Zul | P5 | 0.001223241590 | 0.001221002814 | 0.001219512195–0.001223241590 |
| Multi | Sot | tied | 0.001219512195 | 0.001219512195 | 0.001219512195–0.001219512195 |

Four-language macro prompt-mean F1 is `0.002614876682742` for Mono and
`0.001161031235439` for Multi. These scores are report-only and did not affect
any run, retry, checkpoint, prompt, selection, or correction.

## Sheet verification

Exact API readback of `GDN Results!C16:G19` confirmed the four `3 Sep` dates,
all Intent Mono and Multi strings, the unchanged Base values, and blank General
cells. Wrapping and date formatting are intact.
