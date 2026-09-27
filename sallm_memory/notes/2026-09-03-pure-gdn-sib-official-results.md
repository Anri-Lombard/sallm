# Pure-GDN SIB official Mono/Multi results

Official job `1289212` completed `0:0` on A100-80GB in `00:18:42`. All twelve
frozen arms passed the structural verifier: six languages, separate Multi and
Mono adapters, five prompts, required metric `f1,none`, and exactly `1,020`
expanded rows per arm. All twelve structural-verification sidecars verify.
The complete 48-file result tree is sealed by manifest SHA-256
`ad98b6c0bbf2c03d101fbc79232654d4bfdb125abb317840290a18fc227136d9`.

Scores below are official held-out test results. Each value is the arithmetic
mean across the five frozen prompts for that arm.

| Language | Multi F1 | Mono F1 | Multi accuracy | Mono accuracy |
| --- | ---: | ---: | ---: | ---: |
| Afrikaans | 0.264285 | 0.153694 | 0.323529 | 0.234314 |
| English | 0.154927 | 0.087672 | 0.233333 | 0.179412 |
| Northern Sotho | 0.057005 | 0.049088 | 0.118627 | 0.122549 |
| Southern Sotho | 0.069037 | 0.051380 | 0.140196 | 0.134314 |
| Xhosa | 0.112343 | 0.043285 | 0.168627 | 0.121569 |
| Zulu | 0.163018 | 0.068227 | 0.220588 | 0.139216 |
| Six-language macro mean | 0.136769 | 0.075558 | 0.200817 | 0.155229 |

These scores were opened only after the entire twelve-arm bundle and all
structural hashes verified. They are report-only and must not influence any
remaining evaluation, selection, correction, prompt, or scheduling decision.
Sheet E/F/G remain blank until the complete obtainable non-General result set
and exact readback are ready.
