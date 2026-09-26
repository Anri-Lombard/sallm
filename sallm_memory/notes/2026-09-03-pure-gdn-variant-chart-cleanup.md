# Pure-GDN Variant Charts cleanup

Date: 2026-09-03

The obsolete `Architecture Comparison` tab (sheet ID `216129538`) was deleted
from the canonical `Master's Datasets` Google Sheet at the user's request.
Pre-deletion formula scans confirmed that `Comparison Data`, `Variant
Comparison`, `Variant Charts`, and `Language Charts` did not reference it.

Post-deletion metadata and exact cell readback confirmed that `Variant Charts`
remains present with all 11 charts. Its GDN series still read only the separate
GDN columns T:W from `Variant Comparison`; Qwen remains in P:S. The latest
terminal-valid News, NER, and T2X held-out results are visible. POS and AfriHG
adapted cells remain blank because jobs `1291776` and `1291874` are still
running and have no terminally verified bundle to promote.

