# AfriHG English row removal

The canonical `Master's Datasets` Sheet contained placeholder English rows
for AfriHG even though no English AfriHG split exists. The exact rows were
deleted from:

- `Transformer Results`
- `XLSTM Results`
- `Qwen Results`
- `GDN Results`
- hidden `GDN Results template backup`

`Mamba Results`, `Comparison Data`, `Variant Comparison`, `Variant
Charts`, and `Language Charts` already represented AfriHG only for Xhosa and
Zulu.

Post-delete readback confirms the five result tables now end the AfriHG block
at Xhosa/Zulu, no inspected dependent formula contains `#REF!`, the AfriHG
variant chart remains present, and the language charts contain only AfriHG
Xhosa and Zulu.
