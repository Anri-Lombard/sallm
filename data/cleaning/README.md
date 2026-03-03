# MzansiText Cleaning Pipeline

This folder contains the exact Datatrove-based cleaning pipeline used to
produce the filtered `mzansi-text` corpus released with the LREC 2026 paper.

## Files

- `clean_mzansi_text.py`: filtering and optional concatenation pipeline
- `config.toml.example`: example runtime configuration with placeholder paths

## Usage

```bash
cp data/cleaning/config.toml.example data/cleaning/config.toml
# edit the paths in data/cleaning/config.toml

python data/cleaning/clean_mzansi_text.py \
  --config data/cleaning/config.toml \
  --concat
```

The pipeline applies tuned Gopher, C4, and FineWeb quality filters, then
optionally concatenates short documents to around 1000 tokens.
