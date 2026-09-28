"""Pack document-tokenized parquet shards into the pretrain.py data format (CPU only).

Output DIR/train.bin: flat uint16 stream of whole documents in a seeded document permutation;
DIR/meta.json: n_tokens, n_docs, seed, sha256 of train.bin and of the permutation, source files.
Every row must already be [BOS] ... [EOS] token ids < 65536.

Bench:  python pack.py --repo uctnlp/mzansi-text-tokenized --rev 7dab1eb... --files 3 --out DIR
Matched run: point --parquet-glob at the cleaned dataset's tokenized train shards (untruncated).
"""
import argparse, glob, hashlib, json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--parquet-glob")
ap.add_argument("--repo")
ap.add_argument("--rev")
ap.add_argument("--files", type=int, default=None, help="use only the first N train shards (bench)")
ap.add_argument("--seed", type=int, default=20260926)
a = ap.parse_args()
if a.repo:
    from huggingface_hub import snapshot_download
    root = snapshot_download(a.repo, repo_type="dataset", revision=a.rev, allow_patterns=["data/train-*.parquet"])
    a.parquet_glob = f"{root}/data/train-*.parquet"
files = sorted(glob.glob(a.parquet_glob))[: a.files]
docs = []
for f in files:
    col = pq.read_table(f, columns=["input_ids"]).column("input_ids").combine_chunks()
    flat = pc.list_flatten(col).to_numpy()
    assert flat.max() < 65536 and flat.min() >= 0
    off = col.offsets.to_numpy()
    new = [flat[off[i]:off[i + 1]].astype(np.uint16) for i in range(len(off) - 1)]
    assert all(d[0] == 0 and (d[-1] == 1 or len(d) == 2048) for d in new)  # [BOS]=0 first; [EOS]=1 unless truncated at 2048
    docs += new
perm = np.random.default_rng(a.seed).permutation(len(docs))
a.out.mkdir(parents=True, exist_ok=True)
h = hashlib.sha256()
n = 0
with open(a.out / "train.bin", "wb") as fo:
    for i in perm:
        b = docs[i].tobytes()
        fo.write(b)
        h.update(b)
        n += len(docs[i])
meta = {"n_tokens": n, "n_docs": len(docs), "seed": a.seed, "train_bin_sha256": h.hexdigest(),
        "perm_sha256": hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest(),
        "sources": [Path(f).name for f in files], "repo": a.repo, "rev": a.rev}
(a.out / "meta.json").write_text(json.dumps(meta, indent=2))
print(meta)
