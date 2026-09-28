"""Reformat matched_pretrain_v1 (already packed + shuffled 2048-token chunks in parquet) into pretrain.py's train.bin,
preserving the builder's chunk order exactly (shard order, row order). Read-only on the source dir. CPU only.
Also fetches the held-out split (uctnlp/mzansi-text-deduplicated validation @08bee62)."""
import glob, hashlib, json, sys
from pathlib import Path
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
files = sorted(glob.glob(str(src / "train/train-*.parquet")))
schema = pq.read_schema(files[0])
print(schema)
col = next(f.name for f in schema if str(f.type).startswith(("list", "large_list", "fixed_size_list")))
langs = json.loads((src / "manifest.json").read_text())["packing"]["languages"]
order_lang = np.load(src / "chunk_order_lang.npy")
h, n, k, lang_ok = hashlib.sha256(), 0, 0, True
with open(out / "train.bin", "wb") as fo:
    for f in files:
        t = pq.read_table(f)
        arr = t.column(col).combine_chunks()
        flat = pc.list_flatten(arr).to_numpy()
        assert len(flat) == 2048 * len(arr) and flat.max() < 65536 and flat.min() >= 0
        if "lang" in t.column_names:
            lang_ok &= [langs.index(x) for x in t.column("lang").to_pylist()] == order_lang[k:k + len(arr)].tolist()
        b = flat.astype(np.uint16).tobytes()
        fo.write(b); h.update(b); n += len(flat); k += len(arr)
meta = {"n_tokens": n, "n_chunks": k, "seq_len": 2048, "source": str(src), "id_column": col, "files": [Path(f).name for f in files],
        "train_bin_sha256": h.hexdigest(), "chunk_lang_matches_chunk_order_lang_npy": bool(lang_ok),
        "manifest_sha256": hashlib.sha256((src / "manifest.json").read_bytes()).hexdigest(),
        "note": "order = builder's shuffled chunk order (numpy default_rng(20260926).permutation); EOS-separated docs, no BOS"}
(out / "meta.json").write_text(json.dumps(meta, indent=2))
print(meta)
from huggingface_hub import hf_hub_download, list_repo_files
fs = [x for x in list_repo_files("uctnlp/mzansi-text-deduplicated", repo_type="dataset", revision="08bee62b1cee62a096628b53535145c5ce617421") if "validation" in x and x.endswith(".parquet")]
print("validation files", fs)
import pandas as pd
df = pd.concat([pd.read_parquet(hf_hub_download("uctnlp/mzansi-text-deduplicated", x, repo_type="dataset", revision="08bee62b1cee62a096628b53535145c5ce617421")) for x in sorted(fs)])
print(df.columns.tolist(), len(df), df["lang"].value_counts().to_dict())
df[["text", "lang"]].reset_index(drop=True).to_parquet(out / "validation.parquet")
