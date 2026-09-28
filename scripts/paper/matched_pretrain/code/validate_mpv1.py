"""Pre-launch validation of matched_pretrain_v1 and its train.bin reformat (CPU only, run in a Slurm job)."""
import csv, glob, hashlib, json, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

src, conv, tokdir = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
fails = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name, detail, flush=True)
    if not ok: fails.append(name)

man = json.loads((src / "manifest.json").read_text())
bad = [f for f, v in man["files"].items() if hashlib.sha256((src / f).read_bytes()).hexdigest() != v["sha256"]]
check("manifest file sha256", not bad, f"{len(man['files'])} files checked; mismatches={bad}")

meta = json.loads((conv / "meta.json").read_text())
tok = np.memmap(conv / "train.bin", dtype=np.uint16, mode="r")
N = len(tok) // 2048
check("total tokens == 2,839,511,040", len(tok) == 2839511040 == meta["n_tokens"], f"len={len(tok)}")
check("chunks x 2048 == tokens", N * 2048 == len(tok) and N == meta["n_chunks"], f"chunks={N}")

langs = man["packing"]["languages"]
order_lang = np.load(src / "chunk_order_lang.npy")
check("chunk_order_lang length == chunks", len(order_lang) == N)
counts = np.zeros(65536, np.int64)
eos_per_lang = np.zeros(len(langs), np.int64)
double_eos = 0
step = 20000
for i in range(0, N, step):
    blk = np.asarray(tok[i * 2048:(i + step) * 2048]).reshape(-1, 2048)
    counts += np.bincount(blk.ravel(), minlength=65536)
    np.add.at(eos_per_lang, order_lang[i:i + len(blk)], (blk == 1).sum(1))
    double_eos += int(((blk[:, 1:] == 1) & (blk[:, :-1] == 1)).sum())
check("no BOS(0) in chunks", counts[0] == 0, f"count={counts[0]}")
check("no PAD(2) in chunks", counts[2] == 0, f"count={counts[2]}")
print("UNK(3) count", counts[3], "| EOS(1) count", counts[1], "| max id", int(np.nonzero(counts)[0].max()))
check("no empty documents (EOS EOS)", double_eos == 0, f"adjacent EOS pairs={double_eos}")
# EOS once per document: EOS count per language vs kept documents (minus the documents cut off by the dropped tail, <=1 per lang)
rows = {c["lang"]: c for c in man["counts"]}
for k, l in enumerate(langs):
    exp = rows[l].get("after_decontam_rows")
    got = int(eos_per_lang[k])
    check(f"EOS count {l} ~ kept docs", exp is not None and exp - 2 <= got <= exp, f"eos={got} docs={exp}")

# per-language token share vs per_language_tokens.csv (chunk_tokens) and the lang column
ref = {r["lang"]: int(r["chunk_tokens"]) for r in csv.DictReader(open(src / "per_language_epoch_tokens.csv"))}
mine = {l: int((order_lang == k).sum()) * 2048 for k, l in enumerate(langs)}
check("per-language chunk tokens == per_language_epoch_tokens.csv", mine == ref, json.dumps({l: round(v / len(tok), 4) for l, v in mine.items()}))
lang_col = []
for f in sorted(glob.glob(str(src / "train/train-*.parquet"))):
    lang_col += pq.read_table(f, columns=["lang"]).column("lang").to_pylist()
check("lang column == chunk_order_lang.npy", [langs.index(x) for x in lang_col] == order_lang.tolist())

# chunk order == recorded seeded permutation: reconstruct pre-shuffle index from (lang, lang_chunk_index)
lci = np.concatenate([pq.read_table(f, columns=["lang_chunk_index"]).column(0).to_numpy()
                      for f in sorted(glob.glob(str(src / "train/train-*.parquet")))])
off = np.concatenate([[0], np.cumsum([int((order_lang == k).sum()) for k in range(len(langs))])])
g = off[order_lang] + lci
perm = np.random.default_rng(man["packing"]["seed"]).permutation(N)
check("chunk order == default_rng(20260926).permutation(N) over lang-concatenated chunks", np.array_equal(g, perm),
      f"first 5 pre-shuffle idx={g[:5].tolist()} perm={perm[:5].tolist()}")
check("train.bin row order == parquet row order", all(
    np.array_equal(np.asarray(tok[i * 2048:(i + 1) * 2048]), np.asarray(
        pq.read_table(sorted(glob.glob(str(src / 'train/train-*.parquet')))[0], columns=["input_ids"]).column(0)[i].as_py(), np.uint16))
    for i in (0, 1, 777)))

# DDP: every rank reads disjoint blocks; the union per step is exactly blocks [s*192, (s+1)*192)
B = 192
for world, mb in ((2, 12), (2, 24), (4, 12), (4, 24), (1, 8)):
    ga = B // (mb * world)
    starts = np.array([(m * world + r) * mb for m in range(ga) for r in range(world)])
    idx = (starts[:, None] + np.arange(mb)).ravel()
    check(f"DDP disjoint+complete world={world} mb={mb}", np.array_equal(np.sort(idx), np.arange(B)))

# decode 5 random chunks per major language
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(tokdir)
rng = np.random.default_rng(0)
for l in ("afr", "eng", "zul", "xho", "sot"):
    for i in rng.choice(np.nonzero(order_lang == langs.index(l))[0], 5, replace=False):
        s = t.decode(np.asarray(tok[i * 2048:(i + 1) * 2048]).tolist(), skip_special_tokens=False)
        print(f"--- {l} chunk {i}: {s[:300]!r}")
print("VALIDATION", "FAILED: " + ", ".join(fails) if fails else "ALL PASS")
