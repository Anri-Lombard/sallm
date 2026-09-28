"""Archive the 12 unchosen matched LR-probe runs (lrprobe_v1_*) to one PRIVATE HF repo, one folder per probe, then
remove each probe's local weights dir only after every one of its files verifies against the Hub (user decision
28 Sep 2026: archive, to free HEX scratch). The chosen probe of each architecture stays on scratch. Logs stay too.
Run inside a Slurm job on ada."""
import csv, hashlib, json, os, shutil, sys
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
RUNS = Path("/scratch/lmbanr001/masters/sallm/results/matched_pretrain_20260926/runs")
CHOSEN = {"gdn": "3e-3", "mamba2": "6e-3", "mzansilm": "3e-3", "xlstm": "3e-3"}
REPO = "anrilombard/sallm-matched-lrprobes"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
api = HfApi(token=Path("/home/lmbanr001/.huggingface/token").read_text().strip())
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()
def blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
api.create_repo(REPO, private=True, exist_ok=True)
assert api.repo_info(REPO).private, "repo is not private"
probes = []
for d in sorted(RUNS.glob("lrprobe_v1_*_lr*")):
    arch, lr = d.name[len("lrprobe_v1_"):].rsplit("_lr", 1)
    if CHOSEN.get(arch) != lr and (d / "weights").exists():
        probes.append((d, arch, lr))
assert len(probes) == 12, [p[0].name for p in probes]
with open(OUT / "manifest_lrprobes.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["probe", "file", "bytes", "sha256_or_blobsha1", "verified", "local_weights_removed"])
    for d, arch, lr in probes:
        wdir = next(x for x in (d / "weights").iterdir() if x.is_dir())
        files = {f"{d.name}/{wdir.name}/{n}": wdir / n for n in os.listdir(wdir)}
        files.update({f"{d.name}/{n}": d / n for n in ("run_meta.json", "train_log.jsonl", "event_log.jsonl") if (d / n).exists()})
        local = {k: (p.stat().st_size, sha256(p) if p.suffix in (".bin", ".safetensors") else blob_sha1(p.read_bytes())) for k, p in files.items()}
        def remote():
            info = {f.path: f for f in api.get_paths_info(REPO, list(local))}
            return {k: (info[k].size, info[k].lfs.sha256 if info[k].lfs else info[k].blob_id) if k in info else None for k in local}
        if remote() != local:
            api.create_commit(REPO, [CommitOperationAdd(k, str(p)) for k, p in files.items()], commit_message=f"{d.name} ({wdir.name})")
        ok = remote() == local
        removed = False
        if ok:
            shutil.rmtree(d / "weights"); removed = True
            (d / "ARCHIVED.json").write_text(json.dumps({"repo": REPO, "path": d.name, "files": {k: v[1] for k, v in local.items()}}, indent=1))
        for k, (n, h) in local.items():
            w.writerow([d.name, k, n, h, ok, removed])
        print(d.name, "verified" if ok else "NOT VERIFIED", "removed" if removed else "kept", flush=True)
print("DONE")
