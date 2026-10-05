"""Back up the epoch-1 fine-tuned checkpoints (runs/<arch>/keep) to a private HF repo and verify each one (2 Oct 2026,
scratch quota). Default set: Multi, Multitask (general) and generation (T2X, AfriHG) checkpoints, ~44 GB; --mono adds the
seed-42 Mono classification ones (~52 GB, once there is HF room). A checkpoint counts as backed up once every file of its
<ckpt>.sha256 manifest matches both the local file and the Hub copy (LFS sha256, or git blob sha1 for small files);
then <ckpt>.on_hf is written next to it. Nothing here deletes. Run in a Slurm job (CPU partition ada), not on the login node:
  HF_TOKEN_PATH=~/.huggingface/token python backup_to_hf.py [--mono]"""
import hashlib
import json
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "anrilombard/sallm-ft-epoch1-checkpoints"
RUNS = Path("/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926/runs")
ARCHS = ("mzansilm", "mamba2", "xlstm", "gdn")


def digest(p: Path, algo: str, git: bool = False) -> str:
    h = hashlib.new(algo)
    if git:
        h.update(f"blob {p.stat().st_size}\0".encode())
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def wanted(mono: bool):
    for a in ARCHS:
        for d in sorted((RUNS / a / "keep").glob("*/*_e[0-9]*")):
            if d.is_dir() and ("-multi-" in d.name or d.parent.name in ("general", "t2x", "afrihg") or mono):
                yield a, d


def verify(api: HfApi, prefix: str, d: Path) -> list[str]:
    man = {rel: h for h, rel in (line.split("  ", 1) for line in d.with_name(d.name + ".sha256").read_text().splitlines())}
    # some kept checkpoints gained tokenizer files after their manifest was written: those are checked against the local file
    local = sorted(f"{d.name}/{p.relative_to(d).as_posix()}" for p in d.rglob("*") if p.is_file())
    bad = [f"in manifest, missing locally: {x}" for x in set(man) - set(local)]
    remote = {f.path: f for f in api.get_paths_info(REPO, [f"{prefix}/{rel}" for rel in local], expand=True)}
    for rel in local:
        p, r = d.parent / rel, remote.get(f"{prefix}/{rel}")
        h = digest(p, "sha256")
        if h != man.get(rel, h):
            bad.append(f"local differs from manifest: {rel}")
        elif r is None:
            bad.append(f"missing on hub: {rel}")
        elif (r.lfs.sha256 if r.lfs else r.blob_id) != (h if r.lfs else digest(p, "sha1", git=True)):
            bad.append(f"hub differs: {rel}")
    return bad


def main() -> None:
    api = HfApi()
    api.create_repo(REPO, private=True, exist_ok=True)
    assert api.model_info(REPO).private, "repo must be private"
    todo = [(a, d) for a, d in wanted("--mono" in sys.argv) if not d.with_name(d.name + ".on_hf").exists()]
    print(f"{len(todo)} checkpoints to back up", flush=True)
    for a, d in todo:
        prefix = f"{a}/{d.parent.name}"
        # one commit per checkpoint (folder + manifest + marker): the Hub allows 128 commits per hour
        files = [p for p in sorted(d.rglob("*")) if p.is_file()] + [m for ext in (".sha256", ".selected") if (m := d.with_name(d.name + ext)).exists()]
        ops = [CommitOperationAdd(path_in_repo=f"{prefix}/{p.relative_to(d.parent).as_posix()}", path_or_fileobj=str(p)) for p in files]
        k = 0
        while k < 5:
            try:
                api.create_commit(repo_id=REPO, operations=ops, commit_message=f"add {prefix}/{d.name}")
                bad = verify(api, prefix, d)
                break
            except Exception as e:  # rate limit: wait it out without using up a retry; other errors: retry the checkpoint
                if "storage limit" in str(e):  # quota full: stop, the rest stays on HEX
                    sys.exit(f"STORAGE LIMIT at {prefix}/{d.name}")
                limited = "rate limit" in str(e)
                print(f"{'rate-limited, waiting' if limited else f'retry {k + 1}'} {prefix}/{d.name}: {str(e)[:200]}", flush=True)
                time.sleep(900 if limited else 60 * (k + 1))
                k += not limited
        else:
            print(f"FAILED {prefix}/{d.name}", flush=True)
            continue
        if bad:
            print(f"MISMATCH {prefix}/{d.name}: {bad}", flush=True)
            continue
        d.with_name(d.name + ".on_hf").write_text(json.dumps({"repo": REPO, "path": f"{prefix}/{d.name}", "time": time.time()}) + "\n")
        print(f"verified {prefix}/{d.name}", flush=True)


if __name__ == "__main__":
    main()
