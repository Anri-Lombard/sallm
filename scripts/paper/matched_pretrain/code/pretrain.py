"""Matched-budget pretraining for the four ~125M decoders (one script, one recipe).

Launch: torchrun --nproc_per_node=G pretrain.py --config configs/<arch>_<sched>.json --data DIR --out DIR [...]

Identical for every architecture: packed data stream and order, sequence length, global batch
(sequences), token budget, warmup fraction, schedule shape and floor, optimizer, weight-decay rule,
precision (fp32 master weights + bf16 autocast), loss (plain F.cross_entropy on the model's logits),
seed, eval scorer and checkpoint token points. Only the model class/config and peak LR come from the
per-architecture config.

Data (DIR): train.bin = flat uint16 stream of whole documents ([BOS] ... [EOS]) already shuffled at
document level; meta.json = {"n_tokens": ...}. The stream is cut into seq_len blocks; the global batch
at step s is blocks [s*B, (s+1)*B) independent of GPU count / micro-batch. Epoch e>=1 re-orders blocks
with rng(seed+e).permutation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

LN2 = math.log(2.0)


# ----------------------------------------------------------------------------- models
def build_model(cfg: dict):
    arch, mc = cfg["arch"], dict(cfg["model_config"])
    if arch == "mzansilm":
        from transformers import LlamaConfig, LlamaForCausalLM
        model = LlamaForCausalLM(LlamaConfig(**mc, attn_implementation=cfg.get("attn_implementation", "sdpa")))
    elif arch == "mamba2":
        from fla.models import Mamba2Config, Mamba2ForCausalLM
        model = Mamba2ForCausalLM(Mamba2Config(**mc))
    elif arch == "gdn":
        from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
        model = GatedDeltaNetForCausalLM(GatedDeltaNetConfig(**mc))
    elif arch == "xlstm":
        from transformers import xLSTMConfig, xLSTMForCausalLM
        c = xLSTMConfig(**mc)
        if cfg.get("xlstm_chunkwise_kernel"):  # set after construction: the HF config validator only knows native kernels
            c.chunkwise_kernel = cfg["xlstm_chunkwise_kernel"]
        cls = type(c)  # exact head dims (as the retained xLSTM and the paper loader)
        cls.qk_dim = property(lambda s: int(s.hidden_size * s.qk_dim_factor))
        cls.v_dim = property(lambda s: int(s.hidden_size * s.v_dim_factor))
        model = xLSTMForCausalLM(c)
    else:
        raise ValueError(arch)
    n = sum(p.numel() for p in model.parameters())
    assert n == cfg["expected_params"], (arch, n, cfg["expected_params"])
    return model


def param_groups(model, wd):
    # one rule for all four: decay every >=2-D tensor (matrices, embeddings, conv kernels), not 1-D (norms, biases, A_log, D, dt_bias)
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return [{"params": decay, "weight_decay": wd}, {"params": no, "weight_decay": 0.0}]


def flops_per_token(model, cfg) -> dict:
    """Analytical model FLOPs per training token (forward x3). Dense: 2 FLOPs per multiply-accumulate of every
    nn.Linear weight (incl. the tied LM head) + depthwise short convs. Sequence mixing: algorithmic forward cost of
    the causal mixer at context T (see report.md for derivations); recomputation inside kernels is not counted."""
    import torch.nn as nn
    lin = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))
    conv = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Conv1d))
    c, T = cfg["model_config"], cfg["seq_len"]
    arch = cfg["arch"]
    if arch == "mzansilm":  # Kaplan et al. 2020: 2*n_layer*n_ctx*d_attn (QK^T + AV, causal-halved)
        d_attn = c["num_attention_heads"] * c["head_dim"]
        mix = 2 * c["num_hidden_layers"] * T * d_attn
    elif arch == "mamba2":  # SSD chunked (Dao & Gu 2024): G*Q*N (CB^T) + H*Q*P (masked mix) + 4*H*N*P (states in/out)
        Q, N, P = c["chunk_size"], c["state_size"], c["head_dim"]
        H, G = c["expand"] * c["hidden_size"] // P, c["n_groups"]
        mix = c["num_hidden_layers"] * (G * Q * N + H * Q * P + 4 * H * N * P)
    elif arch == "gdn":  # chunked delta rule (Yang et al. 2024/25): 3*C*dk (KK^T,QK^T,W) + 2*C*dv (U, attn*v) + 6*dk*dv (WS, QS, K^T v)
        C, H, dk = 64, c["num_heads"], c["head_dim"]
        dv = dk * c["expand_v"]
        mix = c["num_hidden_layers"] * H * (3 * C * dk + 2 * C * dv + 6 * dk * dv)
    elif arch == "xlstm":  # mLSTM chunkwise: C*dqk (QK^T) + C*dv (mix) + 4*dqk*dv (state update + readout)
        C, H, d = c["chunk_size"], c["num_heads"], c["hidden_size"]
        dqk, dv = int(d * c["qk_dim_factor"]) // H, int(d * c["v_dim_factor"]) // H
        mix = c["num_blocks"] * H * (C * dqk + C * dv + 4 * dqk * dv)
    fwd_dense = 2 * lin + 2 * conv
    return {"linear_params": lin, "conv_params": conv, "fwd_dense": fwd_dense, "fwd_mix": mix,
            "train_dense": 3 * fwd_dense, "train_mix": 3 * mix, "train_total": 3 * (fwd_dense + mix)}


class LMLoss(torch.nn.Module):
    """Next-token loss, identical definition for all four. baseline: model logits -> fp32 F.cross_entropy.
    fused: final hidden state -> one shared Liger fused-linear-CE (fp32 accumulation) against the tied LM head,
    with the model's own logit soft-cap (xLSTM 30) applied inside the kernel."""

    def __init__(self, model, fused: bool, compile_: bool = False):
        super().__init__()
        self.model, self.fused = model, fused
        body = getattr(model, "model", None) or model.backbone
        # compile only the backbone: Liger's autograd.Function fails under dynamo (addmm out_dtype); eval uses the eager model
        self.body = torch.compile(body) if compile_ else body
        if fused:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            cap = getattr(model.config, "output_logit_soft_cap", None)
            self.flce = LigerFusedLinearCrossEntropyLoss(softcap=cap, accum_dtype=torch.float32)

    def forward(self, x):
        if not self.fused:
            logits = self.model(input_ids=x, use_cache=False).logits
            return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        h = self.body(input_ids=x, use_cache=False).last_hidden_state
        return self.flce(self.model.get_output_embeddings().weight, h[:, :-1].reshape(-1, h.size(-1)), x[:, 1:].reshape(-1))


# ----------------------------------------------------------------------------- schedule
def lr_at(step: int, s: dict, peak: float, total: int, warmup: int) -> float:
    floor = s["min_lr_ratio"] * peak
    if step < warmup:
        return peak * (step + 1) / warmup
    if s["type"] == "cosine":
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * p))
    if s["type"] == "wsd":  # Hagele et al. 2024: constant, then (1-sqrt) cooldown over the last decay_frac of the budget
        decay = round(s["decay_frac"] * total)
        start = total - decay
        if step < start:
            return peak
        p = min(1.0, (step - start) / max(1, decay))
        return floor + (peak - floor) * (1 - math.sqrt(p))
    raise ValueError(s["type"])


# ----------------------------------------------------------------------------- data
class Blocks:
    def __init__(self, data_dir: Path, seq: int, seed: int):
        meta = json.loads((data_dir / "meta.json").read_text())
        self.tok = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
        assert len(self.tok) == meta["n_tokens"], (len(self.tok), meta["n_tokens"])
        self.seq, self.seed, self.meta = seq, seed, meta
        self.n_blocks = len(self.tok) // seq
        self._perm = {}

    def phys(self, gb: int) -> int:
        e, j = divmod(gb, self.n_blocks)
        if e == 0:
            return j
        if e not in self._perm:
            self._perm[e] = np.random.default_rng(self.seed + e).permutation(self.n_blocks)
        return int(self._perm[e][j])

    def get(self, first_gb: int, n: int) -> torch.Tensor:
        rows = [self.tok[self.phys(g) * self.seq:(self.phys(g) + 1) * self.seq] for g in range(first_gb, first_gb + n)]
        return torch.from_numpy(np.stack(rows).astype(np.int64))


# ----------------------------------------------------------------------------- eval (set1 scorer)
class HeldOut:
    """dissertation/set1_heldout_lm scorer: whole-document scoring, windows of <=1024 inputs overlapping by
    one token, right-pad to x64 with [PAD], <=8192 tokens/forward, no mask, fp32 CE on logits. Reports text-token
    CE (final [EOS] prediction excluded) and BPB over NFD-detokenised UTF-8 bytes, pooled and per language."""

    W, PAD_MULT, BUDGET = 1024, 64, 8192

    def __init__(self, parquet: str, tokenizer: str, limit_per_lang: int | None = None, doc_start: str = "bos"):
        import pandas as pd
        from tokenizers.decoders import ByteLevel
        from transformers import AutoTokenizer
        df = pd.read_parquet(parquet)
        if limit_per_lang:
            df = df.groupby("lang").head(limit_per_lang).reset_index(drop=True)
        tok = AutoTokenizer.from_pretrained(tokenizer)
        tok.backend_tokenizer.decoder = ByteLevel()
        assert (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id) == (0, 1, 2)
        enc = tok(df["text"].tolist(), add_special_tokens=True)["input_ids"]  # [BOS] text [EOS]
        if doc_start == "eos":  # data trained without [BOS] (EOS-separated stream): a document starts after an [EOS]
            enc = [[tok.eos_token_id] + e[1:] for e in enc]
        self.pad = tok.pad_token_id
        self.langs = sorted(df["lang"].unique())
        self.doc_lang = np.array([self.langs.index(x) for x in df["lang"]])
        self.n_text = np.array([len(e) - 2 for e in enc], dtype=np.float64)
        self.bytes = np.array([len(tok.decode(e[1:-1], skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False).encode()) for e in enc], dtype=np.float64)
        items = []
        for d, ids in enumerate(enc):
            start = 0
            while True:
                items.append((d, start, ids[start:start + self.W], len(ids)))
                if start + self.W >= len(ids):
                    break
                start += self.W - 1
        items.sort(key=lambda x: len(x[2]))
        self.batches, i = [], 0
        rup = lambda n: -(-n // self.PAD_MULT) * self.PAD_MULT  # noqa: E731
        while i < len(items):
            j = i
            while j < len(items) and ((j - i + 1) * rup(len(items[j][2])) <= self.BUDGET or j == i):
                j += 1
            self.batches.append(items[i:j])
            i = j
        self.n_docs, self.n_windows = len(enc), len(items)

    @torch.no_grad()
    def run(self, model, rank: int, world: int, device) -> dict:
        was = model.training
        model.eval()
        text = torch.zeros(self.n_docs, dtype=torch.float64, device=device)
        preds = torch.zeros(self.n_docs, dtype=torch.float64, device=device)
        for bi in range(rank, len(self.batches), world):
            batch = self.batches[bi]
            Lp = -(-max(len(w) for _, _, w, _ in batch) // self.PAD_MULT) * self.PAD_MULT
            x = torch.full((len(batch), Lp), self.pad, dtype=torch.long)
            for b, (_, _, w, _) in enumerate(batch):
                x[b, :len(w)] = torch.tensor(w)
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x, use_cache=False).logits
            lp = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1),
                                 reduction="none").view(len(batch), Lp - 1)
            for b, (d, start, w, n_doc) in enumerate(batch):
                n = len(w) - 1
                ends_doc = start + len(w) == n_doc  # last window holds the final [EOS] prediction
                text[d] += lp[b, :n - 1 if ends_doc else n].double().sum()
                preds[d] += n
            del logits, lp
        dist.all_reduce(text)
        dist.all_reduce(preds)
        assert torch.equal(preds.cpu(), torch.tensor(self.n_text + 1)), "each token must be predicted exactly once"
        text = text.cpu().numpy()
        out = {}
        for k, lang in enumerate(self.langs + ["ALL"]):
            m = np.ones(self.n_docs, bool) if lang == "ALL" else self.doc_lang == k
            ce = text[m].sum() / self.n_text[m].sum()
            out[f"{lang}/ce"], out[f"{lang}/bpb"] = ce, text[m].sum() / LN2 / self.bytes[m].sum()
        out["MACRO/bpb"] = float(np.mean([out[f"{x}/bpb"] for x in self.langs]))
        out["MACRO/ce"] = float(np.mean([out[f"{x}/ce"] for x in self.langs]))
        model.train(was)
        return out


# ----------------------------------------------------------------------------- checkpoints
def save_weights(model, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    model.save_pretrained(tmp, safe_serialization=False)  # tied embeddings (FLA Mamba2, xLSTM) are rejected by safetensors
    tmp.rename(path)


def save_resume(model, opt, step, tokens, path: Path, extra: dict):
    tmp = path.with_name(path.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "step": step, "tokens": tokens,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), **extra}, tmp / "state.pt")
    model.config.save_pretrained(tmp)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--budget-epochs", type=float, default=None, help="override config budget (WSD extension/branches)")
    ap.add_argument("--resume", type=Path, default=None, help="resume-format checkpoint dir (state.pt)")
    ap.add_argument("--stop-before-decay", action="store_true", help="WSD: save stable ckpt with optimizer at decay start and exit")
    ap.add_argument("--max-steps", type=int, default=None, help="benchmark: stop after this many steps")
    ap.add_argument("--max-minutes", type=float, default=None, help="benchmark: stop after this much training wall time")
    ap.add_argument("--eval-at", type=str, default=None, help="benchmark: comma list of steps to eval at (overrides schedule)")
    ap.add_argument("--eval-limit-per-lang", type=int, default=None)
    ap.add_argument("--stack", choices=["baseline", "fast"], default="baseline",
                    help="baseline: eager, logits+CE, foreach AdamW, TF32 off. fast: shared Liger FLCE, fused AdamW, TF32 on, torch.compile")
    ap.add_argument("--no-compile", action="store_true", help="fast stack without torch.compile")
    ap.add_argument("--xlstm-kernel", default=None, help="e.g. chunkwise--triton_xl_chunk")
    ap.add_argument("--micro-batch", type=int, default=None, help="hardware knob only; global batch is fixed")
    ap.add_argument("--total-tokens", type=float, default=None, help="probe: budget in tokens (overrides epochs); schedule defined over it")
    ap.add_argument("--warmup-tokens", type=float, default=None, help="probe: warmup length in tokens (overrides warmup_frac)")
    ap.add_argument("--peak-lr", type=float, default=None, help="probe: override peak LR")
    ap.add_argument("--resume-every", type=int, default=None, help="override resume_every_steps")
    ap.add_argument("--no-step0-eval", action="store_true")
    ap.add_argument("--eval-doc-start", choices=["bos", "eos"], default=None, help="token placed before each held-out doc (match training data)")
    ap.add_argument("--weights-final-only", action="store_true", help="probe: skip the intermediate weights-only checkpoints")
    ap.add_argument("--bench-save", action="store_true", help="benchmark: time one weights-only and one resume save at the end")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.peak_lr:
        cfg["optim"]["peak_lr"] = a.peak_lr
    if a.eval_doc_start:
        cfg["eval_doc_start"] = a.eval_doc_start
    if a.resume_every is not None:
        cfg["resume_every_steps"] = a.resume_every
    cfg["eval_parquet"], cfg["tokenizer"] = os.path.expandvars(cfg["eval_parquet"]), os.path.expandvars(cfg["tokenizer"])

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    a.out.mkdir(parents=True, exist_ok=True)

    fast = a.stack == "fast"
    if a.xlstm_kernel:
        cfg["xlstm_chunkwise_kernel"] = a.xlstm_kernel
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = fast
    B, T, mb = cfg["global_batch_seqs"], cfg["seq_len"], a.micro_batch or cfg["micro_batch"]
    assert B % (mb * world) == 0, (B, mb, world)
    GA = B // (mb * world)
    data = Blocks(a.data, T, cfg["seed"])
    epochs = a.budget_epochs if a.budget_epochs is not None else cfg["budget_epochs"]
    tok_per_step = B * T
    total = math.ceil(a.total_tokens / tok_per_step) if a.total_tokens else int(data.n_blocks * epochs) // B
    assert not a.total_tokens or total * B <= data.n_blocks, "probe budget must fit in one pass of the data"
    warmup = (round(a.warmup_tokens / tok_per_step) if a.warmup_tokens
              else round(cfg["schedule"]["warmup_frac"] * (data.n_blocks // B)))  # fixed by the 1-epoch step count
    decay_start = total - round(cfg["schedule"].get("decay_frac", 0) * total) if cfg["schedule"]["type"] == "wsd" else None

    torch.manual_seed(cfg["seed"])
    model = build_model(cfg).to(dev)  # fp32 master weights
    opt_cfg = cfg["optim"]
    opt = torch.optim.AdamW(param_groups(model, opt_cfg["weight_decay"]), lr=opt_cfg["peak_lr"],
                            betas=tuple(opt_cfg["betas"]), eps=opt_cfg["eps"], fused=fast)
    step = 0
    if a.resume:
        st = torch.load(a.resume / "state.pt", map_location=dev, weights_only=True)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optim"])
        step = st["step"]
        torch.set_rng_state(st["torch_rng"].cpu())
        torch.cuda.set_rng_state(st["cuda_rng"].cpu())
    fl = flops_per_token(model, cfg)
    cfg["flops_per_token"] = fl["train_total"]
    ddp = torch.nn.parallel.DistributedDataParallel(LMLoss(model, fused=fast, compile_=fast and not a.no_compile), device_ids=[local], gradient_as_bucket_view=True)
    fwd = ddp
    if fast and not a.no_compile:  # FLA layers recompile per layer_idx; the default limit (8) leaves later layers eager
        import torch._dynamo as dynamo  # (a bare `import torch._dynamo` would make `torch` local to main)
        dynamo.config.cache_size_limit = dynamo.config.recompile_limit = 64
    kernels = kernel_report(model, cfg["arch"])

    # eval / checkpoint points in steps (token targets -> first step whose cumulative tokens reach them)
    # every 100M tokens up to 1B, then every 250M (config), plus the final step
    pts = [t for t in range(int(cfg["eval_every_tokens"]), int(1e9) + 1, int(cfg["eval_every_tokens"]))]
    t = 1e9 + cfg.get("eval_every_tokens_after_1b", cfg["eval_every_tokens"])
    while t <= total * tok_per_step:
        pts.append(t)
        t += cfg.get("eval_every_tokens_after_1b", cfg["eval_every_tokens"])
    auto = {math.ceil(t / tok_per_step) for t in pts if math.ceil(t / tok_per_step) <= total} | {total}
    eval_steps = {int(x) for x in a.eval_at.split(",")} if a.eval_at else auto
    ck_steps = {math.ceil(t / tok_per_step): t for t in cfg["weights_at_tokens"] if math.ceil(t / tok_per_step) <= total}
    if a.weights_final_only:
        ck_steps = {}
    ck_steps[total] = "final"
    resume_every = cfg["resume_every_steps"]
    heldout = HeldOut(cfg["eval_parquet"], cfg["tokenizer"], a.eval_limit_per_lang, cfg.get("eval_doc_start", "bos"))

    run_meta = {"cfg": cfg, "args": {k: str(v) for k, v in vars(a).items()}, "world": world, "grad_accum": GA,
                "n_blocks_per_epoch": data.n_blocks, "total_steps": total, "warmup_steps": warmup,
                "decay_start": decay_start, "tokens_per_step": tok_per_step, "data_meta": data.meta,
                "eval_steps_n": len(eval_steps), "weights_ckpt_steps": {str(k): v for k, v in ck_steps.items()},
                "env": env_info(), "flops": fl, "micro_batch": mb, "stack": a.stack,
                "compile": fast and not a.no_compile, "kernels": kernels, "heldout": {"docs": heldout.n_docs, "windows": heldout.n_windows,
                                               "langs": heldout.langs}}
    wb = None
    if rank == 0:
        (a.out / ("run_meta.json" if step == 0 else f"run_meta_resumed_step{step}.json")).write_text(json.dumps(run_meta, indent=2, default=str))
        if os.environ.get("WANDB_MODE") != "disabled":
            import wandb
            wb = wandb.init(project=cfg.get("wandb_project", "sallm-matched-pretrain"), name=a.out.name,
                            dir=str(a.out), config=run_meta, resume="allow", id=a.out.name.replace("/", "-")[-64:])
    tlog = open(a.out / "train_log.jsonl", "a") if rank == 0 else None
    elog = open(a.out / "event_log.jsonl", "a") if rank == 0 else None

    def event(kind, **kw):
        if rank == 0:
            rec = {"kind": kind, "step": step, "tokens": step * tok_per_step, "flops": step * tok_per_step * cfg["flops_per_token"], **kw}
            elog.write(json.dumps(rec) + "\n")
            elog.flush()
            if wb:
                wb.log({f"{kind}/{k}": v for k, v in kw.items() if isinstance(v, (int, float))}
                       | {"tokens": rec["tokens"], "flops": rec["flops"]}, step=step)

    def do_eval():
        torch.cuda.synchronize()
        t0 = time.time()
        r = heldout.run(model, rank, world, dev)
        torch.cuda.synchronize()
        event("eval", seconds=time.time() - t0, **r)

    if step == 0 and not a.eval_at and not a.no_step0_eval:
        do_eval()  # untrained reference point
    ddp.train()
    from contextlib import nullcontext
    attn_ctx = nullcontext  # SDPA picks its backend; forcing FLASH fails for HF Llama GQA (no kernel available)
    t_train, t_start = 0.0, time.time()
    stop_at = decay_start if a.stop_before_decay else total
    if a.max_steps:
        stop_at = min(stop_at, step + a.max_steps)
    while step < stop_at:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        lr = lr_at(step, cfg["schedule"], opt_cfg["peak_lr"], total, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_sum = torch.zeros((), device=dev)
        t_data = 0.0
        data_ck = None
        for m in range(GA):
            td = time.time()
            xc = data.get(step * B + (m * world + rank) * mb, mb)
            if m == 0:
                data_ck = int(xc[:, :64].sum())  # data-position fingerprint of rank 0's first micro-batch
            x = xc.to(dev, non_blocking=True)
            t_data += time.time() - td
            ctx = ddp.no_sync() if m < GA - 1 else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16), attn_ctx():
                    loss = fwd(x)
                (loss / GA).backward()
            loss_sum += loss.detach()
            del loss
        gn = torch.nn.utils.clip_grad_norm_(ddp.parameters(), opt_cfg["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        stats = torch.stack([loss_sum / GA, torch.tensor(torch.cuda.max_memory_allocated() / 2**30, device=dev)])
        dist.all_reduce(stats[:1])
        mem = stats[1:].clone()
        dist.all_reduce(mem, op=dist.ReduceOp.MAX)
        torch.cuda.synchronize()
        dt = time.time() - t0
        t_train += dt
        step += 1
        if rank == 0:
            rec = {"step": step, "tokens": step * tok_per_step, "flops": step * tok_per_step * cfg["flops_per_token"],
                   "loss": stats[0].item() / world, "lr": lr, "grad_norm": gn.item(), "step_time": dt,
                   "data_time": t_data, "tokens_per_s": tok_per_step / dt, "peak_mem_gb": mem.item(),
                   "wall": time.time() - t_start, "data_ck": data_ck}
            tlog.write(json.dumps(rec) + "\n")
            tlog.flush()  # every step: a killed job must not lose log lines
            if wb:
                wb.log({f"train/{k}": v for k, v in rec.items()}, step=step)
        if step in eval_steps:
            do_eval()
        if rank == 0 and step in ck_steps:
            t0 = time.time()
            save_weights(model, a.out / "weights" / f"step{step:06d}_tok{step * tok_per_step}")
            event("save_weights", seconds=time.time() - t0, target=str(ck_steps[step]))
        if rank == 0 and resume_every and step % resume_every == 0 and step < stop_at:
            t0 = time.time()
            save_resume(model, opt, step, step * tok_per_step, a.out / "resume_latest", {"cfg": cfg})
            event("save_resume", seconds=time.time() - t0)
        dist.barrier() if (step in ck_steps or (resume_every and step % resume_every == 0)) else None
        if rank == 0 and decay_start and step == decay_start and not a.stop_before_decay:
            t0 = time.time()  # WSD pre-decay checkpoint with optimizer state, kept permanently
            save_resume(model, opt, step, step * tok_per_step, a.out / f"stable_step{step:06d}", {"cfg": cfg})
            event("save_stable", seconds=time.time() - t0)
        if a.max_minutes:  # collective decision: ranks' clocks differ, a one-sided break deadlocks NCCL
            flag = torch.tensor(float(t_train > a.max_minutes * 60), device=dev)
            dist.broadcast(flag, 0)
            if flag.item():
                break
    if rank == 0:
        tlog.flush()
        if a.stop_before_decay and step == decay_start:
            t0 = time.time()
            save_resume(model, opt, step, step * tok_per_step, a.out / f"stable_step{step:06d}", {"cfg": cfg})
            event("save_stable", seconds=time.time() - t0)
        if a.bench_save:
            t0 = time.time()
            save_weights(model, a.out / "weights" / f"bench_step{step:06d}")
            event("save_weights", seconds=time.time() - t0, target="bench")
            t0 = time.time()
            save_resume(model, opt, step, step * tok_per_step, a.out / "resume_latest", {"cfg": cfg})
            event("save_resume", seconds=time.time() - t0)
        event("end", train_seconds=t_train, wall_seconds=time.time() - t_start)
        if wb:
            wb.finish()
    dist.barrier()
    dist.destroy_process_group()


def kernel_report(model, arch) -> dict:
    """Which kernels actually run (asserted where a silent slow fallback exists)."""
    r = {"arch": arch, "model_class": f"{type(model).__module__}.{type(model).__name__}"}
    if arch == "mamba2":
        import fla.layers.mamba2 as m
        r["mamba_split_conv1d_scan_combined"] = m.mamba_split_conv1d_scan_combined is not None
        r["causal_conv1d_fn"] = getattr(m, "causal_conv1d_fn", None) is not None
        assert r["mamba_split_conv1d_scan_combined"] and r["causal_conv1d_fn"], r
    if arch == "xlstm":
        c = model.config
        r.update({k: getattr(c, k, None) for k in ("chunkwise_kernel", "sequence_kernel", "step_kernel", "mode", "chunk_size")})
        from transformers.utils import is_xlstm_available
        r["xlstm_package_used"] = is_xlstm_available()
        blk = next((m for m in model.modules() if hasattr(m, "config") and hasattr(m.config, "chunkwise_kernel") and m is not model), None)
        r["block_chunkwise_kernel"] = getattr(getattr(blk, "config", None), "chunkwise_kernel", None)
    if arch == "mzansilm":
        r["attn_implementation"] = model.config._attn_implementation
    if arch == "gdn":
        r.update({k: getattr(model.config, k) for k in ("fuse_norm", "fuse_swiglu", "fuse_cross_entropy", "attn_mode")})
    return r


def env_info() -> dict:
    import importlib.metadata as md
    import platform
    info = {"python": platform.python_version(), "host": platform.node(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "gpu": torch.cuda.get_device_name(0), "nccl": ".".join(map(str, torch.cuda.nccl.version()))}
    for pkg in ("transformers", "flash-linear-attention", "fla-core", "mamba-ssm", "causal-conv1d", "triton",
                "tilelang", "tokenizers", "numpy", "wandb", "xlstm", "mlstm_kernels", "flash-attn", "safetensors",
                "liger-kernel"):
        try:
            info[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            info[pkg] = None
    info["env"] = {k: v for k, v in os.environ.items() if k.startswith(("FLA_", "MAMBA_", "TRITON_", "PYTORCH_", "NCCL_", "CUDA_", "OMP_"))}
    return info


if __name__ == "__main__":
    main()
