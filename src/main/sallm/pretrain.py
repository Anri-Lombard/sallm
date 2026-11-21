"""Matched-budget pretraining for all five ~125M SALLM decoder architectures.

Launch::

    torchrun --nproc_per_node=2 -m sallm.pretrain \
        --config src/conf/pretrain/gdn_wsd.json \
        --data /path/to/train_bin_dir \
        --out /path/to/run_dir \
        --stack fast

Supported architectures: ``llama`` (alias ``mzansilm``), ``mamba2``, ``xlstm``,
``gated_deltanet`` (alias ``gdn``), and ``hybrid`` (3:1 GDN + softmax attention).
Every architecture shares the packed uint16 token stream, global batch, warmup,
WSD/cosine schedule, AdamW with ``>=2-D`` weight decay, fp32 master weights with
bf16 autocast, optional Liger fused linear cross-entropy, and held-out per-language
bits-per-byte scorer.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as importlib_metadata
import json
import logging
import math
import os
import platform
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from sallm.models.llama_compatibility import LlamaConfig, LlamaForCausalLM
from sallm.models.registry import (
    patch_mamba2_grouped_rmsnorm,
    tie_xlstm_embeddings,
    use_exact_xlstm_head_dims,
    validate_xlstm_pretrain_kernels,
)

logger = logging.getLogger(__name__)
LN2 = math.log(2.0)

_ARCH_ALIASES = {
    "mzansilm": "llama",
    "llama": "llama",
    "mamba2": "mamba2",
    "xlstm": "xlstm",
    "gdn": "gated_deltanet",
    "gated_deltanet": "gated_deltanet",
    "hybrid": "hybrid",
}


def canonical_arch(arch: str) -> str:
    key = arch.strip().lower()
    if key not in _ARCH_ALIASES:
        raise ValueError(
            f"Unsupported pretraining architecture {arch!r}; "
            f"expected one of {sorted(_ARCH_ALIASES)}."
        )
    return _ARCH_ALIASES[key]


def build_pretrain_model(
    cfg: dict[str, Any],
    *,
    allow_native_xlstm: bool = False,
) -> nn.Module:
    """Build one of the five matched architectures and assert its parameter count."""
    arch = canonical_arch(cfg["arch"])
    mc = dict(cfg["model_config"])
    model: nn.Module

    if arch == "llama":
        mc["attn_implementation"] = cfg.get("attn_implementation", "sdpa")
        model = LlamaForCausalLM(LlamaConfig(**mc))
    elif arch == "mamba2":
        try:
            fla_models = importlib.import_module("fla.models")
            model = fla_models.Mamba2ForCausalLM(fla_models.Mamba2Config(**mc))
        except ModuleNotFoundError:
            from transformers import Mamba2Config, Mamba2ForCausalLM

            fla_to_hf = {
                "norm_eps": "layer_norm_epsilon",
                "dt_min": "time_step_min",
                "dt_max": "time_step_max",
                "dt_init_floor": "time_step_floor",
            }
            drop_keys = {"fuse_cross_entropy", "rmsnorm"}
            hf_mc = {
                fla_to_hf.get(k, k): v for k, v in mc.items() if k not in drop_keys
            }
            if "num_heads" not in hf_mc:
                hf_mc["num_heads"] = (
                    int(hf_mc["expand"])
                    * int(hf_mc["hidden_size"])
                    // int(hf_mc["head_dim"])
                )
            model = Mamba2ForCausalLM(Mamba2Config(**hf_mc))
            patch_mamba2_grouped_rmsnorm(model)
    elif arch in ("gated_deltanet", "hybrid"):
        fla_models = importlib.import_module("fla.models")
        model = fla_models.GatedDeltaNetForCausalLM(
            fla_models.GatedDeltaNetConfig(**mc)
        )
    elif arch == "xlstm":
        transformers_mod: Any = cast(Any, importlib.import_module("transformers"))
        config_obj = transformers_mod.xLSTMConfig(**mc)
        # Set via object.__setattr__ because transformers 5's @strict xLSTMConfig
        # validator only enumerates native kernels when mlstm_kernels is absent.
        if cfg.get("xlstm_chunkwise_kernel"):
            object.__setattr__(
                config_obj, "chunkwise_kernel", cfg["xlstm_chunkwise_kernel"]
            )
        if cfg.get("xlstm_sequence_kernel"):
            object.__setattr__(
                config_obj, "sequence_kernel", cfg["xlstm_sequence_kernel"]
            )
        if cfg.get("xlstm_step_kernel"):
            object.__setattr__(config_obj, "step_kernel", cfg["xlstm_step_kernel"])
        use_exact_xlstm_head_dims(config_obj)
        validate_xlstm_pretrain_kernels(
            config_obj,
            allow_native=allow_native_xlstm,
        )
        model = transformers_mod.xLSTMForCausalLM(config_obj)
        tie_xlstm_embeddings(model)
    else:
        raise ValueError(f"Unsupported architecture: {arch!r}")

    expected = cfg.get("expected_params")
    if expected is not None:
        n_params = sum(p.numel() for p in model.parameters())
        if n_params != int(expected):
            raise ValueError(
                f"Parameter count mismatch for {arch}: got {n_params}, "
                f"expected {expected}."
            )
    return model


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Decay >=2-D tensors only; exclude 1-D norms, biases, A_log, D, and dt_bias."""
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def flops_per_token(model: nn.Module, cfg: dict[str, Any]) -> dict[str, int]:
    """Analytical model FLOPs per training token (3x forward dense + mixer)."""
    lin = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))
    conv = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Conv1d))
    c = cfg["model_config"]
    seq_len = int(cfg["seq_len"])
    arch = canonical_arch(cfg["arch"])

    if arch == "llama":
        head_dim = int(
            c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
        )
        d_attn = int(c["num_attention_heads"]) * head_dim
        mix = 2 * int(c["num_hidden_layers"]) * seq_len * d_attn
    elif arch == "mamba2":
        chunk = int(c["chunk_size"])
        state = int(c["state_size"])
        head_dim = int(c["head_dim"])
        heads = int(c["expand"]) * int(c["hidden_size"]) // head_dim
        groups = int(c["n_groups"])
        mix = int(c["num_hidden_layers"]) * (
            groups * chunk * state
            + heads * chunk * head_dim
            + 4 * heads * state * head_dim
        )
    elif arch in ("gated_deltanet", "hybrid"):
        chunk = 64
        heads = int(c["num_heads"])
        dk = int(c["head_dim"])
        dv = dk * int(c["expand_v"])
        attn_cfg = c.get("attn")
        n_attn = len(attn_cfg["layers"]) if attn_cfg else 0
        mix = (
            (int(c["num_hidden_layers"]) - n_attn)
            * heads
            * (3 * chunk * dk + 2 * chunk * dv + 6 * dk * dv)
        )
        if n_attn:
            mix += 2 * n_attn * seq_len * int(c["hidden_size"])
    elif arch == "xlstm":
        chunk = int(c["chunk_size"])
        heads = int(c["num_heads"])
        hidden = int(c["hidden_size"])
        dqk = int(hidden * float(c["qk_dim_factor"])) // heads
        dv = int(hidden * float(c["v_dim_factor"])) // heads
        n_blocks = int(c.get("num_blocks", c["num_hidden_layers"]))
        mix = n_blocks * heads * (chunk * dqk + chunk * dv + 4 * dqk * dv)
    else:
        mix = 0

    fwd_dense = 2 * lin + 2 * conv
    return {
        "linear_params": lin,
        "conv_params": conv,
        "fwd_dense": fwd_dense,
        "fwd_mix": mix,
        "train_dense": 3 * fwd_dense,
        "train_mix": 3 * mix,
        "train_total": 3 * (fwd_dense + mix),
    }


class LMLoss(nn.Module):
    """Next-token cross-entropy loss, optionally via Liger fused linear CE."""

    def __init__(self, model: nn.Module, *, fused: bool, compile_body: bool = False):
        super().__init__()
        self.model = model
        self.fused = fused
        body: Any = getattr(model, "model", None) or cast(Any, model).backbone
        # Compile only the backbone: Liger's autograd.Function does not trace cleanly.
        self.body: Any = torch.compile(body) if compile_body else body
        self.flce: Any = None
        if fused:
            liger_mod = importlib.import_module("liger_kernel.transformers")

            cap = getattr(getattr(model, "config", None), "output_logit_soft_cap", None)
            self.flce = liger_mod.LigerFusedLinearCrossEntropyLoss(
                softcap=cap, accum_dtype=torch.float32
            )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.fused:
            logits = self.model(input_ids=input_ids, use_cache=False).logits
            return F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
            )
        hidden = self.body(input_ids=input_ids, use_cache=False).last_hidden_state
        head_weight = cast(Any, self.model).get_output_embeddings().weight
        return self.flce(
            head_weight,
            hidden[:, :-1].reshape(-1, hidden.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )


def lr_at(
    step: int,
    schedule: dict[str, Any],
    peak_lr: float,
    total_steps: int,
    warmup_steps: int,
) -> float:
    """Compute the learning rate at ``step`` for cosine or WSD (1 - sqrt) schedules."""
    floor = float(schedule["min_lr_ratio"]) * peak_lr
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    sched_type = schedule["type"]
    if sched_type == "cosine":
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return floor + (peak_lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    if sched_type == "wsd":
        decay_steps = round(float(schedule["decay_frac"]) * total_steps)
        decay_start = total_steps - decay_steps
        if step < decay_start:
            return peak_lr
        progress = min(1.0, (step - decay_start) / max(1, decay_steps))
        return floor + (peak_lr - floor) * (1.0 - math.sqrt(progress))
    raise ValueError(f"Unsupported schedule type: {sched_type!r}")


class PackedTokenBlocks:
    """Deterministic memmap reader over packed uint16 token blocks."""

    def __init__(self, data_dir: Path, seq_len: int, seed: int):
        meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
        self.tokens = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
        if len(self.tokens) != int(meta["n_tokens"]):
            raise ValueError(
                f"train.bin length {len(self.tokens)} != "
                f"meta n_tokens {meta['n_tokens']}"
            )
        self.seq_len = seq_len
        self.seed = seed
        self.meta = meta
        self.n_blocks = len(self.tokens) // seq_len
        self._epoch_perms: dict[int, np.ndarray] = {}

    def physical_index(self, global_block: int) -> int:
        epoch, offset = divmod(global_block, self.n_blocks)
        if epoch == 0:
            return offset
        if epoch not in self._epoch_perms:
            self._epoch_perms[epoch] = np.random.default_rng(
                self.seed + epoch
            ).permutation(self.n_blocks)
        return int(self._epoch_perms[epoch][offset])

    def get(self, first_block: int, count: int) -> torch.Tensor:
        rows = [
            self.tokens[
                self.physical_index(idx) * self.seq_len : (self.physical_index(idx) + 1)
                * self.seq_len
            ]
            for idx in range(first_block, first_block + count)
        ]
        return torch.from_numpy(np.stack(rows).astype(np.int64))


class HeldOutBPBScorer:
    """Whole-document held-out bits-per-byte scorer (pooled and per language)."""

    WINDOW = 1024
    PAD_MULT = 64
    TOKEN_BUDGET = 8192

    def __init__(
        self,
        parquet_path: str,
        tokenizer_path: str,
        limit_per_lang: int | None = None,
        doc_start: str = "bos",
    ):
        import pandas as pd
        from tokenizers.decoders import ByteLevel
        from transformers import AutoTokenizer

        df = pd.read_parquet(parquet_path)
        if limit_per_lang:
            df = df.groupby("lang").head(limit_per_lang).reset_index(drop=True)
        tok = cast(Any, AutoTokenizer.from_pretrained(tokenizer_path))
        if getattr(tok, "backend_tokenizer", None) is not None:
            tok.backend_tokenizer.decoder = ByteLevel()
        encoded = tok(df["text"].tolist(), add_special_tokens=True)["input_ids"]
        if doc_start == "eos":
            encoded = [[tok.eos_token_id] + ids[1:] for ids in encoded]

        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")
        self.pad_id = int(pad_id)
        self.langs = sorted(df["lang"].unique())
        self.doc_lang = np.array([self.langs.index(lang) for lang in df["lang"]])
        self.n_text = np.array([len(ids) - 2 for ids in encoded], dtype=np.float64)
        self.bytes = np.array(
            [
                len(
                    str(
                        tok.decode(
                            ids[1:-1],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    ).encode()
                )
                for ids in encoded
            ],
            dtype=np.float64,
        )

        items: list[tuple[int, int, list[int], int]] = []
        for doc_idx, ids in enumerate(encoded):
            start = 0
            while True:
                items.append(
                    (doc_idx, start, ids[start : start + self.WINDOW], len(ids))
                )
                if start + self.WINDOW >= len(ids):
                    break
                start += self.WINDOW - 1
        items.sort(key=lambda item: len(item[2]))

        self.batches: list[list[tuple[int, int, list[int], int]]] = []
        idx = 0
        while idx < len(items):
            end = idx
            while end < len(items) and (
                (end - idx + 1) * self._round_up(len(items[end][2]))
                <= self.TOKEN_BUDGET
                or end == idx
            ):
                end += 1
            self.batches.append(items[idx:end])
            idx = end
        self.n_docs = len(encoded)
        self.n_windows = len(items)

    def _round_up(self, length: int) -> int:
        return -(-length // self.PAD_MULT) * self.PAD_MULT

    @torch.no_grad()
    def run(
        self, model: nn.Module, rank: int, world: int, device: torch.device
    ) -> dict[str, float]:
        was_training = model.training
        model.eval()
        text_nll = torch.zeros(self.n_docs, dtype=torch.float64, device=device)
        preds = torch.zeros(self.n_docs, dtype=torch.float64, device=device)

        for batch_idx in range(rank, len(self.batches), world):
            batch = self.batches[batch_idx]
            padded_len = self._round_up(max(len(win) for _, _, win, _ in batch))
            x = torch.full((len(batch), padded_len), self.pad_id, dtype=torch.long)
            for row, (_, _, win, _) in enumerate(batch):
                x[row, : len(win)] = torch.tensor(win)
            x = x.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(input_ids=x, use_cache=False).logits
            token_ce = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.size(-1)),
                x[:, 1:].reshape(-1),
                reduction="none",
            ).view(len(batch), padded_len - 1)
            for row, (doc_idx, start, win, n_doc) in enumerate(batch):
                n_pred = len(win) - 1
                ends_doc = start + len(win) == n_doc
                text_nll[doc_idx] += (
                    token_ce[row, : n_pred - 1 if ends_doc else n_pred].double().sum()
                )
                preds[doc_idx] += n_pred
            del logits, token_ce

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(text_nll)
            dist.all_reduce(preds)
        if not torch.equal(preds.cpu(), torch.tensor(self.n_text + 1)):
            raise RuntimeError("Held-out scorer token count invariant violated.")

        text_np = text_nll.cpu().numpy()
        out: dict[str, float] = {}
        for lang_idx, lang in enumerate([*self.langs, "ALL"]):
            mask = (
                np.ones(self.n_docs, bool)
                if lang == "ALL"
                else self.doc_lang == lang_idx
            )
            ce = float(text_np[mask].sum() / self.n_text[mask].sum())
            bpb = float(text_np[mask].sum() / LN2 / self.bytes[mask].sum())
            out[f"{lang}/ce"] = ce
            out[f"{lang}/bpb"] = bpb
        out["MACRO/bpb"] = float(np.mean([out[f"{lang}/bpb"] for lang in self.langs]))
        out["MACRO/ce"] = float(np.mean([out[f"{lang}/ce"] for lang in self.langs]))
        model.train(was_training)
        return out


def save_weights(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp, safe_serialization=False)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


def save_resume(
    model: Any,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens: int,
    path: Path,
    extra: dict[str, Any],
) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "step": step,
        "tokens": tokens,
        "torch_rng": torch.get_rng_state(),
        **extra,
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state()
    torch.save(state, tmp / "state.pt")
    model.config.save_pretrained(tmp)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


def kernel_report(model: nn.Module, arch: str) -> dict[str, Any]:
    """Report resolved kernels and assert against silent slow fallbacks."""
    canon = canonical_arch(arch)
    report: dict[str, Any] = {
        "arch": canon,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
    }
    if canon == "mamba2" and type(model).__module__.startswith("fla."):
        fla_mamba2 = importlib.import_module("fla.layers.mamba2")

        report["mamba_split_conv1d_scan_combined"] = (
            fla_mamba2.mamba_split_conv1d_scan_combined is not None
        )
        report["causal_conv1d_fn"] = (
            getattr(fla_mamba2, "causal_conv1d_fn", None) is not None
        )
        if not (
            report["mamba_split_conv1d_scan_combined"] and report["causal_conv1d_fn"]
        ):
            raise RuntimeError(f"Mamba-2 CUDA kernels unavailable: {report}")
    elif canon == "xlstm":
        cfg = cast(Any, model).config
        report.update(
            {
                key: getattr(cfg, key, None)
                for key in (
                    "chunkwise_kernel",
                    "sequence_kernel",
                    "step_kernel",
                    "mode",
                    "chunk_size",
                )
            }
        )
    elif canon == "llama":
        report["attn_implementation"] = getattr(
            cast(Any, model).config, "_attn_implementation", None
        )
    elif canon in ("gated_deltanet", "hybrid"):
        cfg = cast(Any, model).config
        report.update(
            {
                key: getattr(cfg, key, None)
                for key in (
                    "fuse_norm",
                    "fuse_swiglu",
                    "fuse_cross_entropy",
                    "attn_mode",
                    "attn",
                )
            }
        )
    return report


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "host": platform.node(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
        info["tf32_cudnn"] = torch.backends.cudnn.allow_tf32
    for pkg in (
        "transformers",
        "flash-linear-attention",
        "mamba-ssm",
        "causal-conv1d",
        "triton",
        "tokenizers",
        "numpy",
        "wandb",
        "xlstm",
        "mlstm_kernels",
        "flash-attn",
        "safetensors",
        "liger-kernel",
    ):
        try:
            info[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            info[pkg] = None
    return info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Matched-budget DDP pretraining for SALLM decoder architectures."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--budget-epochs", type=float, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--stop-before-decay", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-at", type=str, default=None)
    parser.add_argument("--eval-limit-per-lang", type=int, default=None)
    parser.add_argument("--stack", choices=["baseline", "fast"], default="fast")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--xlstm-kernel", default=None)
    parser.add_argument("--allow-native-xlstm", action="store_true")
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--total-tokens", type=float, default=None)
    parser.add_argument("--warmup-tokens", type=float, default=None)
    parser.add_argument("--peak-lr", type=float, default=None)
    parser.add_argument("--resume-every", type=int, default=None)
    parser.add_argument("--no-step0-eval", action="store_true")
    parser.add_argument("--eval-doc-start", choices=["bos", "eos"], default=None)
    parser.add_argument("--weights-final-only", action="store_true")
    args = parser.parse_args(argv)

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.peak_lr is not None:
        cfg["optim"]["peak_lr"] = args.peak_lr
    if args.eval_doc_start is not None:
        cfg["eval_doc_start"] = args.eval_doc_start
    if args.resume_every is not None:
        cfg["resume_every_steps"] = args.resume_every
    if args.xlstm_kernel:
        cfg["xlstm_chunkwise_kernel"] = args.xlstm_kernel
    cfg["eval_parquet"] = os.path.expandvars(cfg["eval_parquet"])
    cfg["tokenizer"] = os.path.expandvars(cfg["tokenizer"])

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.out.mkdir(parents=True, exist_ok=True)

    fast = args.stack == "fast"
    torch.backends.cuda.matmul.allow_tf32 = fast
    torch.backends.cudnn.allow_tf32 = fast

    global_batch = int(cfg["global_batch_seqs"])
    seq_len = int(cfg["seq_len"])
    micro_batch = int(args.micro_batch or cfg["micro_batch"])
    if global_batch % (micro_batch * world) != 0:
        raise ValueError(
            f"global_batch_seqs={global_batch} must be divisible by "
            f"micro_batch*world ({micro_batch}*{world})."
        )
    grad_accum = global_batch // (micro_batch * world)
    data = PackedTokenBlocks(args.data, seq_len, int(cfg["seed"]))
    epochs = float(
        args.budget_epochs if args.budget_epochs is not None else cfg["budget_epochs"]
    )
    tok_per_step = global_batch * seq_len
    total_steps = (
        math.ceil(args.total_tokens / tok_per_step)
        if args.total_tokens
        else int(data.n_blocks * epochs) // global_batch
    )
    warmup_steps = (
        round(args.warmup_tokens / tok_per_step)
        if args.warmup_tokens
        else round(
            float(cfg["schedule"]["warmup_frac"]) * (data.n_blocks // global_batch)
        )
    )
    decay_start = (
        total_steps - round(float(cfg["schedule"].get("decay_frac", 0)) * total_steps)
        if cfg["schedule"]["type"] == "wsd"
        else None
    )

    torch.manual_seed(int(cfg["seed"]))
    model = build_pretrain_model(cfg, allow_native_xlstm=args.allow_native_xlstm).to(
        device
    )
    opt_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        param_groups(model, float(opt_cfg["weight_decay"])),
        lr=float(opt_cfg["peak_lr"]),
        betas=(float(opt_cfg["betas"][0]), float(opt_cfg["betas"][1])),
        eps=float(opt_cfg["eps"]),
        fused=fast,
    )

    step = 0
    if args.resume:
        state = torch.load(
            args.resume / "state.pt", map_location=device, weights_only=True
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optim"])
        step = int(state["step"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if "cuda_rng" in state:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())

    flops = flops_per_token(model, cfg)
    cfg["flops_per_token"] = flops["train_total"]
    ddp = torch.nn.parallel.DistributedDataParallel(
        LMLoss(model, fused=fast, compile_body=fast and not args.no_compile),
        device_ids=[local_rank],
        gradient_as_bucket_view=True,
    )
    if fast and not args.no_compile:
        import torch._dynamo as dynamo

        dynamo_cfg = cast(Any, dynamo.config)
        dynamo_cfg.cache_size_limit = 64
        dynamo_cfg.recompile_limit = 64
    kernels = kernel_report(model, cfg["arch"])

    pts = list(
        range(
            int(cfg["eval_every_tokens"]), int(1e9) + 1, int(cfg["eval_every_tokens"])
        )
    )
    after_1b = int(cfg.get("eval_every_tokens_after_1b", cfg["eval_every_tokens"]))
    tok_cursor = int(1e9) + after_1b
    while tok_cursor <= total_steps * tok_per_step:
        pts.append(tok_cursor)
        tok_cursor += after_1b
    auto_eval = {
        math.ceil(t / tok_per_step)
        for t in pts
        if math.ceil(t / tok_per_step) <= total_steps
    } | {total_steps}
    eval_steps = (
        {int(x) for x in args.eval_at.split(",")} if args.eval_at else auto_eval
    )
    ck_steps: dict[int, int | str] = (
        {}
        if args.weights_final_only
        else {
            math.ceil(int(t) / tok_per_step): int(t)
            for t in cfg.get("weights_at_tokens", [])
            if math.ceil(int(t) / tok_per_step) <= total_steps
        }
    )
    ck_steps[total_steps] = "final"
    resume_every = int(cfg.get("resume_every_steps", 0))
    heldout = HeldOutBPBScorer(
        cfg["eval_parquet"],
        cfg["tokenizer"],
        args.eval_limit_per_lang,
        cfg.get("eval_doc_start", "bos"),
    )

    run_meta = {
        "cfg": cfg,
        "world": world,
        "grad_accum": grad_accum,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "decay_start": decay_start,
        "tokens_per_step": tok_per_step,
        "env": env_info(),
        "flops": flops,
        "kernels": kernels,
    }
    wb: Any = None
    if rank == 0:
        meta_name = (
            "run_meta.json" if step == 0 else f"run_meta_resumed_step{step}.json"
        )
        (args.out / meta_name).write_text(
            json.dumps(run_meta, indent=2, default=str), encoding="utf-8"
        )
        if os.environ.get("WANDB_MODE") != "disabled":
            import wandb

            wb = wandb.init(
                project=cfg.get("wandb_project", "sallm-matched-pretrain"),
                name=args.out.name,
                dir=str(args.out),
                config=run_meta,
                resume="allow",
                id=args.out.name.replace("/", "-")[-64:],
            )

    tlog = (
        (args.out / "train_log.jsonl").open("a", encoding="utf-8")
        if rank == 0
        else None
    )
    elog = (
        (args.out / "event_log.jsonl").open("a", encoding="utf-8")
        if rank == 0
        else None
    )

    def log_event(kind: str, **kw: Any) -> None:
        if rank == 0 and elog is not None:
            rec = {
                "kind": kind,
                "step": step,
                "tokens": step * tok_per_step,
                "flops": step * tok_per_step * cfg["flops_per_token"],
                **kw,
            }
            elog.write(json.dumps(rec) + "\n")
            elog.flush()
            if wb is not None:
                wb.log(
                    {
                        f"{kind}/{k}": v
                        for k, v in kw.items()
                        if isinstance(v, (int, float))
                    }
                    | {"tokens": rec["tokens"], "flops": rec["flops"]},
                    step=step,
                )

    def run_eval() -> None:
        torch.cuda.synchronize()
        t0 = time.time()
        res = heldout.run(model, rank, world, device)
        torch.cuda.synchronize()
        log_event("eval", seconds=time.time() - t0, **res)

    if step == 0 and not args.eval_at and not args.no_step0_eval:
        run_eval()

    ddp.train()
    stop_at = decay_start if (args.stop_before_decay and decay_start) else total_steps
    if args.max_steps:
        stop_at = min(stop_at, step + args.max_steps)

    t_train = 0.0
    t_start = time.time()
    while step < stop_at:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        lr = lr_at(
            step, cfg["schedule"], float(opt_cfg["peak_lr"]), total_steps, warmup_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss_sum = torch.zeros((), device=device)
        for micro_idx in range(grad_accum):
            xc = data.get(
                step * global_batch + (micro_idx * world + rank) * micro_batch,
                micro_batch,
            )
            x = xc.to(device, non_blocking=True)
            sync_ctx = ddp.no_sync() if micro_idx < grad_accum - 1 else nullcontext()
            with sync_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ddp(x)
                (loss / grad_accum).backward()
            loss_sum += loss.detach()
            del loss

        grad_norm = torch.nn.utils.clip_grad_norm_(
            ddp.parameters(), float(opt_cfg["grad_clip"])
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        dist.all_reduce(loss_sum)
        torch.cuda.synchronize()
        dt = time.time() - t0
        t_train += dt
        step += 1

        if rank == 0 and tlog is not None:
            rec = {
                "step": step,
                "tokens": step * tok_per_step,
                "loss": (loss_sum / (grad_accum * world)).item(),
                "lr": lr,
                "grad_norm": float(grad_norm.item()),
                "step_time": dt,
                "tokens_per_s": tok_per_step / dt,
            }
            tlog.write(json.dumps(rec) + "\n")
            tlog.flush()
            if wb is not None:
                wb.log({f"train/{k}": v for k, v in rec.items()}, step=step)

        if step in eval_steps:
            run_eval()
        if rank == 0 and step in ck_steps:
            save_weights(
                model,
                args.out / "weights" / f"step{step:06d}_tok{step * tok_per_step}",
            )
        if rank == 0 and resume_every and step % resume_every == 0 and step < stop_at:
            save_resume(
                model,
                optimizer,
                step,
                step * tok_per_step,
                args.out / "resume_latest",
                {"cfg": cfg},
            )
        if rank == 0 and decay_start and step == decay_start:
            save_resume(
                model,
                optimizer,
                step,
                step * tok_per_step,
                args.out / f"stable_step{step:06d}",
                {"cfg": cfg},
            )
        if step in ck_steps or (resume_every and step % resume_every == 0):
            dist.barrier()

    if rank == 0:
        if tlog is not None:
            tlog.close()
        log_event("end", train_seconds=t_train, wall_seconds=time.time() - t_start)
        if elog is not None:
            elog.close()
        if wb is not None:
            wb.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
