"""xLSTM training kernel: TFLA (mlstm_kernels 2.0.2 chunkwise--triton_xl_chunk) at padded power-of-two head dims.
Used by train_fft.py for new xLSTM training units from 27 Sep 2026 (user decision; checks and the gate override in
the RUNBOOK "Speed settings", tfla_check.py / tfla_check.sbatch). self_check() runs before the swap.

At this model's head dims (qk 92, v 184; also 96/192) the TFLA Triton kernels intermittently return a wrong forward or
near-orthogonal gradients on identical inputs; at 64/128 and 128/256 every call agrees (tfla_check.py kernel). So the
training kernel zero-pads q/k to 128 and v to 256 and calls TFLA with the unpadded scale 1/sqrt(92) (the public
wrapper always uses 1/sqrt(padded dim)); h and the last states are sliced back. Zero qk columns add nothing to q.k or
to the normaliser n.q, zero v columns give zero h/C columns, the gates and the stabiliser m are per head, and the
gradients of the padded columns are dropped by the slice/pad autograd.

use_tfla() replaces only each transformers mLSTM backend's training function (`_train_fn`, mode "train"). The sealed
xLSTM runtime has no `xlstm` package, so the model code path, its config, the inference/step/sequence kernels and the
saved checkpoints are unchanged: generation and scoring (separate processes loading the checkpoint) stay native.
"""

from __future__ import annotations

import sys
from functools import partial

import torch
import torch.nn.functional as F

PYDEPS = "/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926/pydeps_tfla"  # mlstm_kernels 2.0.2 + einops 0.8.2
NAME = "tfla_padded128"  # train_kernel recorded per run


def _pow2(d: int) -> int:
    return 1 << (d - 1).bit_length()


def tfla_padded(q, k, v, i, f, c_initial=None, n_initial=None, m_initial=None, return_last_states=False, eps=1e-6,
                chunk_size=64, autocast_kernel_dtype=torch.bfloat16):
    """Same signature and result as mlstm_kernels' chunkwise kernels, (B, NH, S, D) inputs."""
    if PYDEPS not in sys.path:
        sys.path.insert(0, PYDEPS)
    from mlstm_kernels.torch.chunkwise.triton_xl_chunk.fwbw import _get_chunkwise_fwbw_kernel

    dq, dv = q.shape[-1], v.shape[-1]
    pq, pv = _pow2(dq) - dq, _pow2(dv) - dv
    q, k, v = F.pad(q, (0, pq)), F.pad(k, (0, pq)), F.pad(v, (0, pv))
    if c_initial is not None:
        c_initial, n_initial = F.pad(c_initial, (0, pv, 0, pq)), F.pad(n_initial, (0, pq))
    h, c, n, m = _get_chunkwise_fwbw_kernel(autocast_kernel_dtype).apply(
        q, k, v, i, f, c_initial, n_initial, m_initial,
        dq**-0.5,  # qk_scale of the unpadded head dim
        return_last_states, eps, chunk_size,
        None, None, None, None, None, None, None, None, None, None,  # kernel tiling: library heuristics
        True,  # recompute_states_in_bw (the library default)
    )
    h = h[..., :dv]
    return (h, (c[..., :dq, :dv], n[..., :dq], m)) if return_last_states else h


def _train_fn(kw, query, key, value, igate, fgate, c_initial=None, n_initial=None, m_initial=None, return_last_states=False):
    return tfla_padded(query, key, value, igate, fgate, c_initial, n_initial, m_initial, return_last_states, **kw)


def self_check(dq: int = 92, dv: int = 184, calls: int = 3) -> dict:
    """Padded TFLA vs mlstm_kernels' native_custbw in fp32 on random inputs (2 x 4 heads x 256, this model's head
    dims), `calls` repeated calls: forward rel. error <= 0.012 and every input-gradient cosine >= 0.999, else raises."""
    if PYDEPS not in sys.path:
        sys.path.insert(0, PYDEPS)
    from mlstm_kernels.torch import get_mlstm_kernel

    g = torch.Generator(device="cuda").manual_seed(0)
    base = [torch.randn(2, 4, 256, d, device="cuda", generator=g) * 0.5 for d in (dq, dq, dv)]
    base += [torch.randn(2, 4, 256, device="cuda", generator=g), torch.randn(2, 4, 256, device="cuda", generator=g) + 3]
    dh = torch.randn(2, 4, 256, dv, device="cuda", generator=g)

    def run(fn, dtype):
        ts = [t.detach().clone().to(dtype).requires_grad_() for t in base]
        h = fn(*ts, eps=1e-6, chunk_size=64, autocast_kernel_dtype=dtype)
        (h.float() * dh).sum().backward()
        return h.detach().float(), [t.grad.float().flatten() for t in ts]

    href, gref = run(get_mlstm_kernel("chunkwise--native_custbw"), torch.float32)
    res = []
    for _ in range(calls):
        h, gs = run(tfla_padded, torch.bfloat16)
        res.append({"h_rel": round(float((h - href).norm() / href.norm()), 5),
                    "grad_cos_min": round(min(float(F.cosine_similarity(a, b, dim=0)) for a, b in zip(gs, gref)), 5)})
    if not all(r["h_rel"] <= 0.012 and r["grad_cos_min"] >= 0.999 for r in res):
        raise RuntimeError(f"tfla_padded self-check failed: {res}")
    return {"calls": res, "gpu": torch.cuda.get_device_name(0)}


def use_tfla(model) -> int:
    """Swap every mLSTM backend's training kernel for padded TFLA (all or none); returns the number swapped."""
    from transformers.utils import is_xlstm_available

    assert not is_xlstm_available(), "the xlstm package would change the model code path"
    backends = [m for m in model.modules() if type(m).__name__ == "xLSTMBackend"]
    assert len(backends) == model.config.num_hidden_layers, (len(backends), model.config.num_hidden_layers)
    assert all(m.config.mode == "train" for m in backends)  # no padding wrapper: inputs are padded to chunk multiples
    for m in backends:
        c = m.config
        m._train_fn = partial(_train_fn, dict(eps=c.eps, chunk_size=c.chunk_size,
                                              autocast_kernel_dtype=getattr(torch, c.autocast_kernel_dtype)))
    return len(backends)
