"""NOT USED by the rollout (27 Sep 2026): the TFLA Triton kernels gave intermittently wrong results at this model's head
dims 92/184 (tfla_check.py kernel/parity; RUNBOOK "Speed settings"). Kept for those checks.

xLSTM training kernel swap: TFLA (mlstm_kernels chunkwise--triton_xl_chunk) in place of transformers' native kernel.

The sealed xLSTM runtime has no `xlstm` package, so transformers 4.57.3 builds its own mLSTM backend with the
native chunkwise kernel. use_tfla() replaces only each backend's training function (`_train_fn`, used in mode
"train") with the TFLA kernel that pretrained the matched base (mlstm_kernels 2.0.2, same chunk size, eps and
autocast dtype). The model config, the inference/step/sequence kernels and the saved checkpoints stay as they were,
so generation and scoring (a separate process loading the saved checkpoint) are unchanged.
"""

from __future__ import annotations

import sys
from functools import partial

import torch

PYDEPS = "/scratch/lmbanr001/masters/sallm/results/fft_rollout_20260926/pydeps_tfla"  # mlstm_kernels 2.0.2 + einops 0.8.2
KERNEL = "chunkwise--triton_xl_chunk"


def _train_fn(kernel, kw, query, key, value, igate, fgate, c_initial=None, n_initial=None, m_initial=None,
              return_last_states=False):
    return kernel(q=query, k=key, v=value, i=igate, f=fgate, c_initial=c_initial, n_initial=n_initial, m_initial=m_initial,
                  return_last_states=return_last_states, **kw)


def use_tfla(model) -> int:
    """Swap every mLSTM backend's training kernel for TFLA; returns the number of backends swapped."""
    if PYDEPS not in sys.path:
        sys.path.insert(0, PYDEPS)
    from mlstm_kernels.torch import get_mlstm_kernel
    from transformers.utils import is_xlstm_available

    assert not is_xlstm_available(), "the xlstm package would change the model code path"
    kernel = get_mlstm_kernel(KERNEL)
    n = 0
    for m in model.modules():
        if type(m).__name__ == "xLSTMBackend":
            c = m.config
            assert c.mode == "train", c.mode  # no padding wrapper: sequences are padded to multiples of chunk_size
            m._train_fn = partial(_train_fn, kernel, dict(eps=c.eps, chunk_size=c.chunk_size,
                                                          autocast_kernel_dtype=getattr(torch, c.autocast_kernel_dtype)))
            n += 1
    assert n == model.config.num_hidden_layers, (n, model.config.num_hidden_layers)
    return n
