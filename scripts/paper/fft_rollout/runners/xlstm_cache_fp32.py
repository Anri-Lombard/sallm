"""Keep xLSTM generation-cache states in fp32 (27 Sep 2026).

transformers 4.57.3's xLSTMModel.forward allocates its xLSTMCache in the embedding dtype, ignoring
config.inference_state_dtype, so a bf16-loaded xLSTM decodes with bf16 recurrent states. This wraps
xLSTMCache.__init__ to allocate the states in fp32 whatever dtype is requested. The rollout loads xLSTM in fp32, so
its caches were already fp32 and this is a guard (no-op there); it logs the first allocation.
"""

import torch
from transformers.models.xlstm import modeling_xlstm as _mx

_init = _mx.xLSTMCache.__init__
_logged = False


def _init_fp32(self, config, max_batch_size, dtype=torch.bfloat16, device=None, **kwargs):
    global _logged
    if not _logged:
        print(f"XLSTM_CACHE_FP32_PATCH active: cache states float32 (requested {dtype})", flush=True)
        _logged = True
    _init(self, config, max_batch_size, dtype=torch.float32, device=device, **kwargs)


_mx.xLSTMCache.__init__ = _init_fp32
