"""No-future-leak check under the training/eval code path, plus eval-denominator identity check.

For each (arch, weights dir): build the model exactly as pretrain.py does (fast stack: fp32 master weights,
bf16 autocast, xLSTM TFLA kernel), load the probe's final weights, take a real 2048-token held-out sequence in the
training format (EOS-separated docs, no BOS), then randomise every position > t and recompute. Outputs at positions
<= t must not change. Checked on: (a) eval path model(...).logits, (b) training path = backbone hidden states that
feed the shared fused CE, eager and torch.compile'd (not xLSTM), (c) per-position fused-CE loss, and (d) a 2-row batch where only row 1 changes
(row 0 must not change: no cross-row leak). Usage: python causality.py OUT.json arch:weights_dir[:xlstm_kernel] ...
"""
import json, math, sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

import pretrain as P

out = Path(sys.argv[1])
res = {}
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True  # fast stack setting
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(P.os.environ["MP_TOKENIZER"])
df = pd.read_parquet(P.os.environ["MP_VAL_PARQUET"])
ids = []
for t in df["text"]:  # training format: doc tokens + EOS, no BOS, concatenated
    ids += tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(ids) >= 4096:
        break
x = torch.tensor(ids[:2048])[None]
x2 = torch.tensor(ids[2048:4096])[None]
g = torch.Generator().manual_seed(0)
CUTS = (1024, 1000, 1)  # positions > t randomised


def fwd(model, body, inp, soft):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = model(input_ids=inp, use_cache=False).logits.float()
        h = body(input_ids=inp, use_cache=False).last_hidden_state.float()
    W = model.get_output_embeddings().weight.float()
    zz = h @ W.t()
    if soft:
        zz = soft * torch.tanh(zz / soft)
    lp = F.cross_entropy(zz[0, :-1], inp[0, 1:], reduction="none")  # per-position loss of the training objective
    return z, h, lp


for spec in sys.argv[2:]:
    arch, wdir, *kern = spec.split(":")
    cfg = json.loads(Path(f"../configs/{arch}_wsd.json").read_text())
    if kern:
        cfg["xlstm_chunkwise_kernel"] = kern[0]
    torch.manual_seed(0)
    model = P.build_model(cfg).to(dev)
    sd = torch.load(Path(wdir) / "pytorch_model.bin", map_location=dev, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all("lm_head" in m for m in missing), (missing, unexpected)
    model.tie_weights()
    model.eval()
    body = getattr(model, "model", None) or model.backbone
    soft = getattr(model.config, "output_logit_soft_cap", None)
    r = {"weights": wdir, "kernels": P.kernel_report(model, arch)}
    base = fwd(model, body, x.to(dev), soft)
    for t in CUTS:
        y = x.clone()
        y[0, t + 1:] = torch.randint(4, 65536, (2047 - t,), generator=g)
        o = fwd(model, body, y.to(dev), soft)
        r[f"cut{t}"] = {
            "logits_max_abs_diff_prefix": (base[0][0, :t + 1] - o[0][0, :t + 1]).abs().max().item(),
            "hidden_max_abs_diff_prefix": (base[1][0, :t + 1] - o[1][0, :t + 1]).abs().max().item(),
            "loss_max_abs_diff_prefix": (base[2][:t] - o[2][:t]).abs().max().item(),  # predictions of tokens 1..t
            "logits_max_abs_diff_suffix(should be large)": (base[0][0, t + 1:] - o[0][0, t + 1:]).abs().max().item(),
            "logits_abs_scale": base[0][0, :t + 1].abs().mean().item(),
        }
    # cross-row: batch [x, x2] vs [x, rand]
    b1 = torch.cat([x, x2]).to(dev)
    b2 = torch.cat([x, torch.randint(4, 65536, (1, 2048), generator=g)]).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        za = model(input_ids=b1, use_cache=False).logits.float()
        zb = model(input_ids=b2, use_cache=False).logits.float()
    r["cross_row_max_abs_diff_row0"] = (za[0] - zb[0]).abs().max().item()
    if arch != "xlstm":  # compiled backbone, as in the fast training stack
        import torch._dynamo as dynamo
        dynamo.config.cache_size_limit = dynamo.config.recompile_limit = 64
        cbody = torch.compile(body)
        hb = fwd(model, cbody, x.to(dev), soft)[1]
        y = x.clone()
        y[0, 1025:] = torch.randint(4, 65536, (1023,), generator=g)
        hc = fwd(model, cbody, y.to(dev), soft)[1]
        r["compiled_hidden_max_abs_diff_prefix_cut1024"] = (hb[0, :1025] - hc[0, :1025]).abs().max().item()
    res[arch] = r
    print(arch, json.dumps(r, indent=1), flush=True)
    del model, body
    torch.cuda.empty_cache()
out.write_text(json.dumps(res, indent=2))
print("DONE")
