from collections import defaultdict
import glob
import os
import sys
import wandb
import itertools

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import uuid
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

torch.empty(
    1, device="cuda", requires_grad=True
).backward()  # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist

# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention

torch._inductor.config.coordinate_descent_tuning = (
    True  # turn this off for a faster compile time (but slightly slower run)
)

# -----------------------------------------------------------------------------
# Custom operators : FP8 matmul for lm_head by @YouJiacheng


@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor) -> Tensor:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        out = torch.matmul(x.to(torch.bfloat16), w.t().to(torch.bfloat16))
        return out

    return impl(x, w)


@mm_op.register_fake
def _(x: Tensor, w: Tensor):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x.to(torch.bfloat16) @ w.t().to(torch.bfloat16)


@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x: Tensor, w: Tensor):
        assert grad.is_contiguous()
        # Perform the backward pass in bfloat16
        grad_x = torch.matmul(grad.to(torch.bfloat16), w.to(torch.bfloat16))
        grad_w = torch.matmul(grad.t().to(torch.bfloat16), x.to(torch.bfloat16)).t()
        return grad_x, grad_w

    return impl(g, x, w)


@mm_backward_op.register_fake
def _(g: Tensor, x: Tensor, w: Tensor):
    return x.to(torch.bfloat16), w.to(torch.bfloat16)


def backward(ctx, grad_out: Tensor, *_):
    x, w = ctx.saved_tensors
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(grad_out, x, w)
    return grad_x, grad_w


def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    x, w, *_ = inputs  # Remove scaling factors from inputs
    ctx.save_for_backward(x, w)
    ctx.set_materialize_grads(False)


mm_op.register_autograd(backward, setup_context=setup_context)


def lm_head_fp8(x: Tensor, w: Tensor) -> Tensor:
    _x = x.flatten(0, -2)
    out: Tensor = torch.nn.functional.linear(
        _x.to(torch.bfloat16), w.to(torch.bfloat16)
    )
    return out.reshape(*x.shape[:-1], -1)


# -----------------------------------------------------------------------------
# Muon optimizer


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven"t tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        rank=0,
        world_size=1,
    ):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        assert all(isinstance(p, Tensor) for p in params)
        sizes = {p.numel() for p in params}

        def create_update_buffer(size: int):
            b = torch.empty(self.world_size, size, dtype=torch.bfloat16, device="cuda")
            return dict(
                update_buffer=b,
                update_buffer_views=[b[i] for i in range(self.world_size)],
            )

        param_groups = [
            dict(
                params=[p for p in params if p.numel() == size],
                **create_update_buffer(size),
            )
            for size in sizes
        ]
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            update_buffer = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None

            def update_prev():  # optimized Muon implementation contributed by @YouJiacheng
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5,
                    )

            for base_i in range(len(params))[:: self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - momentum)
                    g = g.lerp_(buf, momentum) if nesterov else buf
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps).flatten()
                else:
                    g = update_buffer_views[self.rank]
                update_prev()  # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self) -> None:
        std = 0.5 * (
            self.in_features**-0.5
        )  # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3**0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len=65536):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(
            0, 1, steps=dim // 4, dtype=torch.float32
        )
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = (
            self.cos[None, : x_BTHD.size(-3), None, :],
            self.sin[None, : x_BTHD.size(-3), None, :],
        )
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        std = 0.5 * (dim**-0.5)
        bound = (3**0.5) * std  # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, dim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(dim // num_heads)  # dim // num_heads = head_dim
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.detach().zero_()  # zero init suggested by @Grad62304977
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1)  # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = (
            F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x))
            .view(B, T, 3 * self.num_heads, -1)
            .chunk(3, dim=-2)
        )
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(
                v
            )  # @KoszarskyB & @Grad62304977
        else:  # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
        q, k = norm(q), norm(k)  # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
            scale=self.attn_scale,
        )
        y = (
            y.transpose(1, 2).contiguous().view_as(x)
        )  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.detach().zero_()  # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(
            x
        ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = (
            CausalSelfAttention(model_dim, num_heads, layer_idx)
            if layer_idx != 7
            else None
        )
        self.mlp = MLP(model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x, ve, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x


class ValueEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        num_unique_embeddings = min(
            3, (num_layers + 2) // 3
        )  # Calculate how many unique embeddings we need
        self.embed = nn.ModuleList(
            [
                nn.Embedding(num_embeddings, embedding_dim)
                for _ in range(num_unique_embeddings)
            ]
        )

    def forward(self, input_seq) -> list[Tensor | None]:
        ve = [emb(input_seq) for emb in self.embed]
        if self.num_layers <= 3:
            return ve[: self.num_layers]

        # For larger models, create a pattern that repeats every 3 layers
        pattern = []
        for i in range(self.num_layers):
            if i < len(ve):
                pattern.append(ve[i])
            else:
                idx = i % len(ve)
                pattern.append(ve[idx] if idx < len(ve) else None)
        return pattern


# -----------------------------------------------------------------------------
# The main model


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class GPT(nn.Module):
    def __init__(
        self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        self.value_embeds = ValueEmbedding(vocab_size, model_dim, num_layers)
        self.blocks = nn.ModuleList(
            [Block(model_dim, num_heads, layer_idx) for layer_idx in range(num_layers)]
        )
        # U-net design by @brendanh0gan
        self.num_encoder_layers = num_layers // 2  # Half of the layers for encoder
        self.num_decoder_layers = (
            num_layers - self.num_encoder_layers
        )  # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128))
        self.lm_head.weight.detach().zero_()  # @Grad62304977

    def forward(
        self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor
    ):
        BLOCK_SIZE = 128
        assert input_seq.ndim == 1
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        docs = (input_seq == 50256).cumsum(0)
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_mask: Tensor):
            num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
            indices = (
                dense_mask.argsort(dim=-1, descending=False, stable=True)
                .flip(-1)
                .to(torch.int32)
            )
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        def create_doc_swc_block_masks(sliding_window_num_blocks: Tensor):
            kv_idx = block_idx = torch.arange(
                NUM_BLOCKS, dtype=torch.int32, device="cuda"
            )
            q_idx = block_idx[:, None]
            causal_bm = q_idx >= kv_idx
            causal_full_bm = q_idx > kv_idx
            document_bm = (docs_low[:, None] <= docs_high) & (
                docs_low <= docs_high[:, None]
            )
            document_full_bm = (docs_low[:, None] == docs_high) & (
                docs_low == docs_high[:, None]
            )
            nonzero_bm = causal_bm & document_bm
            full_bm = causal_full_bm & document_full_bm
            kv_num_blocks, kv_indices = dense_to_ordered(nonzero_bm & ~full_bm)
            full_kv_num_blocks, full_kv_indices = dense_to_ordered(full_bm)

            def build_bm(sw_num_blocks: Tensor) -> BlockMask:
                return BlockMask.from_kv_blocks(
                    torch.clamp_max(
                        kv_num_blocks,
                        torch.clamp_min(sw_num_blocks - full_kv_num_blocks, 1),
                    ),
                    kv_indices,
                    torch.clamp_max(full_kv_num_blocks, sw_num_blocks - 1),
                    full_kv_indices,
                    BLOCK_SIZE=BLOCK_SIZE,
                    mask_mod=document_causal,
                )

            return build_bm(sliding_window_num_blocks), build_bm(
                sliding_window_num_blocks // 2
            )

        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        long_bm, short_bm = create_doc_swc_block_masks(sliding_window_num_blocks)

        x = x0 = norm(self.embed(input_seq)[None])  # use of norm here by @Grad62304977
        ve = self.value_embeds(input_seq)
        assert len(ve) == len(self.blocks)
        ve_enc, ve_dec = ve[: self.num_encoder_layers], ve[self.num_encoder_layers :]
        assert (
            len(ve_enc) == self.num_encoder_layers
            and len(ve_dec) == self.num_decoder_layers
        )

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm]
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0, block_masks[i])
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        block_masks.reverse()
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.blocks[self.num_encoder_layers + i](
                x, ve_dec[i], x0, block_masks[i]
            )
        x = norm(x)
        logits = self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits.float() / 7.5)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq)

        predictions = logits.view(-1, logits.size(-1)).argmax(dim=-1)
        correct_predictions = (predictions == target_seq).float().mean()

        return loss, correct_predictions


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader


def _load_data_shard(file: Path):
    header = torch.from_file(
        f"{file}", False, 256, dtype=torch.int32
    )  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(
            num_tokens, dtype=torch.uint16, pin_memory=True
        )  # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())  # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def distributed_data_generator(
    filename_pattern: str, batch_size: int, rank: int, world_size: int
):
    # Convert to Path object and handle absolute paths
    pattern_path = Path(filename_pattern)
    if pattern_path.is_absolute():
        # If absolute path, use parent directory and filename pattern separately
        files = sorted(pattern_path.parent.glob(pattern_path.name))
    else:
        # If relative path, use as is
        files = sorted(Path.cwd().glob(filename_pattern))

    assert len(files) > 0, f"No files found matching pattern: {filename_pattern}"
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size

    # Use itertools.cycle to loop indefinitely over the files
    file_cycle = itertools.cycle(files)
    tokens, pos = _load_data_shard(next(file_cycle)), 0  # Load the first file

    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = (
                _load_data_shard(next(file_cycle)),
                0,
            )  # Load the next file in the cycle
        buf = tokens[pos + rank * local_batch_size :][: local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs, targets


# -----------------------------------------------------------------------------
# int main


@dataclass
class Hyperparameters:
    # data
    train_files = "/scratch/lmbanr001/masters/dataset/filtered_data_binary_custom_tokenizer/custom_train_*.bin"
    val_files = "/scratch/lmbanr001/masters/dataset/filtered_data_binary_custom_tokenizer/custom_val_*.bin"
    batch_size = 64 * 1024  # batch size in tokens
    epochs = 6  # TODO: 10 given scaling laws?

    def __init__(self):
        # Calculate total tokens from binary files
        import glob
        import os

        # Get all training files matching the pattern
        train_files_list = glob.glob(self.train_files)

        # Calculate total tokens by summing file sizes (since they're binary files)
        total_bytes = sum(os.path.getsize(f) for f in train_files_list)
        # Assuming each token is stored as a 32-bit (4-byte) integer
        self.train_tokens = total_bytes // 4

        # Round to nearest multiple of 8192 for efficiency
        self.train_tokens = (self.train_tokens // 8192) * 8192

        # Calculate number of iterations
        self.num_iterations = (self.train_tokens * self.epochs) // self.batch_size

    # Rest of your parameters
    cooldown_frac = 0.4
    val_loss_every = 2000
    seq_len = 2 * 1024
    save_checkpoint = True
    wandb_project = "sallm"
    wandb_entity = "anri-m-lombard"
    wandb_name = None


args = Hyperparameters()

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = rank == 0  # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config={
            "batch_size": args.batch_size,
            "num_iterations": args.num_iterations,
            "cooldown_frac": args.cooldown_frac,
            "val_loss_every": args.val_loss_every,
            "seq_len": args.seq_len,
            "world_size": world_size,
        },
    )
    run_id = uuid.uuid4()
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)


def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)


# begin by printing this file (the Python code)
print0(code)
print0("=" * 100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(
    f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
)


def nvidia_smi():
    import subprocess  # avoid top level import

    return subprocess.run(
        ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ).stdout


print0(nvidia_smi())
print0("=" * 100)

# load data
train_loader = distributed_data_generator(
    args.train_files, args.batch_size, rank, world_size
)

# 153M parameters
model = GPT(vocab_size=50257, num_layers=8, num_heads=8, model_dim=512).cuda()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
embed_params = [model.embed.weight, *model.value_embeds.parameters()]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# init the optimizer(s)
adam_params = [
    dict(params=head_params, lr=0.008),
    dict(params=embed_params, lr=0.6),
    dict(params=scalar_params, lr=0.04),
]
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), fused=True, eps=1e-10)
optimizer2 = Muon(
    hidden_matrix_params, lr=0.025, momentum=0.95, rank=rank, world_size=world_size
)
optimizers = [optimizer1, optimizer2]


# learning rate schedule: stable then decay
def get_lr(it: int):
    t = 1 - it / args.num_iterations  # time remaining in training
    assert 1 >= t >= 0
    w = min(t / args.cooldown_frac, 1.0)  # 1 -> 0
    return w * 1.0 + (1 - w) * 0.1


schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]


@lru_cache(1)
def sw_num_blks(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(
        non_blocking=True
    )


def get_language_from_filename(filename: str) -> str:
    """Extract language code from validation filename."""
    parts = Path(filename).stem.split("_")
    if len(parts) < 3 or parts[0] != "custom" or parts[1] != "val":
        raise ValueError(
            f"Invalid filename format: {filename}. "
            "Expected format: custom_val_LANG_XXXXXX.bin"
        )
    lang = parts[2]
    valid_langs = {
        "afr",
        "eng",
        "nbl",
        "nso",
        "sot",
        "ssw",
        "tsn",
        "tso",
        "ven",
        "xho",
        "zul",
    }
    if lang not in valid_langs:
        raise ValueError(
            f"Invalid language code '{lang}' in filename: {filename}. "
            f"Must be one of: {sorted(valid_langs)}"
        )
    return lang


class LanguageValidationStats:
    def __init__(self):
        self.total_loss = 0.0
        self.total_accuracy = 0.0
        self.total_tokens = 0

    def update(self, loss: float, accuracy: float, num_tokens: int):
        self.total_loss += loss * num_tokens
        self.total_accuracy += accuracy * num_tokens
        self.total_tokens += num_tokens

    def get_average_loss(self):
        return self.total_loss / self.total_tokens if self.total_tokens > 0 else 0.0

    def get_average_accuracy(self):
        return self.total_accuracy / self.total_tokens if self.total_tokens > 0 else 0.0

    def get_perplexity(self):
        return torch.exp(torch.tensor(self.get_average_loss())).item()


def evaluate_validation_by_language(
    model,
    val_files_pattern: str,
    world_size: int,
    rank: int,
    seq_len: int,
    window_size: int,
):
    """Evaluate validation loss and accuracy separately for each language."""
    model.eval()
    val_files = sorted(glob.glob(val_files_pattern))
    lang_files = defaultdict(list)
    for file in val_files:
        lang = get_language_from_filename(file)
        lang_files[lang].append(file)

    lang_stats = {lang: LanguageValidationStats() for lang in lang_files}

    for lang, files in lang_files.items():
        for file in files:
            val_bs = world_size * seq_len
            tokens = _load_data_shard(Path(file))
            num_batches = len(tokens) // val_bs

            with torch.no_grad():
                for i in range(num_batches):
                    start_idx = i * val_bs + rank * seq_len
                    end_idx = start_idx + seq_len + 1
                    if end_idx <= len(tokens):
                        batch = tokens[start_idx:end_idx]
                        x = batch[:-1].to(
                            device="cuda", dtype=torch.int32, non_blocking=True
                        )
                        y = batch[1:].to(
                            device="cuda", dtype=torch.int64, non_blocking=True
                        )
                        loss, accuracy = model(x, y, sw_num_blks(window_size))

                        # Reduce loss and accuracy across all processes
                        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                        dist.all_reduce(accuracy, op=dist.ReduceOp.AVG)

                        lang_stats[lang].update(loss.item(), accuracy.item(), seq_len)

    return lang_stats


model: nn.Module = torch.compile(model)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = step == train_steps
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.perf_counter()
    timed_steps = (
        float("nan") if step <= 11 else (step - 10) + 1
    )  # <= 11 to avoid bug in val

    if master_process:
        approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
        current_lr = schedulers[0].get_last_lr()[
            0
        ]  # Get learning rate from first scheduler

        wandb.log(
            {
                "train/step_time_ms": approx_time / timed_steps,
                "train/total_time_ms": approx_time,
                "train/learning_rate": current_lr,
                "train/momentum": optimizer2.param_groups[0]["momentum"],
            },
            step=step,
        )

    # Linearly increase the block-wise sliding window size over training 128 -> 1792:
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * step / train_steps, n=128)
    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # Stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)

        # Evaluate each language separately
        lang_stats = evaluate_validation_by_language(
            model, args.val_files, world_size, rank, args.seq_len, window_size
        )

        if master_process:
            # Log overall statistics
            total_loss = sum(stats.total_loss for stats in lang_stats.values())
            total_accuracy = sum(stats.total_accuracy for stats in lang_stats.values())
            total_tokens = sum(stats.total_tokens for stats in lang_stats.values())

            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
            avg_accuracy = total_accuracy / total_tokens if total_tokens > 0 else 0.0
            overall_perplexity = torch.exp(torch.tensor(avg_loss)).item()

            # Log to wandb
            metrics = {
                "val/overall_loss": avg_loss,
                "val/overall_accuracy": avg_accuracy,
                "val/overall_perplexity": overall_perplexity,
                "train/step": step,
                "train/progress": step / train_steps,
                "train/window_size": window_size,
            }

            # Add per-language metrics
            for lang, stats in lang_stats.items():
                metrics.update(
                    {
                        f"val/{lang}/loss": stats.get_average_loss(),
                        f"val/{lang}/accuracy": stats.get_average_accuracy(),
                        # f"val/{lang}/perplexity": stats.get_perplexity(),
                        # f"val/{lang}/tokens": stats.total_tokens,
                    }
                )

            wandb.log(metrics, step=step)

            # Print validation results
            print0(
                f"step:{step}/{train_steps} overall_val_loss:{avg_loss:.4f} "
                f"overall_accuracy:{avg_accuracy:.4f} overall_perplexity:{overall_perplexity:.4f}",
                console=True,
            )
            for lang, stats in sorted(lang_stats.items()):
                print0(
                    f"{lang}: loss={stats.get_average_loss():.4f} perplexity={stats.get_perplexity():.4f} tokens={stats.total_tokens}",
                    console=True,
                )

        model.train()
        # Start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(
                step=step,
                code=code,
                model=model.state_dict(),
                optimizers=[opt.state_dict() for opt in optimizers],
            )
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    inputs, targets = next(train_loader)
    for input_seq, target_seq in zip(
        inputs.split(args.seq_len), targets.split(args.seq_len)
    ):
        loss, accuracy = model(input_seq, target_seq, sw_num_blks(window_size))
        loss.backward()

        if master_process:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/accuracy": accuracy.item(),
                },
                step=step,
            )
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    # momentum warmup for Muon
    frac = min(step / 300, 1)
    for group in optimizer2.param_groups:
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(
        f"step:{step + 1}/{train_steps} train_time:{approx_time:.0f}ms step_avg:{approx_time / timed_steps:.2f}ms",
        console=True,
    )

if master_process:
    wandb.finish()

print0(
    f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
)
dist.destroy_process_group()
