from collections import defaultdict
import glob
import os
import sys
import wandb
import itertools
import argparse

with open(sys.argv[0]) as f:
    code = f.read()
import uuid
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GPT model with configurable hyperparameters"
    )

    parser.add_argument(
        "--train_files",
        type=str,
        default="/scratch/lmbanr001/masters/dataset/filtered_data_binary_custom_tokenizer/custom_train_*.bin",
        help="Pattern for training files",
    )
    parser.add_argument(
        "--val_files",
        type=str,
        default="/scratch/lmbanr001/masters/dataset/filtered_data_binary_custom_tokenizer/custom_val_*.bin",
        help="Pattern for validation files",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64 * 1024, help="Batch size for training"
    )
    parser.add_argument(
        "--seq_len", type=int, default=2048, help="Sequence length for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )

    parser.add_argument(
        "--cooldown_frac",
        type=float,
        default=0.4,
        help="Cooldown fraction for learning rate schedule",
    )
    parser.add_argument(
        "--val_loss_every", type=int, default=2000, help="Validate every N steps"
    )
    parser.add_argument(
        "--head_lr", type=float, default=0.008, help="Learning rate for head parameters"
    )
    parser.add_argument(
        "--embed_lr",
        type=float,
        default=0.6,
        help="Learning rate for embedding parameters",
    )
    parser.add_argument(
        "--scalar_lr",
        type=float,
        default=0.04,
        help="Learning rate for scalar parameters",
    )
    parser.add_argument(
        "--hidden_lr",
        type=float,
        default=0.025,
        help="Learning rate for hidden matrix parameters",
    )
    parser.add_argument(
        "--momentum_start", type=float, default=0.85, help="Starting momentum value"
    )
    parser.add_argument(
        "--momentum_end", type=float, default=0.95, help="Final momentum value"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.8, help="Beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="Beta2 for Adam optimizer"
    )
    parser.add_argument(
        "--adam_eps", type=float, default=1e-10, help="Epsilon for Adam optimizer"
    )

    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument(
        "--num_layers", type=int, default=8, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--model_dim", type=int, default=512, help="Model dimension")

    parser.add_argument(
        "--save_checkpoint",
        action="store_true",
        help="Whether to save the FINAL checkpoint at the end of training",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="sallm-transformer-hyperparameter-tuning",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="anri-m-lombard",
        help="Weights & Biases entity name",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="transformer-experiment-1",
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/scratch/lmbanr001/masters/sallm/trained_models/final",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--save_epochs",
        type=str,
        default="",
        help="Comma-separated list of epochs to save checkpoints for (e.g., '10,20,30').",
    )
    parser.add_argument(
        "--save_best_checkpoint",
        action="store_true",
        help="Save the checkpoint with the best validation loss seen so far.",
    )

    return parser.parse_args()


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True

import torch._inductor.config

backends = torch._inductor.config.max_autotune_gemm_backends
torch._inductor.config.max_autotune_gemm_backends = "ATEN," + backends

torch.empty(1, device="cuda", requires_grad=True).backward()
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.nn.attention.flex_attention import BlockMask, flex_attention

torch._inductor.config.coordinate_descent_tuning = True


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
    x, w, *_ = inputs
    ctx.save_for_backward(x, w)
    ctx.set_materialize_grads(False)


mm_op.register_autograd(backward, setup_context=setup_context)


def lm_head_fp8(x: Tensor, w: Tensor) -> Tensor:
    _x = x.flatten(0, -2)
    out: Tensor = torch.nn.functional.linear(
        _x.to(torch.bfloat16), w.to(torch.bfloat16)
    )
    return out.reshape(*x.shape[:-1], -1)


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
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
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None

            def update_prev():
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
                update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features**-0.5)
        bound = (3**0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len=65536):
        super().__init__()
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
        bound = (3**0.5) * std
        self.qkv_w = nn.Parameter(torch.empty(3, dim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(dim // num_heads)
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.detach().zero_()
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1)
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = (
            F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x))
            .view(B, T, 3 * self.num_heads, -1)
            .chunk(3, dim=-2)
        )
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
            scale=self.attn_scale,
        )
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.detach().zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
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
        num_unique_embeddings = min(3, (num_layers + 2) // 3)
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
        pattern = []
        for i in range(self.num_layers):
            if i < len(ve):
                pattern.append(ve[i])
            else:
                idx = i % len(ve)
                pattern.append(ve[idx] if idx < len(ve) else None)
        return pattern


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class GPT(nn.Module):
    def __init__(
        self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = ValueEmbedding(vocab_size, model_dim, num_layers)
        self.blocks = nn.ModuleList(
            [Block(model_dim, num_heads, layer_idx) for layer_idx in range(num_layers)]
        )
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128))
        self.lm_head.weight.detach().zero_()

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

        long_bm, short_bm = create_doc_swc_block_masks(sliding_window_num_blocks)
        x = x0 = norm(self.embed(input_seq)[None])
        ve = self.value_embeds(input_seq)
        assert len(ve) == len(self.blocks)
        ve_enc, ve_dec = ve[: self.num_encoder_layers], ve[self.num_encoder_layers :]
        assert (
            len(ve_enc) == self.num_encoder_layers
            and len(ve_dec) == self.num_decoder_layers
        )
        base_pattern = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm]
        encoder_block_masks = []
        for i in range(self.num_encoder_layers):
            encoder_block_masks.append(base_pattern[i % len(base_pattern)])
        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0, encoder_block_masks[i])
            skip_connections.append(x)
        decoder_block_masks = []
        for i in range(self.num_decoder_layers):
            idx = (self.num_decoder_layers - 1 - i) % len(base_pattern)
            decoder_block_masks.append(base_pattern[idx])
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.blocks[self.num_encoder_layers + i](
                x, ve_dec[i], x0, decoder_block_masks[i]
            )
        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.sigmoid(logits.float() / 7.5)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq)
        predictions = logits.view(-1, logits.size(-1)).argmax(dim=-1)
        correct_predictions = (predictions == target_seq).float().mean()
        return loss, correct_predictions


def _load_data_shard(file: Path):
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def distributed_data_generator(
    filename_pattern: str, batch_size: int, rank: int, world_size: int
):
    pattern_path = Path(filename_pattern)
    if pattern_path.is_absolute():
        files = sorted(pattern_path.parent.glob(pattern_path.name))
    else:
        files = sorted(Path.cwd().glob(filename_pattern))
    assert len(files) > 0, f"No files found matching pattern: {filename_pattern}"
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_cycle = itertools.cycle(files)
    tokens, pos = _load_data_shard(next(file_cycle)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = (
                _load_data_shard(next(file_cycle)),
                0,
            )
        buf = tokens[pos + rank * local_batch_size :][: local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs, targets


@dataclass
class Hyperparameters:
    def __init__(self, args):
        self.train_files = args.train_files
        self.val_files = args.val_files
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.epochs = args.epochs
        self.cooldown_frac = args.cooldown_frac
        self.val_loss_every = args.val_loss_every
        self.head_lr = args.head_lr
        self.embed_lr = args.embed_lr
        self.scalar_lr = args.scalar_lr
        self.hidden_lr = args.hidden_lr
        self.momentum_start = args.momentum_start
        self.momentum_end = args.momentum_end
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.adam_eps = args.adam_eps
        self.vocab_size = args.vocab_size
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.model_dim = args.model_dim
        self.save_checkpoint = args.save_checkpoint
        self.wandb_project = args.wandb_project
        self.wandb_entity = args.wandb_entity
        self.wandb_name = args.wandb_name
        self.checkpoint_dir = args.checkpoint_dir
        self.save_epochs = args.save_epochs
        self.save_best_checkpoint = args.save_best_checkpoint

        import glob
        import torch

        train_files_list = glob.glob(self.train_files)
        total_tokens = 0
        for file in train_files_list:
            header = torch.from_file(str(file), False, 256, dtype=torch.int32)
            assert header[0] == 20240520, f"Invalid magic number in {file}"
            assert header[1] == 1, f"Unsupported version in {file}"
            total_tokens += int(header[2])
        self.train_tokens = (total_tokens // 8192) * 8192
        self.num_iterations = (self.train_tokens * self.epochs) // self.batch_size


def main():
    args = parse_args()
    hparams = Hyperparameters(args)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    master_process = rank == 0

    logfile = None
    if master_process:
        wandb.init(
            project=hparams.wandb_project,
            entity=hparams.wandb_entity,
            name=hparams.wandb_name,
            config={
                k: v
                for k, v in vars(hparams).items()
                if k not in ["train_files", "val_files", "save_checkpoint"]
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

    print0(code)
    print0("=" * 100)
    print0(f"Running Python {sys.version}")
    print0(
        f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
    )

    def nvidia_smi():
        import subprocess

        return subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ).stdout

    print0(nvidia_smi())
    print0("=" * 100)

    train_loader = distributed_data_generator(
        hparams.train_files, hparams.batch_size, rank, world_size
    )
    model = GPT(
        vocab_size=hparams.vocab_size,
        num_layers=hparams.num_layers,
        num_heads=hparams.num_heads,
        model_dim=hparams.model_dim,
    ).cuda()
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    hidden_matrix_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
    embed_params = [model.embed.weight, *model.value_embeds.parameters()]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    adam_params = [
        dict(params=head_params, lr=hparams.head_lr),
        dict(params=embed_params, lr=hparams.embed_lr),
        dict(params=scalar_params, lr=hparams.scalar_lr),
    ]
    optimizer1 = torch.optim.Adam(
        adam_params,
        betas=(hparams.adam_beta1, hparams.adam_beta2),
        fused=True,
        eps=hparams.adam_eps,
    )
    optimizer2 = Muon(
        hidden_matrix_params,
        lr=hparams.hidden_lr,
        momentum=hparams.momentum_end,
        rank=rank,
        world_size=world_size,
    )
    optimizers = [optimizer1, optimizer2]

    def get_lr(it: int):
        t = 1 - it / hparams.num_iterations
        assert 1 >= t >= 0
        w = min(t / hparams.cooldown_frac, 1.0)
        return w * 1.0 + (1 - w) * 0.1

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    @lru_cache(1)
    def sw_num_blks(window_size: int):
        return torch.tensor(
            window_size // 128, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

    def get_language_from_filename(filename: str) -> str:
        parts = Path(filename).stem.split("_")
        if len(parts) < 3 or parts[0] != "custom" or parts[1] != "val":
            raise ValueError(
                f"Invalid filename format: {filename}. Expected format: custom_val_LANG_XXXXXX.bin"
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
                f"Invalid language code '{lang}' in filename: {filename}. Must be one of: {sorted(valid_langs)}"
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
            return (
                self.total_accuracy / self.total_tokens
                if self.total_tokens > 0
                else 0.0
            )

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
                            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                            dist.all_reduce(accuracy, op=dist.ReduceOp.AVG)
                            lang_stats[lang].update(
                                loss.item(), accuracy.item(), seq_len
                            )
        return lang_stats

    model: nn.Module = torch.compile(model)

    try:
        save_epochs_set = (
            set(map(int, hparams.save_epochs.split(",")))
            if hparams.save_epochs
            else set()
        )
        print0(f"Will save checkpoints at epochs: {sorted(list(save_epochs_set))}")
    except ValueError:
        print0(
            f"Error: Invalid format for --save_epochs: '{hparams.save_epochs}'. Should be comma-separated integers.",
            console=True,
        )
        save_epochs_set = set()

    if master_process:
        os.makedirs(hparams.checkpoint_dir, exist_ok=True)
        print0(f"Checkpoints will be saved in: {hparams.checkpoint_dir}")

    best_val_loss = float("inf")
    steps_per_epoch = hparams.train_tokens // hparams.batch_size

    training_time_ms = 0
    running_loss = 0.0
    running_accuracy = 0.0  # Initialize running accuracy
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_steps = hparams.num_iterations

    for step in range(train_steps + 1):
        last_step = step == train_steps
        if step == 10:
            training_time_ms = 0
            t0 = time.perf_counter()
        timed_steps = float("nan") if step <= 11 else (step - 10) + 1
        current_epoch = (step // steps_per_epoch) + 1 if steps_per_epoch > 0 else 1

        if master_process and step > 0:  # Avoid logging at step 0 before timing starts
            approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
            current_lr = schedulers[0].get_last_lr()[0]
            wandb.log(
                {
                    "train/step_time_ms": approx_time / timed_steps,
                    "train/total_time_ms": approx_time,
                    "train/learning_rate": current_lr,
                    "train/momentum": optimizer2.param_groups[0]["momentum"],
                    "train/epoch": current_epoch,
                },
                step=step,
            )

        window_size = next_multiple_of_n(1728 * step / train_steps, n=128)

        if last_step or (
            hparams.val_loss_every > 0
            and step % hparams.val_loss_every == 0
            and step > 0
        ):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            lang_stats = evaluate_validation_by_language(
                model, hparams.val_files, world_size, rank, hparams.seq_len, window_size
            )

            if master_process:
                total_loss = sum(stats.total_loss for stats in lang_stats.values())
                total_accuracy = sum(
                    stats.total_accuracy for stats in lang_stats.values()
                )
                total_tokens = sum(stats.total_tokens for stats in lang_stats.values())
                avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
                avg_accuracy = (
                    total_accuracy / total_tokens if total_tokens > 0 else 0.0
                )
                overall_perplexity = torch.exp(torch.tensor(avg_loss)).item()

                is_best = hparams.save_best_checkpoint and avg_loss < best_val_loss
                save_this_epoch = current_epoch in save_epochs_set
                checkpoint_path = None  # Define to check later

                if is_best:
                    best_val_loss = avg_loss
                    print0(
                        f"*** New best validation loss {avg_loss:.4f} at step {step} (epoch ~{current_epoch}) ***",
                        console=True,
                    )
                    checkpoint_path = os.path.join(
                        hparams.checkpoint_dir, "checkpoint_best_loss.pt"
                    )
                    print0(f"Saving best checkpoint to {checkpoint_path}", console=True)
                    log_best = dict(
                        step=step,
                        epoch=current_epoch,
                        best_val_loss=best_val_loss,
                        model=model.state_dict(),
                        optimizers=[opt.state_dict() for opt in optimizers],
                    )
                    torch.save(log_best, checkpoint_path)

                if save_this_epoch:
                    epoch_checkpoint_path = os.path.join(
                        hparams.checkpoint_dir, f"checkpoint_epoch_{current_epoch}.pt"
                    )
                    print0(
                        f"Saving checkpoint for epoch {current_epoch} to {epoch_checkpoint_path}",
                        console=True,
                    )
                    log_epoch = dict(
                        step=step,
                        epoch=current_epoch,
                        current_val_loss=avg_loss,
                        model=model.state_dict(),
                        optimizers=[opt.state_dict() for opt in optimizers],
                    )
                    if not (is_best and checkpoint_path == epoch_checkpoint_path):
                        torch.save(log_epoch, epoch_checkpoint_path)

                metrics = {
                    "val/overall_loss": avg_loss,
                    "val/overall_accuracy": avg_accuracy,
                    "val/overall_perplexity": overall_perplexity,
                    "train/step": step,
                    "train/progress": step / train_steps,
                    "train/window_size": window_size,
                    "val/best_loss": best_val_loss,
                }
                for lang, stats in lang_stats.items():
                    metrics.update(
                        {
                            f"val/{lang}/loss": stats.get_average_loss(),
                            f"val/{lang}/accuracy": stats.get_average_accuracy(),
                        }
                    )
                wandb.log(metrics, step=step)

                print0(
                    f"step:{step}/{train_steps} overall_val_loss:{avg_loss:.4f} overall_accuracy:{avg_accuracy:.4f} overall_perplexity:{overall_perplexity:.4f}",
                    console=True,
                )
                for lang, stats in sorted(lang_stats.items()):
                    print0(
                        f"{lang}: loss={stats.get_average_loss():.4f} perplexity={stats.get_perplexity():.4f} tokens={stats.total_tokens}",
                        console=True,
                    )

            model.train()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if master_process and hparams.save_checkpoint:
                final_checkpoint_path = os.path.join(
                    hparams.checkpoint_dir, f"checkpoint_final_step{step}.pt"
                )
                print0(
                    f"Saving final checkpoint to {final_checkpoint_path}", console=True
                )
                log_final = dict(
                    step=step,
                    epoch=current_epoch,
                    final_val_loss=avg_loss if "avg_loss" in locals() else float("nan"),
                    model=model.state_dict(),
                    optimizers=[opt.state_dict() for opt in optimizers],
                )
                torch.save(log_final, final_checkpoint_path)
            break

        inputs, targets = next(train_loader)
        for input_seq, target_seq in zip(
            inputs.split(hparams.seq_len), targets.split(hparams.seq_len)
        ):
            loss, accuracy = model(input_seq, target_seq, sw_num_blks(window_size))
            loss.backward()

            if master_process:
                loss_item = loss.item()
                accuracy_item = accuracy.item()
                if step > 0:
                    running_loss = running_loss * step / (step + 1) + loss_item / (
                        step + 1
                    )
                    running_accuracy = running_accuracy * step / (
                        step + 1
                    ) + accuracy_item / (step + 1)
                else:  # Handle step 0
                    running_loss = loss_item
                    running_accuracy = accuracy_item
                wandb.log(
                    {
                        "train/loss": loss_item,
                        "train/running_loss": running_loss,
                        "train/accuracy": accuracy_item,
                        "train/running_accuracy": running_accuracy,  # Log running accuracy
                    },
                    step=step,
                )

        for param in model.parameters():
            if param.grad is not None:  # Check if grad exists before reducing
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        frac = min(step / 300, 1)
        for group in optimizer2.param_groups:
            group["momentum"] = (
                1 - frac
            ) * hparams.momentum_start + frac * hparams.momentum_end

        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()

        model.zero_grad(set_to_none=True)

        if master_process and step > 0:  # Avoid printing at step 0
            approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
            print0(
                f"step:{step + 1}/{train_steps} train_time:{approx_time:.0f}ms step_avg:{approx_time / timed_steps:.2f}ms epoch:{current_epoch}",
                console=True,
            )

    if master_process:
        wandb.finish()

    print0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
