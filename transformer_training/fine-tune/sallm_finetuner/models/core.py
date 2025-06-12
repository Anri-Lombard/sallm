import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from ..config import ModelConfig


def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

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
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        seq_len = x_BTHD.size(1)
        cos, sin = self.cos[None, :seq_len, None, :], self.sin[None, :seq_len, None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


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


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_w = nn.Parameter(torch.empty(3, dim, dim))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(self.head_dim)
        self.c_proj = CastedLinear(dim, dim)
        self.attn_scale = 0.12
        std = 0.5 * (dim**-0.5)
        bound = (3**0.5) * std
        nn.init.uniform_(self.qkv_w, -bound, bound)
        self.c_proj.weight.detach().zero_()

    def forward(self, x: Tensor, ve: Tensor | None):
        B, T, C = x.shape
        qkv = F.linear(x, self.qkv_w.flatten(end_dim=1))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        if ve is not None:
            ve_reshaped = ve.view(B, T, self.num_heads, self.head_dim)
            v = self.lambdas[0] * v + self.lambdas[1] * ve_reshaped
        else:
            v = self.lambdas[0] * v
        q, k = rms_norm(q), rms_norm(k)
        q, k = self.rotary(q), self.rotary(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, scale=self.attn_scale, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
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
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.attn = CausalSelfAttention(model_dim, num_heads)
        self.mlp = MLP(model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x, ve, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(rms_norm(x), ve)
        x = x + self.mlp(rms_norm(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.model_dim)
        self.value_embeds = ValueEmbedding(
            config.vocab_size, config.model_dim, config.num_layers
        )
        self.blocks = nn.ModuleList(
            [
                Block(config.model_dim, config.num_heads)
                for _ in range(config.num_layers)
            ]
        )
        self.num_encoder_layers = config.num_layers // 2
        self.num_decoder_layers = config.num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        x0 = rms_norm(x)
        ve = self.value_embeds(input_ids)
        ve_enc = ve[: self.num_encoder_layers]
        ve_dec = ve[self.num_encoder_layers :]
        skip_connections = []
        x = x0
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0)
            skip_connections.append(x)
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            block_idx = self.num_encoder_layers + i
            x = self.blocks[block_idx](x, ve_dec[i], x0)
        return rms_norm(x)
