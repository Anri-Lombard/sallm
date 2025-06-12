import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


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
