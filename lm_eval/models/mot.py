"""
Modified from https://github.com/KellerJordan/modded-nanogpt/blob/master/records/021425_GPT2MediumOptCoeffs/1baa66b2-bff7-4850-aced-d63885ffb4b6.txt
"""

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import einops

#torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
import tiktoken
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
import torch.nn.functional as F
from torch import Tensor, nn
import safetensors.torch
from huggingface_hub import hf_hub_download

# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import MODEL_REGISTRY, register_model


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model
@dataclass
class ByteHyperparameters:
    bytes_per_token: int = 16
    vocab_size: int = 458
    byte_mixin_method: Literal["cross_attn", "concat", "noop"] = "noop"
    byte_mixout_method: Literal["noop", "copy", "split"] = "noop"
    use_byte_self_attn: bool = False
    padding_in: Literal["left", "right"] = "left"
    padding_out: Literal["left", "right"] = "left"
    pull_in: bool = True
    pull_out: bool = True
    add_padded_and_pulled: bool = False
    sliding_window_tokens: int = 8
    n_layer_out: int = 1
    mix_bytes_within_tok_in: bool = False
    mix_bytes_within_tok_out: bool = False


@dataclass
class ModelDims:
    model_dim: int = 768
    byte_dim: int = 768
    token_dim: int = 768
    expansion_factor: float = 4.0


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3), f"{self.cos.shape=}, {x_BTHD.shape=}"
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class CrossAttention(nn.Module):
    """
    Only project bytes_per_token bytes into their one corresponding token
    --> causality or blocks are irrelevant

    But do add rotary embeddings
    """
    def __init__(self, dim: int, num_heads: int, max_seq_len_q: int, max_seq_len_kv: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.q_w = nn.Parameter(torch.empty(hdim, dim).uniform_(-bound, bound))
        self.kv_w = nn.Parameter(torch.empty(2, hdim, dim).uniform_(-bound, bound))
        self.lambda_factor = nn.Parameter(torch.tensor(0.5))
        self.rotary_q = Rotary(head_dim, max_seq_len_q)
        self.rotary_k = Rotary(head_dim, max_seq_len_kv)
        self.c_proj = CastedLinear(hdim, dim)  # No zero init because there won't be a residual!!! TODO: check if a residaul makes sense
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.12

    def forward(self, xq: Tensor, xkv: Tensor):
        Bq, Tq = xq.size(0), xq.size(1)
        Bkv, Tkv = xkv.size(0), xkv.size(1)
        assert Bq == Bkv == 1, "Must use batch size = 1 for FlexAttention"
        k, v = F.linear(xkv, self.kv_w.flatten(end_dim=1).type_as(xkv)).view(Bq, Tkv, 2 * self.num_heads, self.head_dim).chunk(2, dim=-2)
        q = F.linear(xq, self.q_w.type_as(xq)).view(Bq, Tq, self.num_heads, self.head_dim)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary_q(q), self.rotary_k(k)
        v = self.lambda_factor * v

        # Because we always attend from n chars to 1 token, we can re-shape, use BMM, and save use the attention mask
        chars_per_token = Tkv // Tq
        q = q.transpose(1, 2).unsqueeze(3)  # einops.rearrange(q, "b tq h d -> b h tq 1 d")
        k = k.view(k.shape[0], k.shape[2], -1, chars_per_token, k.shape[3])  # einops.rearrange(k, "b (t c) h d -> b h t c d", c=chars_per_token)
        v = v.view(v.shape[0], v.shape[2], -1, chars_per_token, v.shape[3])

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        y = torch.matmul(attn_weights, v)
        y = y.squeeze(3).transpose(1, 2)  # einops.rearrange(y, "b h tq 1 d -> b tq h d")

        y = y.contiguous().view(Bq, Tq, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, dim: int, expansion_factor: float = 4.0):
        super().__init__()
        hdim = next_multiple_of_n(int(expansion_factor * dim), n=128)
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dims: ModelDims, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dims.model_dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dims.model_dim, dims.expansion_factor)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x


class FlexibleEmbedding(nn.Module):
    def __init__(self, dims: ModelDims, vocab_size, byte_params: ByteHyperparameters):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dims.token_dim if byte_params.byte_mixin_method != "noop" else dims.model_dim)
        self.embed_bytes = nn.Embedding(byte_params.vocab_size, dims.byte_dim) if byte_params.byte_mixin_method != "noop" else nn.Identity()

        if byte_params.byte_mixin_method == "noop":
            self.forward = self._forward_tokens
        elif not byte_params.pull_in:
            self.forward = self._forward_bytes_padded
        elif not byte_params.add_padded_and_pulled:
            self.forward = self._forward_bytes_pulled
        else:
            self.forward = self._forward_bytes_padded_and_pulled

    def _forward_tokens(
            self,
            tokens: Tensor,
            byte_tensor: Tensor | None,
            byte_tensor_pulled: Tensor | None,
    ) -> tuple[Tensor, None]:
        return norm(self.embed_tokens(tokens)), None
    
    def _forward_bytes_padded(
            self,
            tokens: Tensor,
            byte_tensor: Tensor,
            byte_tensor_pulled: Tensor,
    ) -> tuple[Tensor, Tensor]:
        token_embs = norm(self.embed_tokens(tokens))
        byte_embs = norm(self.embed_bytes(byte_tensor))
        return token_embs, byte_embs


    def _forward_bytes_pulled(
            self,
            tokens: Tensor,
            byte_tensor: Tensor | None,
            byte_tensor_pulled: Tensor,
    ) -> tuple[Tensor, Tensor]:
        token_embs = norm(self.embed_tokens(tokens))
        byte_embs = norm(self.embed_bytes(byte_tensor_pulled))
        return token_embs, byte_embs

    def _forward_bytes_padded_and_pulled(
            self,
            tokens: Tensor,
            byte_tensor: Tensor,
            byte_tensor_pulled: Tensor,
    ) -> tuple[Tensor, Tensor]:
        token_embs = norm(self.embed_tokens(tokens))
        byte_embs = norm(self.embed_bytes(byte_tensor) + self.embed_bytes(byte_tensor_pulled))
        return token_embs, byte_embs


class ByteSelfAttn(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, byte_params: ByteHyperparameters, mix_byte_in_tok: bool = False):
        super().__init__()
        self.byte_params = byte_params
        self.mix_byte_in_tok = mix_byte_in_tok
        self.attention = CausalSelfAttention(
            dim=dim,
            num_heads=max(1, dim//128),
            max_seq_len=max_seq_len * byte_params.bytes_per_token,
            head_dim=128,
        ) if byte_params.use_byte_self_attn else nn.Identity()

        swt = self.byte_params.sliding_window_tokens
        bpt = self.byte_params.bytes_per_token

        def sliding_window_causal_mask(b, h, q_idx, kv_idx):
            causality = q_idx >= kv_idx
            sliding_window = q_idx - kv_idx < swt * bpt
            return causality & sliding_window

        def sliding_window_block_causal_mask(b, h, q_idx, kv_idx):
            block_causality = q_idx // bpt >= kv_idx // bpt
            sliding_window = q_idx - kv_idx < swt * bpt
            return block_causality & sliding_window
        
        self.block_masks = [
            create_block_mask(
                mask_mod=sliding_window_block_causal_mask if self.mix_byte_in_tok else sliding_window_causal_mask,
                B=None,
                H=None,
                Q_LEN=(T+1)*bpt,
                KV_LEN=(T+1)*bpt,
            )
            for T in range(4096)
        ] if self.byte_params.use_byte_self_attn else None

    def forward(self, byte_embs: Tensor) -> Tensor:
        T = byte_embs.size(-2) // self.byte_params.bytes_per_token
        if self.byte_params.use_byte_self_attn:
            byte_embs = byte_embs + self.attention(byte_embs, None, self.block_masks[T-1])
        return byte_embs


class ByteMixinNoop(nn.Module):
    def __init__(self, dims: ModelDims, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()
        self.attention = self.mixin = nn.Identity()

    def forward(self, x, *args):
        return x


class ByteMixinConcat(nn.Module):
    def __init__(self, dims: ModelDims, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()
        self.byte_params = byte_params
        self.attention = ByteSelfAttn(
            dims.byte_dim, max_seq_len, byte_params, byte_params.mix_bytes_within_tok_in
        ) if byte_params.use_byte_self_attn else nn.Identity()
        self.mixin = CastedLinear(dims.token_dim + dims.byte_dim * byte_params.bytes_per_token, dims.model_dim)

    def forward(self, tok_embs: Tensor, byte_embs: Tensor) -> Tensor:
        if self.byte_params.use_byte_self_attn:
            byte_embs = self.attention(byte_embs)
        byte_embs = einops.rearrange(byte_embs, "B (S bpt) D -> B S (bpt D)", bpt=self.byte_params.bytes_per_token)
        return norm(self.mixin(torch.cat([tok_embs, byte_embs], dim=-1)))


class ByteMixinCrossAttn(nn.Module):
    def __init__(self, dims: ModelDims, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()
        assert dims.byte_dim == dims.token_dim == dims.model_dim
        self.byte_params = byte_params
        self.attention = ByteSelfAttn(
            dims.byte_dim, max_seq_len, byte_params, byte_params.mix_bytes_within_tok_in
        ) if byte_params.use_byte_self_attn else nn.Identity()
        self.mixin = CrossAttention(
            dim=dims.model_dim,
            num_heads=dims.model_dim//128,
            max_seq_len_kv=max_seq_len * byte_params.bytes_per_token,
            max_seq_len_q=max_seq_len,
            head_dim=128,
        )
    
    def forward(self, token_embs: Tensor, byte_embs: Tensor) -> Tensor:
        byte_embs = self.attention(byte_embs)
        return self.mixin(xq=token_embs, xkv=byte_embs)


class ByteMixin(nn.Module):
    def __init__(self, dims: ModelDims, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()
        if byte_params.byte_mixin_method == "noop":
            self.mixin = ByteMixinNoop(dims, max_seq_len, byte_params)
        elif byte_params.byte_mixin_method == "cross_attn":
            self.mixin = ByteMixinCrossAttn(dims, max_seq_len, byte_params)
        elif byte_params.byte_mixin_method == "concat":
            self.mixin = ByteMixinConcat(dims, max_seq_len, byte_params)
        else:
            raise RuntimeError(f"Invalid byte mixin method: {byte_params.byte_mixin_method}")
    
    def forward(self, tok_embs: Tensor, byte_embs: Tensor) -> Tensor:
        return self.mixin(tok_embs, byte_embs)


class ByteMixoutCopy(nn.Module):
    def __init__(self, dims: ModelDims, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()
        self.attention_layers = nn.ModuleList([
            ByteSelfAttn(dims.model_dim, max_seq_len, byte_params, byte_params.mix_bytes_within_tok_out)  # use model dim at output
            for _ in range(byte_params.n_layer_out)
        ])
        self.bpt = byte_params.bytes_per_token

    def forward(self, x: Tensor) -> Tensor:
        x = einops.repeat(x, "... T D-> ... (T bpt) D", bpt=self.bpt)
        for layer in self.attention_layers:
            x = x + layer(norm(x))
        return x


class ByteMixoutSplit(nn.Module):
    def __init__(self, dims: ModelDims, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()
        assert dims.model_dim % byte_params.bytes_per_token == 0, f"{dims.model_dim=}, {byte_params.bytes_per_token=}"
        self.attention_layers = nn.ModuleList([
            ByteSelfAttn(
                dims.model_dim // byte_params.bytes_per_token,
                max_seq_len,
                byte_params,
                byte_params.mix_bytes_within_tok_out,
            )
            for _ in range(byte_params.n_layer_out)
        ])
        self.bpt = byte_params.bytes_per_token

    def forward(self, x: Tensor) -> Tensor:
        x = einops.rearrange(x, "... T (bpt D) -> ... (T bpt) D", bpt=self.bpt)
        for layer in self.attention_layers:
            x = x + layer(norm(x))
        return x


class ByteMixoutNoop(nn.Module):
    def __init__(self, dims: ModelDims, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class ByteMixout(nn.Module):
    def __init__(self, dims: ModelDims, max_seq_len: int, byte_params: ByteHyperparameters):
        super().__init__()
        self.mixout = {
            "noop": ByteMixoutNoop,
            "copy": ByteMixoutCopy,
            "split": ByteMixoutSplit,
        }[byte_params.byte_mixout_method](dims, max_seq_len, byte_params)

    def forward(self, x: Tensor) -> Tensor:
        return self.mixout(x)


# -----------------------------------------------------------------------------
# The main model


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class GPT(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            num_heads: int,
            model_dims: ModelDims,
            max_seq_len: int,
            byte_params: ByteHyperparameters,
    ):
        super().__init__()
        self.embed = FlexibleEmbedding(dims=model_dims, vocab_size=vocab_size, byte_params=byte_params)
        self.byte_mixin = ByteMixin(
            dims=model_dims, max_seq_len=max_seq_len, byte_params=byte_params
        )
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dims.model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dims, num_heads, max_seq_len, i) for i in range(num_layers)])
        self.byte_mixout = ByteMixout(
            dims=model_dims, max_seq_len=max_seq_len, byte_params=byte_params
        )
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        lm_head_in_dim = model_dims.model_dim if byte_params.byte_mixout_method != "split" else model_dims.model_dim // byte_params.bytes_per_token
        self.lm_head_out_dim = vocab_size if byte_params.byte_mixout_method == "noop" else byte_params.vocab_size
        self.lm_head = CastedLinear(lm_head_in_dim, next_multiple_of_n(self.lm_head_out_dim, n=128))
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers//2))

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        
        self.block_masks = [
            create_block_mask(
                mask_mod=causal_mask,
                B=None,
                H=None,
                Q_LEN=T+1,
                KV_LEN=T+1,
            )
            for T in range(4096)
        ]

    def forward(
            self,
            toks_in: Tensor,
            bytes_padded_in: Tensor | None,
            bytes_pulled_in: Tensor | None,
    ):
        T = toks_in.size(1)
        ve = [value_embed(toks_in) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        xt, xb = self.embed(tokens=toks_in, byte_tensor=bytes_padded_in, byte_tensor_pulled=bytes_pulled_in)
        x = x0 = self.byte_mixin(xt, xb)

        # U-net design by @brendanh0gan
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, self.block_masks[T-1])
            if i < n:
                skip_connections.append(x)

        x = self.byte_mixout(x)
        x = norm(x)
        logits = self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits.float() / 7.5)
        return logits[..., :self.lm_head_out_dim]  # undo the next_multiple_of_n(..., 128) -> remove impossible tokens

# -----------------------------------------------------------------------------
# Tokens to Bytes


def load_ttb(filename: str) -> dict[int, list[int]]:
    path = Path(os.path.abspath(__file__).replace("/mot.py", "")) / "embeddings" / filename
    with open(path, "r") as f:
        text = f.read()
    ttb = json.loads(text)
    ttb = {int(k): [int(x) for x in v] for k, v in ttb.items()}
    return ttb


def make_embedding(filename: str, vocab_size: int) -> nn.Embedding:
    dim = int(filename.split("_")[1])
    emb = nn.Embedding(vocab_size, dim)
    ttb = load_ttb(filename)
    for idx in ttb:
        emb.weight.data[idx] = torch.tensor(ttb[idx])
    emb.weight.requires_grad = False
    return emb


def tokens_to_bytes(tokens: Tensor, emb: nn.Embedding) -> Tensor:
    with torch.no_grad():
        byte_tensor = emb(tokens).to(torch.int64)
    if tokens.ndim == 2:
        return byte_tensor.view(byte_tensor.shape[0], -1)  # einops.rearrange(byte_tensor, "b n c -> b (n c)")
    else:
        return byte_tensor.view(-1).unsqueeze(0)  # einops.rearrange(byte_tensor, "n c -> (n c)")[None]


# Thanks Google Gemini Pro 2.5 for the 150x speedup!
@torch.compile(mode="reduce-overhead")
def pull_from_right(
    byte_tensor: Tensor, bytes_per_token: int, pad_byte: int, eot_byte: int
) -> Tensor:
    """
    Pulls valid bytes towards the left boundary of each token, considering EOT tokens
    as sequence breaks. Bytes are taken from the current token up to (but not including)
    the next EOT token. EOT tokens retain their original bytes.
    Vectorized implementation.
    """
    B, T = byte_tensor.size()
    if T == 0: # Handle empty input
        return byte_tensor
    T_reduced = T // bytes_per_token
    assert T % bytes_per_token == 0, "T must be divisible by bytes_per_token"
    if T_reduced == 0: # Handle case where T < bytes_per_token
         return torch.full_like(byte_tensor, pad_byte)

    # 1. Preprocessing
    byte_tensor_view = byte_tensor.view(B, T_reduced, bytes_per_token)
    device = byte_tensor.device

    non_pad_mask = byte_tensor_view != pad_byte  # (B, Tr, bpt)
    is_eot_token = torch.all(byte_tensor_view == eot_byte, dim=2) # (B, Tr)

    valid_bytes_per_token = non_pad_mask.sum(dim=2) # (B, Tr)

    # Cumulative count of *valid* bytes up to the start of each token
    cum_valid_bytes = torch.cumsum(
        torch.cat([torch.zeros_like(valid_bytes_per_token[:, :1]), valid_bytes_per_token], dim=1),
        dim=1
    ) # (B, Tr + 1)
    start_valid_byte_idx = cum_valid_bytes[:, :-1] # (B, Tr) - Global index of first valid byte for token t
    end_valid_byte_idx = cum_valid_bytes[:, 1:]   # (B, Tr) - Global index of last valid byte + 1 for token t
    total_valid_bytes_per_batch = cum_valid_bytes[:, -1] # (B,)

    # 2. Find Next EOT Boundary (using searchsorted loop for now)
    next_eot_token_indices = torch.full_like(is_eot_token, T_reduced, dtype=torch.long) # (B, Tr)
    for b in range(B):
        eot_pos = is_eot_token[b].nonzero(as_tuple=True)[0]
        if eot_pos.numel() > 0:
            token_indices = torch.arange(T_reduced, device=device)
            # Find index in eot_pos for the first EOT >= token_indices
            next_eot_rel_idx = torch.searchsorted(eot_pos, token_indices, side='left')
            # Clamp indices to valid range and get actual token positions
            valid_mask = next_eot_rel_idx < eot_pos.numel()
            next_eot_token_indices[b, valid_mask] = eot_pos[next_eot_rel_idx[valid_mask]]
            # Tokens after the last EOT will correctly point to T_reduced (default value)

    # 3. Calculate Byte Ranges to Pull
    # Global index of the first valid byte in the *next* EOT token (or end of sequence)
    # Use gather with the corrected indexing (no extra unsqueeze)
    next_eot_valid_byte_start_idx = cum_valid_bytes.gather(1, next_eot_token_indices) # (B, Tr)

    # Number of valid bytes available from current token up to (not including) next EOT
    available_bytes = next_eot_valid_byte_start_idx - start_valid_byte_idx # (B, Tr)
    bytes_to_pull = torch.minimum(available_bytes, torch.tensor(bytes_per_token, device=device)) # (B, Tr)
    bytes_to_pull = torch.clamp(bytes_to_pull, min=0) # Ensure non-negative

    # 4. Gather Bytes (Using Flattening Approach)
    flat_indices_b, flat_indices_t, flat_indices_k = non_pad_mask.nonzero(as_tuple=True)
    flat_valid_bytes = byte_tensor_view[flat_indices_b, flat_indices_t, flat_indices_k] # (total_valid_bytes,)

    # Mapping from batch item to its range in flat_valid_bytes
    batch_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), total_valid_bytes_per_batch.cumsum(0)[:-1]]) # (B,)

    # Create indices for the output tensor (B, Tr, bpt)
    k_indices = torch.arange(bytes_per_token, device=device).view(1, 1, bytes_per_token).expand(B, T_reduced, -1)

    # Calculate the global valid byte index we want for each output slot (b, t, k)
    # target_global_valid_idx = start_valid_byte_idx[b, t] + k
    target_global_valid_idx = start_valid_byte_idx.unsqueeze(2) + k_indices # (B, Tr, bpt)

    # Create a mask for indices we actually need to gather (k < bytes_to_pull)
    gather_mask = k_indices < bytes_to_pull.unsqueeze(2) # (B, Tr, bpt)

    # Adjust target global indices to be relative to the flattened array
    absolute_gather_idx = target_global_valid_idx + batch_offsets.view(B, 1, 1) # (B, Tr, bpt)

    # Pad flat_valid_bytes to handle potential out-of-bounds gathers safely
    total_flat_size = batch_offsets[-1] + total_valid_bytes_per_batch[-1] if B > 0 else 0
    # Use a large index for invalid gathers and replace later
    safe_indices = torch.where(gather_mask, absolute_gather_idx, total_flat_size) # Use index outside valid range

    # Add a padding value at the end of flat_valid_bytes
    # Ensure dtype matches byte_tensor
    padded_flat_valid_bytes = torch.cat([flat_valid_bytes, torch.tensor([pad_byte], device=device, dtype=byte_tensor.dtype)])

    # Perform the gather
    # Clamp indices just in case, although safe_indices should handle it
    clamped_indices = torch.clamp(safe_indices, max=total_flat_size)
    gathered_bytes_flat = padded_flat_valid_bytes[clamped_indices] # (B, Tr, bpt)

    # Reshape gathered bytes and apply padding where gather_mask was false
    pulled_non_eot = torch.where(gather_mask, gathered_bytes_flat, torch.tensor(pad_byte, device=device, dtype=byte_tensor.dtype)) # (B, Tr, bpt)

    # 5. Handle EOT Tokens
    # EOT tokens keep their original bytes, exactly as they were.
    final_pulled_tensor = torch.where(
        is_eot_token.unsqueeze(-1),
        byte_tensor_view, # Keep original bytes exactly as they were for EOTs
        pulled_non_eot      # Use the pulled bytes for non-EOTs
    )

    # 6. Reshape back
    return final_pulled_tensor.view(B, T)


@torch.compile(mode="reduce-overhead")
def pull_from_left(
    byte_tensor: Tensor, bytes_per_token: int, pad_byte: int, eot_byte: int
) -> Tensor:
    """
    Pulls valid bytes towards the right boundary of each token, considering EOT tokens
    as sequence breaks. Bytes are taken from the token after the previous EOT up to
    the current token. The rightmost available bytes are kept if capacity is exceeded.
    EOT tokens retain their original bytes. Vectorized implementation.
    """
    B, T = byte_tensor.size()
    if T == 0:
        return byte_tensor
    T_reduced = T // bytes_per_token
    assert T % bytes_per_token == 0, "T must be divisible by bytes_per_token"
    if T_reduced == 0:
        return torch.full_like(byte_tensor, pad_byte)

    # 1. Preprocessing (Identical to pull_from_right)
    byte_tensor_view = byte_tensor.view(B, T_reduced, bytes_per_token)
    device = byte_tensor.device

    non_pad_mask = byte_tensor_view != pad_byte
    is_eot_token = torch.all(byte_tensor_view == eot_byte, dim=2)

    valid_bytes_per_token = non_pad_mask.sum(dim=2)

    cum_valid_bytes = torch.cumsum(
        torch.cat([torch.zeros_like(valid_bytes_per_token[:, :1]), valid_bytes_per_token], dim=1),
        dim=1
    )
    # start_valid_byte_idx = cum_valid_bytes[:, :-1] # Not directly needed here
    end_valid_byte_idx = cum_valid_bytes[:, 1:]   # (B, Tr) - Global index of last valid byte + 1 for token t
    total_valid_bytes_per_batch = cum_valid_bytes[:, -1] # (B,)

    # 2. Find Previous EOT Boundary (using searchsorted loop for now)
    prev_eot_token_indices = torch.full_like(is_eot_token, -1, dtype=torch.long) # (B, Tr)
    for b in range(B):
        eot_pos = is_eot_token[b].nonzero(as_tuple=True)[0]
        if eot_pos.numel() > 0:
            token_indices = torch.arange(T_reduced, device=device)
            # Find index in eot_pos for the last EOT <= token_indices
            prev_eot_rel_idx = torch.searchsorted(eot_pos, token_indices, side='right') - 1
            # Get actual token positions for valid indices
            valid_mask = prev_eot_rel_idx >= 0
            prev_eot_token_indices[b, valid_mask] = eot_pos[prev_eot_rel_idx[valid_mask]]
            # Tokens before the first EOT correctly have -1

    # 3. Calculate Byte Ranges to Pull
    # Global index of the first valid byte *after* the previous EOT token
    # Need to handle prev_eot_token_indices == -1
    prev_eot_plus_1 = (prev_eot_token_indices + 1).clamp(min=0) # Clamp to handle -1 -> 0
    # Use gather with the corrected indexing (no extra unsqueeze)
    pull_range_start_idx = cum_valid_bytes.gather(1, prev_eot_plus_1) # (B, Tr)
    # Use mask for tokens before first EOT (where prev_eot == -1) to ensure start is 0
    pull_range_start_idx = torch.where(prev_eot_token_indices == -1, torch.tensor(0, device=device, dtype=torch.long), pull_range_start_idx)

    # Global index of the last valid byte + 1 for the *current* token `t`
    pull_range_end_idx = end_valid_byte_idx # (B, Tr)

    # Number of valid bytes available from (prev EOT + 1) up to current token `t`
    available_bytes = pull_range_end_idx - pull_range_start_idx # (B, Tr)
    available_bytes = torch.clamp(available_bytes, min=0)

    # We need to take the *rightmost* `bytes_per_token` of these available bytes
    bytes_to_use = torch.minimum(available_bytes, torch.tensor(bytes_per_token, device=device)) # (B, Tr)

    # Calculate the starting global index for the bytes we actually want to gather
    gather_start_global_idx = pull_range_end_idx - bytes_to_use # (B, Tr)

    # 4. Gather Bytes (Using Flattening Approach)
    flat_indices_b, flat_indices_t, flat_indices_k = non_pad_mask.nonzero(as_tuple=True)
    flat_valid_bytes = byte_tensor_view[flat_indices_b, flat_indices_t, flat_indices_k]

    batch_offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), total_valid_bytes_per_batch.cumsum(0)[:-1]])

    # Indices for the output tensor relative to the gather operation (0 to bytes_to_use - 1)
    k_indices_relative = torch.arange(bytes_per_token, device=device).view(1, 1, -1) # (1, 1, bpt)

    # Calculate the absolute global valid byte index we want for each relative k
    # target_global_valid_idx = gather_start_global_idx[b, t] + k_relative
    target_global_valid_idx = gather_start_global_idx.unsqueeze(2) + k_indices_relative # (B, Tr, bpt)

    # Create mask for indices we need (k_relative < bytes_to_use)
    gather_mask = k_indices_relative < bytes_to_use.unsqueeze(2) # (B, Tr, bpt)

    # Adjust target global indices for the flattened array
    absolute_gather_idx = target_global_valid_idx + batch_offsets.view(B, 1, 1) # (B, Tr, bpt)

    # Pad flat_valid_bytes and gather safely
    total_flat_size = batch_offsets[-1] + total_valid_bytes_per_batch[-1] if B > 0 else 0
    safe_indices = torch.where(gather_mask, absolute_gather_idx, total_flat_size)
    # Ensure dtype matches byte_tensor
    padded_flat_valid_bytes = torch.cat([flat_valid_bytes, torch.tensor([pad_byte], device=device, dtype=byte_tensor.dtype)])
    clamped_indices = torch.clamp(safe_indices, max=total_flat_size)
    gathered_bytes_flat = padded_flat_valid_bytes[clamped_indices] # (B, Tr, bpt) - These are the desired bytes, but left-aligned in this tensor

    # 5. Place Bytes Right-Aligned and Handle EOTs
    # Create the output tensor, initially padded
    pulled_non_eot = torch.full_like(byte_tensor_view, pad_byte)

    # Calculate the destination k index for placing the gathered bytes
    # The k_th gathered byte (0 <= k < bytes_to_use) should go to slot (bpt - bytes_to_use + k)
    dest_k_indices = bytes_per_token - bytes_to_use.unsqueeze(2) + k_indices_relative # (B, Tr, bpt)

    # We only place where gather_mask is true
    # Create full B, Tr indices for scatter/advanced indexing
    b_indices = torch.arange(B, device=device).view(B, 1, 1).expand_as(dest_k_indices)
    t_indices = torch.arange(T_reduced, device=device).view(1, T_reduced, 1).expand_as(dest_k_indices)

    # Use advanced indexing to place the bytes
    # Only update positions where gather_mask is true
    valid_dest_k = dest_k_indices[gather_mask]
    valid_b = b_indices[gather_mask]
    valid_t = t_indices[gather_mask]
    valid_gathered_bytes = gathered_bytes_flat[gather_mask]

    if valid_b.numel() > 0: # Check if there's anything to place
        pulled_non_eot[valid_b, valid_t, valid_dest_k] = valid_gathered_bytes

    # Handle EOT Tokens: Keep original bytes, exactly as they were.
    final_pulled_tensor = torch.where(
        is_eot_token.unsqueeze(-1),
        byte_tensor_view, # Keep original bytes exactly as they were for EOTs
        pulled_non_eot      # Use the pulled bytes for non-EOTs
    )

    # 6. Reshape back
    return final_pulled_tensor.view(B, T)


@torch.no_grad()
class TokensToBytes:
    def __init__(
            self,
            byte_params: ByteHyperparameters,
            vocab_size: int = 50257,
            device: torch.device = "cpu",
    ):
        self.byte_params = byte_params
        bpt = byte_params.bytes_per_token
        ttb_left_pad = make_embedding(f"ttb_{bpt}_left_pad.json", vocab_size).to(device) if byte_params.byte_mixin_method != "noop" and byte_params.padding_in == "left" else None
        ttb_right_pad = make_embedding(f"ttb_{bpt}_right_pad.json", vocab_size).to(device) if byte_params.byte_mixin_method != "noop" and byte_params.padding_in == "right" else None

        ttb_in = ttb_left_pad if byte_params.padding_in == "left" else ttb_right_pad
        ttb_in = ttb_in.to(device) if ttb_in else None

        pull_kwargs = dict(bytes_per_token=bpt, pad_byte=456, eot_byte=457)
        pull_in = functools.partial(pull_from_left, **pull_kwargs) if byte_params.padding_in == "left" else functools.partial(pull_from_right, **pull_kwargs)

        bpt = byte_params.bytes_per_token
        def _create_data_from_toks_TT(toks: Tensor):
            toks = toks.unsqueeze(0) if toks.ndim == 1 else toks
            assert toks.ndim == 2, f"{toks.shape=}"
            bytes_padded_in = tokens_to_bytes(toks, ttb_in)
            bytes_pulled_in = pull_in(bytes_padded_in)
            return toks, bytes_padded_in, bytes_pulled_in

        def _create_data_from_toks_TF(toks: Tensor):
            toks = toks.unsqueeze(0) if toks.ndim == 1 else toks
            assert toks.ndim == 2, f"{toks.shape=}"
            bytes_padded_in = tokens_to_bytes(toks, ttb_in)
            bytes_pulled_in = None
            return toks, bytes_padded_in, bytes_pulled_in

        def _create_data_from_toks_FF(toks: Tensor):
            toks = toks.unsqueeze(0) if toks.ndim == 1 else toks
            assert toks.ndim == 2, f"{toks.shape=}"
            bytes_padded_in = None
            bytes_pulled_in = None
            return toks, bytes_padded_in, bytes_pulled_in

        self.create_data_from_toks = {
            # (byte_in, pull_in, byte_out, pull_out): function
            (True, True): _create_data_from_toks_TT,
            (True, False): _create_data_from_toks_TF,
            (False, False): _create_data_from_toks_FF,
        }[(byte_params.byte_mixin_method != "noop", byte_params.pull_in)]
        self.device = device
        self._int_to_bytes = None

    @torch.no_grad()
    def __call__(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.create_data_from_toks(tokens.to(self.device))
    
    @property
    def int_to_bytes(self):
        if self._int_to_bytes is not None:
            return self._int_to_bytes
        
        with open("embeddings/int_to_byte.json", "r") as f:
            self._int_to_bytes = json.loads(f.read())
        return self._int_to_bytes
    
    @torch.no_grad()
    def bytes_to_string(self, bytes: Tensor) -> str:
        assert self.byte_params.byte_mixout_method != "noop"
        bytes = bytes.squeeze()
        assert bytes.ndim == 1
        bytes = bytes.tolist()
        return "".join([self.int_to_bytes[b] for b in bytes])

# -----------------------------------------------------------------------------
# Download model


# Thanks Google Gemini
def parse_name_to_hyperparams(name_str: str) -> tuple[ByteHyperparameters, ModelDims]:
    bh = ByteHyperparameters()
    md = ModelDims()

    original_name_str = name_str # for error messages

    if name_str.startswith("snimu/"):
        name_str = name_str[len("snimu/"):]

    if not name_str.startswith("MoT_"):
        raise ValueError(f"Name string '{original_name_str}' must start with 'MoT_' after optional 'snimu/' prefix")
    
    name_str = name_str[len("MoT_"):]
    
    # The last two parts are niter and seed, which we don't need for these dataclasses.
    # However, BTMdim can have hyphens, so we can't just split by '-' everywhere.
    # We split by '_' first.
    parts = name_str.split('_')

    # Iterate through parts to find relevant segments
    # Note: Some segments in make_name are conditional (e.g., _add, _mix-in, _mix-out, _nlo)
    # If they are not present, the default values in ByteHyperparameters/ModelDims will be used.

    for part in parts:
        if part.startswith("pad-"):
            val = part[len("pad-"):]
            if len(val) == 2:
                bh.padding_in = "left" if val[0] == 'l' else "right"
                bh.padding_out = "left" if val[1] == 'l' else "right"
            else:
                print(f"Warning: Malformed pad segment: {part}")
        elif part.startswith("pull-"):
            val = part[len("pull-"):]
            if len(val) == 2:
                bh.pull_in = True if val[0] == 'y' else False
                bh.pull_out = True if val[1] == 'y' else False
            else:
                print(f"Warning: Malformed pull segment: {part}")
        elif part == "add":
            bh.add_padded_and_pulled = True
        elif part == "mix-in":
            # According to make_name: "_mix-in" if args.mix_bytes_within_tok_in and args.use_byte_self_attn
            # So if "mix-in" is present, both are true.
            bh.mix_bytes_within_tok_in = True
            bh.use_byte_self_attn = True
        elif part == "mix-out":
            bh.mix_bytes_within_tok_out = True
        elif part.startswith("bpt-"):
            try:
                bh.bytes_per_token = int(part[len("bpt-"):])
            except ValueError:
                print(f"Warning: Malformed bpt segment: {part}")
        elif part.startswith("how-"):
            # Format: how-{byte_mixin_method}-{byte_mixout_method}
            # Methods themselves can't contain hyphens based on Literal types
            methods_str = part[len("how-"):]
            method_parts = methods_str.split('-', 1) # Split only on the first hyphen
            if len(method_parts) == 2:
                bh.byte_mixin_method = method_parts[0] # type: ignore
                bh.byte_mixout_method = method_parts[1] # type: ignore
            else:
                print(f"Warning: Malformed how segment: {part}")
        elif part.startswith("nlo-"):
            # This part is conditional on byte_mixout_method != "noop" in make_name
            try:
                bh.n_layer_out = int(part[len("nlo-"):])
            except ValueError:
                print(f"Warning: Malformed nlo segment: {part}")
        elif part.startswith("BTMdim-"):
            # Format: BTMdim-{byte_dim}-{token_dim}-{model_dim}
            dims_str = part[len("BTMdim-"):]
            dim_values = dims_str.split('-')
            if len(dim_values) == 3:
                try:
                    md.byte_dim = int(dim_values[0])
                    md.token_dim = int(dim_values[1])
                    md.model_dim = int(dim_values[2])
                except ValueError:
                    print(f"Warning: Malformed BTMdim segment values: {part}")
            else:
                print(f"Warning: Malformed BTMdim segment structure: {part}")
        # Ignored parts: niter-..., and the seed (which is just a number)
        elif part.startswith("niter-"):
            pass # Not needed for ByteHyperparameters or ModelDims
        else:
            # Could be the seed, or an unknown part
            try:
                int(part) # Check if it's a number (likely seed)
            except ValueError:
                if part: # Avoid warning for empty strings if name ends with _
                    print(f"Info: Unrecognized or ignored segment: {part}")
                    
    return bh, md


def load_model(name: str) -> GPT:
    bh, md = parse_name_to_hyperparams(name)
    path = hf_hub_download(
        repo_id=name,
        filename="model.safetensors",
        token=os.getenv("HF_TOKEN"),
    )
    model_dict = safetensors.torch.load_file(path, device="cpu")
    model_dict = {k.replace("_orig_mod.", ""): v for k, v in model_dict.items()}
    model = GPT(
        vocab_size=50257,
        num_layers=16,
        num_heads=8,
        max_seq_len=4096,  # else I get an error in Rotary
        model_dims=md,
        byte_params=bh,
    )
    model.load_state_dict(model_dict)
    return model


# ------------------------------------------------------------------------------
# Sampling functions


class SamplerTokens:
    def __init__(self, temperature: float = 0.0, eot_token_id: int = 50256, eps: float = 1e-4):
        assert temperature >= 0.0
        self.temperature = temperature
        self.eot_token_id = eot_token_id
        self.eps = eps

    def sample_argmax(self, logits: Tensor) -> Tensor:
        return torch.argmax(logits, dim=-1)
    
    def sample_temperature(self, logits: Tensor) -> Tensor:
        return torch.multinomial(F.softmax(logits / self.temperature, dim=-1), num_samples=1)

    def __call__(self, logits: Tensor) -> Tensor:
        logits[..., self.eot_token_id] = -float("inf")
        return self.sample_argmax(logits) if self.temperature < self.eps else self.sample_temperature(logits)


class SamplerBytes:
    def __init__(
            self,
            n: int,
            temperature: float = 0.0,
            pad_byte: int = 456,
            eot_byte: int = 457,
            bpt: int = 16,
            eps: float = 1e-4,
    ):
        assert temperature >= 0.0
        self.temperature = temperature
        self.n = n
        self.pad_byte = pad_byte
        self.eot_byte = eot_byte
        self.bpt = bpt
        self.eps = eps

    def sample_argmax(self, logits: Tensor) -> Tensor:
        return torch.argmax(logits, dim=-1)[:, :self.n]

    def sample_temperature(self, logits: Tensor) -> Tensor:
        """
        I have an LLM that predicts bpt bytes at once, and I would like to sample it with temperature in a sensible manner.
        For this, I will assume that the most likely byte at position n is connected to the most likely byte in position n+1;
        the same for the second most likely one, and so on.
        In other words, I assume that the "probability bands" through the bytes
        represent all the legitimate paths through the byte probabilities.

        So to sample them, I first need to identify the ordering of the bytes at each position.
        Then, I need to find some probability of each byte sequence so that I can sample them.
        So to determine the probability of sampling the nth most likely byte at each position,
        I take the sum over the probability of the nth most likely byte at each position, over all positions.
        Then I divide that number by the temperature, and take the softmax.
        I sample from the resulting distribution to get the index I want;
        but this isn't a byte index, it's a probability index!
        In my last step, I take the index--let's say it's 3--and identify the 3rd most likely byte index at each position.
        Then, I return that tensor.

        This function has (obviously) been aided by Gemini; I'm keeping (some of) the comments.
        """
        sample_logits = logits.clone()[:, :self.n, :]

        if self.n == 1:
            squeezed_logits = sample_logits.squeeze(1) # Shape: [batch_size, vocab_size]
            probs = F.softmax(squeezed_logits / self.temperature, dim=-1)
            sampled_indices = torch.multinomial(probs, num_samples=1) # Shape: [batch_size, 1]
            sampled_indices = einops.rearrange(sampled_indices, "B I -> B 1 I")  # keep the byte seq dim
            return sampled_indices
        else:
            # Implement the "probability bands" logic for self.n > 1 positions

            # 1. Get original probabilities (before temperature scaling) at each of the self.n positions
            # probs_at_positions shape: [batch_size, self.n, vocab_size]
            probs_at_positions = F.softmax(sample_logits, dim=-1)

            # 2. For each position, sort bytes by their original probabilities
            # sorted_original_probs[b, pos, k]: probability of the k-th most likely byte at (b, pos)
            # sorted_byte_indices[b, pos, k]: index (byte value) of the k-th most likely byte at (b, pos)
            # Both shapes: [batch_size, self.n, vocab_size]
            sorted_original_probs, sorted_byte_indices = torch.sort(probs_at_positions, dim=-1, descending=True)

            # 3. Calculate a score for each "probability band" (rank k)
            # The score for the k-th band is the sum of probabilities of the k-th most likely bytes
            # across all self.n positions.
            # path_scores[b, k] = sum_{pos=0}^{self.n-1} sorted_original_probs[b, pos, k]
            # Summing across dim=1 (the 'self.n' dimension, which is the position dimension here)
            path_scores = torch.sum(sorted_original_probs, dim=1)
            # path_scores shape: [batch_size, vocab_size]

            # 4. Apply temperature to these path scores and compute sampling probabilities for bands
            # If temperature is very low, this will strongly favor bands with high summed original probabilities.
            # If temperature is high, band selection becomes more random.
            path_distribution_logits = path_scores / self.temperature
            path_choice_probs = F.softmax(path_distribution_logits, dim=-1)
            # path_choice_probs shape: [batch_size, vocab_size]

            # 5. Sample a "probability band" index (a rank k) for each item in the batch
            # sampled_path_rank[b, 0] will be the chosen rank k for batch item b (e.g., 0 for most likely band, 1 for second, etc.)
            sampled_path_rank = torch.multinomial(path_choice_probs, num_samples=1)
            # sampled_path_rank shape: [batch_size, 1]

            # 6. Construct the output sequence using the chosen rank
            # For each batch item b and position pos, we need the byte corresponding to
            # the sampled_path_rank[b,0]-th most likely byte.
            # This byte is sorted_byte_indices[b, pos, sampled_path_rank[b,0]].
            # We use torch.gather for this.
            # `sampled_path_rank` needs to be expanded to match the dimensions for gather.
            # Target shape for index in gather: [batch_size, self.n, 1] (to pick one from vocab_size dim of sorted_byte_indices)
            expanded_rank_for_gather = sampled_path_rank.unsqueeze(1).expand(-1, self.n, -1)
            # expanded_rank_for_gather shape: [batch_size, self.n, 1]

            output_bytes = torch.gather(sorted_byte_indices, dim=2, index=expanded_rank_for_gather)
            # output_bytes shape: [batch_size, self.n, 1]

            # Remove the last dimension
            output_bytes = output_bytes.squeeze(-1)
            # output_bytes shape: [batch_size, self.n]

            return output_bytes

    def __call__(self, logits: Tensor) -> Tensor:
        assert logits.ndim == 3
        assert logits.shape[1] == self.bpt
        logits[..., self.pad_byte] = -float("inf")
        logits[..., self.eot_byte] = -float("inf")
        return self.sample_argmax(logits) if self.temperature < self.eps else self.sample_temperature(logits)


@torch.inference_mode()
def generate_until__tokens_out(model: GPT, ttb: TokensToBytes, requests: list[Instance], sampler: SamplerTokens) -> list[str]:
    enc = tiktoken.encoding_for_model("gpt-2")
    texts = []
    for request in requests:
        query = request.args[0]
        max_toks = request.args[1].get("max_gen_toks", 1024)
        until = request.args[1].get("until", None)

        toks, bytes_padded_in, bytes_pulled_in = ttb(torch.tensor(enc.encode(query), device="cuda"))
        text = query
        for i in range(max_toks):
            logits = model(toks, bytes_padded_in, bytes_pulled_in).squeeze()[-1]
            toks = torch.cat([toks, sampler(logits).view(1, 1)], dim=1)
            text = enc.decode(toks.squeeze().tolist())
            toks, bytes_padded_in, bytes_pulled_in = ttb(torch.tensor(enc.encode(text), device="cuda"))
            if until and any(stop in text for stop in until):
                break
        texts.append(text)
    return texts


@torch.inference_mode()
def loglikelihood__tokens_out(model: GPT, ttb: TokensToBytes, requests: list[Instance], sampler: SamplerTokens) -> list[tuple[float, bool]]:
    enc = tiktoken.encoding_for_model("gpt-2")
    results = []
    for request in requests:
        query = request.args[0]
        targets = request.args[1]

        len_in = len(enc.encode(query))
        
        toks_in, bytes_padded_in, bytes_pulled_in = ttb(torch.tensor(enc.encode(query), device="cuda"))
        toks_out, bytes_padded_out, bytes_pulled_out = ttb(torch.tensor(enc.encode(targets), device="cuda"))
        toks = torch.cat([toks_in, toks_out], dim=-1)
        bytes_padded = torch.cat([bytes_padded_in, bytes_padded_out], dim=-1) if bytes_padded_in is not None else None
        bytes_pulled = torch.cat([bytes_pulled_in, bytes_pulled_out], dim=-1) if bytes_pulled_in is not None else None

        logits: Tensor = model(toks, bytes_padded, bytes_pulled)
        logits = logits.squeeze()[len_in-1:-1]  # teacher-forced predictions of targets
        is_greedy = torch.all(logits.argmax(dim=-1) == torch.tensor(enc.encode(targets), device="cuda"))
        lls = F.log_softmax(logits, dim=-1).gather(1, torch.tensor(enc.encode(targets), device="cuda").unsqueeze(0)).sum().item()
        results.append((lls, is_greedy))
    return results


@torch.inference_mode()
def loglikelihood_rolling__tokens_out(model: GPT, ttb: TokensToBytes, requests: list[Instance], sampler: SamplerTokens) -> list[tuple[float, bool]]:
    enc = tiktoken.encoding_for_model("gpt-2")
    results = []
    for request in requests:
        query = request.args[0]
        targets = request.args[1]

        len_out = len(enc.encode(targets))

        text = query
        loglikelihood = 0.0
        
        for idx in range(len_out):
            toks, bytes_padded, bytes_pulled = ttb(torch.tensor(enc.encode(text), device="cuda"))
            logits: Tensor = model(toks, bytes_padded, bytes_pulled)
            logits = logits.squeeze()[-1]
            lls = F.log_softmax(logits, dim=-1)
            target = torch.tensor(enc.encode(targets)[idx], device="cuda")
            loglikelihood += lls[target].item()
            next_token = sampler(logits)
            text += enc.decode([next_token])
        results.append((loglikelihood,))
    return results


@torch.inference_mode()
def generate_until__bytes_out(model: GPT, ttb: TokensToBytes, requests: list[Instance], sampler: SamplerBytes) -> list[str]:
    enc = tiktoken.encoding_for_model("gpt-2")
    texts = []
    for request in requests:
        query = request.args[0]
        max_toks = request.args[1].get("max_gen_toks", 1024 * 7 // sampler.n)  # ~7 bpt on avg
        until = request.args[1].get("until", None)

        toks, bytes_padded_in, bytes_pulled_in = ttb(torch.tensor(enc.encode(query), device="cuda"))
        text = query
        for i in range(max_toks):
            logits = model(toks, bytes_padded_in, bytes_pulled_in).squeeze()[-sampler.bpt:]
            bytes = sampler(logits)
            text += ttb.bytes_to_string(bytes)
            toks, bytes_padded_in, bytes_pulled_in = ttb(torch.tensor(enc.encode(text), device="cuda"))
            if until and any(stop in text for stop in until):
                break
        texts.append(text)
    return texts


@torch.inference_mode()
def loglikelihood__bytes_out(model: GPT, ttb: TokensToBytes, requests: list[Instance], sampler: SamplerBytes) -> list[tuple[float, bool]]:
    enc = tiktoken.encoding_for_model("gpt-2")
    results = []
    for request in requests:
        query = request.args[0]
        targets = request.args[1]

        len_in = len(enc.encode(query))
        
        toks_in, bytes_padded_in, bytes_pulled_in = ttb(torch.tensor(enc.encode(query), device="cuda"))
        toks_out, bytes_padded_out, bytes_pulled_out = ttb(torch.tensor(enc.encode(targets), device="cuda"))
        toks = torch.cat([toks_in, toks_out], dim=-1)
        bytes_padded = torch.cat([bytes_padded_in, bytes_padded_out], dim=-1) if bytes_padded_in is not None else None
        bytes_pulled = torch.cat([bytes_pulled_in, bytes_pulled_out], dim=-1) if bytes_pulled_in is not None else None

        logits: Tensor = model(toks, bytes_padded, bytes_pulled)
        logits = logits.squeeze()[(len_in-1)*sampler.bpt:-sampler.bpt]  # teacher-forced predictions of targets
        is_greedy = torch.all(logits.argmax(dim=-1) == torch.tensor(enc.encode(targets), device="cuda"))
        lls = F.log_softmax(logits, dim=-1).gather(1, torch.tensor(enc.encode(targets), device="cuda").unsqueeze(0)).sum().item()
        results.append((lls, is_greedy))
    return results


@torch.inference_mode()
def loglikelihood_rolling__bytes_out(model: GPT, ttb: TokensToBytes, requests: list[Instance], sampler: SamplerBytes) -> list[tuple[float, bool]]:
    enc = tiktoken.encoding_for_model("gpt-2")
    results = []
    for request in requests:
        query = request.args[0]
        targets = request.args[1]

        len_out = len(enc.encode(targets))

        text = query
        loglikelihood = 0.0
        
        for idx in range(len_out):  # TODO
            toks, bytes_padded, bytes_pulled = ttb(torch.tensor(enc.encode(text), device="cuda"))
            logits: Tensor = model(toks, bytes_padded, bytes_pulled)
            logits = logits.squeeze()[-sampler.bpt:]
            lls = F.log_softmax(logits, dim=-1)
            target = torch.tensor(enc.encode(targets)[idx], device="cuda")
            loglikelihood += lls[target].item()
            next_token = sampler(logits)
            text += enc.decode([next_token])
        results.append((loglikelihood,))
    return results


# -----------------------------------------------------------------------------
# MoT model


class MoTModel(LM):
    def __init__(self, name: str, temperature: float = 0.0, n: int = 1, **kwargs) -> None:
        assert os.getenv("HF_TOKEN") is not None, "Please set the HF_TOKEN environment variable."
        super().__init__()
        torch.set_float32_matmul_precision('high')
        self.name = name
        bh, _ = parse_name_to_hyperparams(name)
        self.model = torch.compile(load_model(name).cuda()).eval()
        self.sampler = SamplerTokens(temperature) if bh.byte_mixout_method == "noop" else SamplerBytes(n, temperature, bpt=bh.bytes_per_token)
        self.ttb = TokensToBytes(bh, device="cuda")
        self.toks_out = bh.byte_mixout_method == "noop"

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        if self.toks_out:
            return loglikelihood__tokens_out(self.model, self.ttb, requests, self.sampler)
        else:
            return loglikelihood__bytes_out(self.model, self.ttb, requests, self.sampler)


    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        if self.toks_out:
            return loglikelihood_rolling__tokens_out(self.model, self.ttb, requests, self.sampler)
        else:
            return loglikelihood_rolling__bytes_out(self.model, self.ttb, requests, self.sampler)


    def generate_until(self, requests: list[Instance]) -> list[str]:
        if self.toks_out:
            return generate_until__tokens_out(self.model, self.ttb, requests, self.sampler)
        else:
            return generate_until__bytes_out(self.model, self.ttb, requests, self.sampler)


if "MoT" not in MODEL_REGISTRY:  # doing this as a decorator caused issues in the past.
    register_model("MoT")(MoTModel)
