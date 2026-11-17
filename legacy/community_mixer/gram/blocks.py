"""Transformer primitives: RoPE, attention, SwiGLU, composite blocks.

Conventions from paper §B.1:
- Backbone is a stack of [Attention + SwiGLU] layers (or [SwiGLU + SwiGLU] for the Sudoku-specific variant; not used here for N-Queens).
- Positional encoding is RoPE [Su et al., RoFormer 2024].
- Hidden size D = 512, n_heads = 8, FFN hidden D_h = 512.

References:
- RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", Neurocomputing 2024.
- SwiGLU: Shazeer, "GLU Variants Improve Transformer", arXiv:2002.05202, 2020.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# RoPE — rotary positional embedding
# ----------------------------------------------------------------------------


class RotaryPositionalEmbedding(nn.Module):
    """RoPE applied to query and key tensors.

    Builds `cos` and `sin` tables of shape (max_seq_len, head_dim) at init,
    extends on demand. The actual rotation happens in `apply_rope`.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.base = base
        self._max_seq_len = max_seq_len
        cos, sin = self._build_table(max_seq_len)
        # fp32 master table; cast-on-demand copies are cached by _cache.
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # _cache: (device.type, device.index, dtype) → (cos_view, sin_view) at full max_seq_len.
        # Avoids the .to(device=q.device, dtype=q.dtype) per attention call inside the
        # hot path (was ~480 such casts per optimizer step at our config).
        self._cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def _build_table(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq, head_dim/2)
        # Interleave so that consecutive dims (2k, 2k+1) share an angle.
        emb = torch.repeat_interleave(freqs, 2, dim=-1)  # (seq, head_dim)
        return emb.cos(), emb.sin()

    def _maybe_extend(self, seq_len: int) -> None:
        if seq_len <= self._max_seq_len:
            return
        cos, sin = self._build_table(seq_len)
        # Update fp32 master; invalidate cache so we recast next time.
        self.cos = cos.to(self.cos.device)
        self.sin = sin.to(self.sin.device)
        self._max_seq_len = seq_len
        self._cache.clear()

    def _get_cos_sin(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device.type, device.index, dtype)
        cached = self._cache.get(key)
        if cached is not None and cached[0].shape[0] >= self._max_seq_len:
            return cached
        cos = self.cos.to(device=device, dtype=dtype)
        sin = self.sin.to(device=device, dtype=dtype)
        self._cache[key] = (cos, sin)
        return cos, sin

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate q and k. Shapes: (B, n_heads, L, head_dim)."""
        L = q.shape[-2]
        self._maybe_extend(L)
        full_cos, full_sin = self._get_cos_sin(q.device, q.dtype)
        cos = full_cos[:L].view(1, 1, L, -1)
        sin = full_sin[:L].view(1, 1, L, -1)
        return _rotate(q, cos, sin), _rotate(k, cos, sin)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply the half-shift rotation used by RoPE."""
    # Split pairs (2k, 2k+1), swap and negate one component, recombine.
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    # rotated = [x1*cos_pair - x2*sin_pair, x1*sin_pair + x2*cos_pair]
    cos_pair = cos[..., 0::2]
    sin_pair = sin[..., 0::2]
    rot1 = x1 * cos_pair - x2 * sin_pair
    rot2 = x1 * sin_pair + x2 * cos_pair
    out = torch.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


# ----------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    """Bidirectional MHSA with RoPE applied to q and k.

    Uses F.scaled_dot_product_attention for the inner kernel (PyTorch ≥ 2.0).
    """

    def __init__(self, d_model: int, n_heads: int, rope: RotaryPositionalEmbedding):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)."""
        B, L, D = x.shape
        qkv = self.qkv(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        # Reshape to (B, n_heads, L, head_dim)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope.apply_rope(q, k)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)  # (B, nH, L, hd)
        attn = attn.transpose(1, 2).contiguous().view(B, L, D)
        return self.out(attn)


# ----------------------------------------------------------------------------
# SwiGLU FFN
# ----------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """Gated FFN: SwiGLU(x) = (xW_gate) ⊙ silu(xW_value), projected back to d_model.

    Paper §B.1 uses SwiGLU with d_model = D and FFN hidden = D_h. Convention here:
    `d_hidden` is the inner width before the down-projection.
    """

    def __init__(self, d_model: int, d_hidden: int, bias: bool = False):
        super().__init__()
        # One Linear that emits (gate, value) split; matches common implementations.
        self.w_in = nn.Linear(d_model, 2 * d_hidden, bias=bias)
        self.w_out = nn.Linear(d_hidden, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(gate) * val)


# ----------------------------------------------------------------------------
# RMSNorm (standard for modern Transformers; HRM/TRM use it too)
# ----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


# ----------------------------------------------------------------------------
# Composite block: [Attention + SwiGLU]
# ----------------------------------------------------------------------------


class AttnSwiGLULayer(nn.Module):
    """One [pre-norm Attention + pre-norm SwiGLU] residual layer.

    Pre-norm is the modern default (LLaMA/PaLM/HRM/TRM style) and is what we
    assume here in the absence of an explicit norm-placement statement in the
    paper.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        if residual_scale <= 0.0:
            raise ValueError(f"residual_scale must be positive, got {residual_scale}")
        self.residual_scale = float(residual_scale)
        self.norm_attn = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, rope)
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_scale * self.attn(self.norm_attn(x))
        x = x + self.residual_scale * self.ffn(self.norm_ffn(x))
        return x


class AttnSwiGLUStack(nn.Module):
    """Stack of `n_layers` AttnSwiGLU layers, optionally with a final RMSNorm.

    Used as both `fL` (low-level) and `fH` (high-level) modules in the GRAM
    hierarchical instantiation.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding,
        final_norm: bool = True,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.layers = nn.ModuleList(
            [
                AttnSwiGLULayer(
                    d_model, n_heads, d_ff, rope, residual_scale=residual_scale
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model) if final_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# ----------------------------------------------------------------------------
# Small SwiGLU MLP head — for guidance / posterior / LPRM / halt
# ----------------------------------------------------------------------------


class SwiGLUMLP(nn.Module):
    """Token-wise SwiGLU MLP: D → d_hidden → d_out.

    Paper §B.1 specifies that μ_θ, σ_θ, μ_φ, σ_φ are each a SwiGLU MLP. We
    interpret that as a SwiGLU FFN followed by a linear projection to the
    target dimension. For μ/σ heads on the high-level state, d_out = D.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, bias: bool = False):
        super().__init__()
        self.norm = RMSNorm(d_in)
        self.gate_value = nn.Linear(d_in, 2 * d_hidden, bias=bias)
        self.proj = nn.Linear(d_hidden, d_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        gate, val = self.gate_value(x).chunk(2, dim=-1)
        return self.proj(F.silu(gate) * val)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def init_truncated_normal_(weight: torch.Tensor, std: float = 0.02) -> None:
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2 * std, b=2 * std)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
