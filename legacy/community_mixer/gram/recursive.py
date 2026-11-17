"""Recursive core (Stage C) — one transition and one supervision step.

Paper §2.1 hierarchical instantiation, equations (6)–(9):

    For one transition z_{t-1} → z_t (with z = (h, l)):
        l_{t,0} = l_{t-1}
        l_{t,k} = fL(h_{t-1}, l_{t,k-1}, e_x)        for k = 1, …, K
        u_t     = fH(h_{t-1}, l_{t,K})
        ε_t     ~ N(μ_θ(u_t), σ_θ²(u_t) I)            (prior, inference)
        ε_t     ~ N(μ_φ(u_t, y), σ_φ²(u_t, y) I)      (posterior, training)
        h_t     = u_t + ε_t

A supervision step = T transitions. Gradient propagation is truncated to the
final transition of each supervision step (paper §A.3, §2.2 practical
implementation). This module owns the per-supervision-step orchestration.

Convention for "fL takes (h, l, ex)" — the paper does not specify the exact
injection mechanism; HRM/TRM upstream code uses additive injection at the
input of each transformer stack. We follow the same convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .blocks import AttnSwiGLUStack, RotaryPositionalEmbedding
from .heads import GuidancePriorHead, PosteriorHead, reparameterize


GUIDANCE_MODES = (
    "full",
    "no_sg",
    "stochasticity_only",
    "guide_only",
)


def normalize_guidance_mode(mode: str) -> str:
    """Validate the repo-matched stochastic-guidance ablation mode."""
    normalized = str(mode).strip().lower()
    if normalized not in GUIDANCE_MODES:
        raise ValueError(
            f"unsupported guidance_mode={mode!r}; expected one of {GUIDANCE_MODES}"
        )
    return normalized


def _scaled_reparameterize(
    mu: torch.Tensor,
    var: torch.Tensor,
    noise_scale: float,
) -> torch.Tensor:
    """Sample a diagonal Gaussian, with scale=0 returning the exact mean."""
    if noise_scale < 0.0:
        raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")
    if noise_scale == 0.0:
        return mu
    return reparameterize(mu, var * (noise_scale**2))


def access_aware_residual_scale(
    parameter_accesses: int,
    exponent: float,
) -> float:
    """DeepLoop-style residual scale based on shared-parameter access count."""
    if parameter_accesses < 1:
        raise ValueError(
            f"parameter_accesses must be positive, got {parameter_accesses}"
        )
    if not math.isfinite(exponent) or exponent < 0.0:
        raise ValueError(f"exponent must be finite and non-negative, got {exponent}")
    return float(parameter_accesses) ** (-float(exponent))


@dataclass
class TransitionInfo:
    """Diagnostics returned per transition; used by the loss + by ablation logging."""

    u_t: torch.Tensor              # (B, L, D) — deterministic high-level update
    eps: torch.Tensor              # (B, L, D) — sampled stochastic guidance
    mu_p: torch.Tensor             # (B, L, D) — prior mean
    var_p: torch.Tensor            # (B, L, D) — prior σ²
    mu_q: torch.Tensor | None      # (B, L, D) — posterior mean (None at inference)
    var_q: torch.Tensor | None     # (B, L, D) — posterior σ² (None at inference)
    kl_enabled: bool = True        # False for deterministic repo-matched arms


class RecursiveCore(nn.Module):
    """One full GRAM recursive computation: T transitions, each with K low-level
    updates + 1 stochastic high-level update.

    Initial state z_0 = (h_0, l_0): one fixed N(0, I) sample, saved as a buffer
    and not optimized — paper §B.1 "sample once from N(0,I), save in checkpoint,
    load again, fixed value".
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        seq_len: int,
        fL_layers: int = 2,
        fH_layers: int = 2,
        K_inner: int = 4,
        T_outer: int = 3,
        guidance_hidden_mult: int = 1,
        rope: RotaryPositionalEmbedding | None = None,
        z0_seed: int = 1234,
        residual_scale_exponent: float = 0.0,
        residual_scale_nsup: int = 1,
        guidance_mode: str = "full",
    ):
        super().__init__()
        self.d_model = d_model
        self.K = K_inner
        self.T = T_outer
        self.guidance_mode = normalize_guidance_mode(guidance_mode)
        self.kl_enabled = self.guidance_mode in {"full", "stochasticity_only"}
        self.residual_scale_exponent = float(residual_scale_exponent)
        self.residual_scale_nsup = int(residual_scale_nsup)
        self.fL_parameter_accesses = self.K * self.T * self.residual_scale_nsup
        self.fH_parameter_accesses = self.T * self.residual_scale_nsup
        self.fL_residual_scale = access_aware_residual_scale(
            self.fL_parameter_accesses, self.residual_scale_exponent
        )
        self.fH_residual_scale = access_aware_residual_scale(
            self.fH_parameter_accesses, self.residual_scale_exponent
        )
        self.rope = rope if rope is not None else RotaryPositionalEmbedding(head_dim=d_model // n_heads)
        self.fL = AttnSwiGLUStack(
            fL_layers, d_model, n_heads, d_ff, self.rope,
            residual_scale=self.fL_residual_scale,
        )
        self.fH = AttnSwiGLUStack(
            fH_layers, d_model, n_heads, d_ff, self.rope,
            residual_scale=self.fH_residual_scale,
        )
        self.prior = GuidancePriorHead(d_model, guidance_hidden_mult * d_model)
        self.posterior = PosteriorHead(d_model, guidance_hidden_mult * d_model)

        # Fixed initial state z_0 = (h_0, l_0), one Gaussian sample, frozen.
        # Paper §B.1: "sample once from N(0, I), then save the value within the
        # network checkpoint and load it again". We use a deterministic seed so
        # the buffer is reproducible across checkpoints.
        g = torch.Generator().manual_seed(z0_seed)
        h0 = torch.randn(seq_len, d_model, generator=g)
        l0 = torch.randn(seq_len, d_model, generator=g)
        self.register_buffer("h0", h0, persistent=True)
        self.register_buffer("l0", l0, persistent=True)

    def initial_state(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand the fixed (seq_len, d_model) initial states to (B, seq_len, d_model)."""
        h = self.h0.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        l = self.l0.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        return h, l

    def _training_guidance(
        self,
        u_t: torch.Tensor,
        y_pooled: torch.Tensor,
        noise_scale: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return repo-matched training guidance.

        This is deliberately not presented as a paper-faithful Table 3b
        implementation.  The deterministic arms use the prior path during
        both training and inference and disable KL.  The stochasticity-only
        arm retains the variational variance path while forcing both means
        to exact zero.
        """
        if self.guidance_mode == "no_sg":
            zero = torch.zeros_like(u_t)
            one = torch.ones_like(u_t)
            return zero, zero, one, zero, one

        mu_p_raw, var_p = self.prior(u_t)
        if self.guidance_mode == "guide_only":
            # Train/inference alignment: deterministic prior mean, no
            # posterior or degenerate-Gaussian KL.
            return mu_p_raw, mu_p_raw, var_p, mu_p_raw, var_p

        mu_q_raw, var_q = self.posterior(u_t, y_pooled)
        if self.guidance_mode == "stochasticity_only":
            mu_p = torch.zeros_like(mu_p_raw)
            mu_q = torch.zeros_like(mu_q_raw)
        else:
            mu_p = mu_p_raw
            mu_q = mu_q_raw
        eps = _scaled_reparameterize(mu_q, var_q, noise_scale)
        return eps, mu_p, var_p, mu_q, var_q

    def _inference_guidance(
        self,
        u_t: torch.Tensor,
        noise_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return repo-matched inference guidance and prior diagnostics."""
        if self.guidance_mode == "no_sg":
            zero = torch.zeros_like(u_t)
            return zero, zero, torch.ones_like(u_t)

        mu_p_raw, var_p = self.prior(u_t)
        if self.guidance_mode == "guide_only":
            return mu_p_raw, mu_p_raw, var_p
        if self.guidance_mode == "stochasticity_only":
            mu_p = torch.zeros_like(mu_p_raw)
        else:
            mu_p = mu_p_raw
        eps = _scaled_reparameterize(mu_p, var_p, noise_scale)
        return eps, mu_p, var_p

    def one_transition_train(
        self,
        h_prev: torch.Tensor,
        l_prev: torch.Tensor,
        ex: torch.Tensor,
        y_pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training-only one-transition. Compile-friendly:

          * No optional args (y_pooled is always a Tensor in train mode).
          * No dataclass return — flat tuple.
          * No `is_autocast_enabled()` Python branch — caller pre-casts the
            inputs to the autocast dtype before invoking us, so all residual
            adds inside stay in the right dtype.

        Returns (h_new, l_k, u_t, eps, mu_p, var_p, mu_q, var_q). Designed for
        torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False).
        """
        # K low-level refinements with h held fixed.
        l_k = l_prev
        for _ in range(self.K):
            l_k = self.fL(l_k + h_prev + ex)
        # Deterministic high-level update.
        u_t = self.fH(h_prev + l_k)
        # Repo-matched guidance mode.  The mode is a constructor-time
        # constant, so torch.compile specializes this hot path per arm.
        eps, mu_p, var_p, mu_q, var_q = self._training_guidance(
            u_t, y_pooled
        )
        h_new = u_t + eps
        return h_new, l_k, u_t, eps, mu_p, var_p, mu_q, var_q

    def one_transition(
        self,
        h_prev: torch.Tensor,
        l_prev: torch.Tensor,
        ex: torch.Tensor,
        y_pooled: torch.Tensor | None = None,
        prior_noise_scale: float = 1.0,
        posterior_noise_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, TransitionInfo]:
        """Run one transition: K low-level updates + high-level + stochastic guidance.

        h_prev, l_prev, ex: (B, L, D)
        y_pooled: (B, D) or None (None ⇒ sample from prior; not None ⇒ sample from posterior)
        prior_noise_scale: inference-only multiplier for prior std.
        posterior_noise_scale: diagnostic multiplier for posterior std.
        Returns (h_new, l_new, info).
        """
        # If we're under autocast (bf16/fp16), explicitly cast the persistent
        # state and input embeddings to the autocast dtype. Without this, the
        # residual `l_k + h_prev + ex` adds happen in fp32 (autocast only
        # casts MM/Linear ops automatically), promoting the entire core back
        # to fp32 — defeating the bf16 throughput win on small models.
        # Use legacy stable API (works on PyTorch >= 2.0); 2.5+ has
        # device-aware variants but we don't depend on them here.
        if h_prev.is_cuda and torch.is_autocast_enabled():
            cast_dtype = torch.get_autocast_gpu_dtype()
            h_prev = h_prev.to(cast_dtype)
            l_prev = l_prev.to(cast_dtype)
            ex = ex.to(cast_dtype)
            if y_pooled is not None:
                y_pooled = y_pooled.to(cast_dtype)

        # K low-level refinements with h held fixed.
        l_k = l_prev
        for _ in range(self.K):
            l_k = self.fL(l_k + h_prev + ex)

        # Deterministic high-level update.
        u_t = self.fH(h_prev + l_k)

        # Repo-matched guidance mode, shared with the compile-friendly
        # training path above.
        if y_pooled is None:
            eps, mu_p, var_p = self._inference_guidance(
                u_t, prior_noise_scale
            )
            mu_q, var_q = None, None
        else:
            eps, mu_p, var_p, mu_q, var_q = self._training_guidance(
                u_t, y_pooled, posterior_noise_scale
            )

        h_new = u_t + eps
        info = TransitionInfo(
            u_t=u_t,
            eps=eps,
            mu_p=mu_p,
            var_p=var_p,
            mu_q=mu_q,
            var_q=var_q,
            kl_enabled=self.kl_enabled,
        )
        return h_new, l_k, info

    def supervision_step(
        self,
        h: torch.Tensor,
        l: torch.Tensor,
        ex: torch.Tensor,
        y_pooled: torch.Tensor | None = None,
        truncated_grad: bool = True,
        prior_noise_scale: float = 1.0,
        posterior_noise_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, TransitionInfo]:
        """Run T transitions, returning the FINAL latent state + the final-transition info.

        If `truncated_grad` is True, the first T-1 transitions are run under
        `torch.no_grad()` and only the final transition has gradients flowing
        (paper §A.3). The returned info is for the final transition — the KL
        term in the surrogate ELBO uses exactly this transition's noise.
        """
        if truncated_grad and self.T > 1:
            with torch.no_grad():
                for _ in range(self.T - 1):
                    h, l, _ = self.one_transition(
                        h,
                        l,
                        ex,
                        y_pooled,
                        prior_noise_scale=prior_noise_scale,
                        posterior_noise_scale=posterior_noise_scale,
                    )
            # Detach so the grad doesn't try to flow into the no_grad region.
            h = h.detach()
            l = l.detach()
        else:
            for _ in range(self.T - 1):
                h, l, _ = self.one_transition(
                    h,
                    l,
                    ex,
                    y_pooled,
                    prior_noise_scale=prior_noise_scale,
                    posterior_noise_scale=posterior_noise_scale,
                )

        # Final transition — gradients flow.
        h, l, info = self.one_transition(
            h,
            l,
            ex,
            y_pooled,
            prior_noise_scale=prior_noise_scale,
            posterior_noise_scale=posterior_noise_scale,
        )
        return h, l, info
