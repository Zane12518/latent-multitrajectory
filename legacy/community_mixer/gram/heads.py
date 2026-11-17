"""Stochastic-guidance / posterior / halt / LPRM heads (Stage B).

References to paper:
- §2.1: ε_t ~ N(μ_θ(u_t), σ_θ²(u_t) I) — prior guidance head, operates on u_t.
- §2.2: q_φ(ε_t | u_t, y) — posterior head, conditioned on the target output y.
- §B.1: μ_θ, σ_θ, μ_φ, σ_φ each parameterized as SwiGLU MLPs.
- §A.1: halt head q_ψ : R^D → R^2 (we use the released-code single-q_halt variant).
- §A.2: LPRM v_ψ(z) : R^D → R, regression onto r ∈ [0, 1].

Conventions:
- σ²(·) > 0 enforced via softplus + floor (`SIGMA_FLOOR`) for numerical stability;
  the head's raw output is the pre-activation.
- Halt and LPRM heads read the FIRST token of the high-level state (per §B.1).
- Posterior is conditioned on y by mean-pooling the target tokens through the
  shared token-embedding table — per BTN-method-detail-gaps gap default #1.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import RMSNorm, SwiGLUMLP

# Floor on σ² for numerical stability. Standard VAE practice; matches Dreamer family.
SIGMA_FLOOR = 1e-4


# ----------------------------------------------------------------------------
# Diagonal Gaussian helpers
# ----------------------------------------------------------------------------


def softplus_var(raw: torch.Tensor) -> torch.Tensor:
    """Map raw head output → σ² > 0 with a stability floor."""
    return F.softplus(raw) + SIGMA_FLOOR


def reparameterize(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """ε ~ N(μ, σ² I) using the reparameterization trick. var is σ² (not log-σ²)."""
    std = var.clamp_min(SIGMA_FLOOR).sqrt()
    eps = torch.randn_like(mu)
    return mu + std * eps


def gaussian_kl(mu_q: torch.Tensor, var_q: torch.Tensor, mu_p: torch.Tensor, var_p: torch.Tensor) -> torch.Tensor:
    """KL(N(μ_q, σ_q²) || N(μ_p, σ_p²)) for diagonal Gaussians.

    Returns KL summed over the last (channel) dim; shape = inputs without last dim.
    """
    var_q = var_q.clamp_min(SIGMA_FLOOR)
    var_p = var_p.clamp_min(SIGMA_FLOOR)
    return 0.5 * (torch.log(var_p / var_q) + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0).sum(dim=-1)


def kl_balanced(
    mu_q: torch.Tensor,
    var_q: torch.Tensor,
    mu_p: torch.Tensor,
    var_p: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Dreamer-V2 KL balancing.

    α * KL(sg(q) ‖ p) + (1 - α) * KL(q ‖ sg(p))

    α large (e.g. 0.8) → mostly train the prior to match the posterior;
    α small → mostly train the posterior to match the prior.
    Paper §B.2 uses α = 0.8.
    """
    kl_prior_update = gaussian_kl(mu_q.detach(), var_q.detach(), mu_p, var_p)
    kl_posterior_update = gaussian_kl(mu_q, var_q, mu_p.detach(), var_p.detach())
    return alpha * kl_prior_update + (1.0 - alpha) * kl_posterior_update


# ----------------------------------------------------------------------------
# Guidance heads
# ----------------------------------------------------------------------------


class DiagonalGaussianHead(nn.Module):
    """Predicts (μ, σ²) from an input feature via two parallel SwiGLU MLPs.

    Paper §B.1: μ_θ, σ_θ, μ_φ, σ_φ are each a SwiGLU MLP — interpreted as four
    independent SwiGLU MLPs sharing the same input feature.
    """

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.mu = SwiGLUMLP(d_model, d_hidden, d_model)
        self.var = SwiGLUMLP(d_model, d_hidden, d_model)
        # Bias the σ² head toward small values at init so that early in training
        # the stochastic guidance is small (deterministic-leaning); standard in
        # VAE-style models and avoids early-training noise washing out signal.
        with torch.no_grad():
            for p in self.var.proj.parameters():
                p.zero_()

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(feat)
        var = softplus_var(self.var(feat))
        return mu, var


class GuidancePriorHead(DiagonalGaussianHead):
    """Prior p_θ(ε_t | u_t) = N(μ_θ(u_t), σ_θ²(u_t) I)."""

    pass


class PosteriorHead(nn.Module):
    """Variational posterior q_φ(ε_t | u_t, y) = N(μ_φ, σ_φ²).

    Conditions on y via a mean-pooled token embedding (gap default #1 in
    BTN-method-detail-gaps). The shared input feature for the SwiGLU MLPs is
    `concat(u_t, y_emb_broadcast)`, projected back to d_model.
    """

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.combine = nn.Linear(2 * d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.head = DiagonalGaussianHead(d_model, d_hidden)

    def forward(self, u_t: torch.Tensor, y_pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        u_t: (B, L, D) — high-level deterministic update at this transition
        y_pooled: (B, D) — mean-pooled target-token embedding (gap default)
        """
        # Broadcast pooled y over the L dim, concat, project, norm.
        L = u_t.shape[1]
        y_b = y_pooled.unsqueeze(1).expand(-1, L, -1)
        feat = self.norm(self.combine(torch.cat([u_t, y_b], dim=-1)))
        return self.head(feat)


# ----------------------------------------------------------------------------
# Halt head — paper §A.1 simplified variant (single q_halt scalar)
# ----------------------------------------------------------------------------


class HaltHead(nn.Module):
    """q_halt(h) — predicts a halting score. We use the released-code simplified
    variant: halt when σ(q_halt) > 0.5. Reads the FIRST token of h per §B.1.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, L, D) → (B,) scalar logit."""
        return self.proj(h[:, 0, :]).squeeze(-1)


# ----------------------------------------------------------------------------
# LPRM head — paper §A.2
# ----------------------------------------------------------------------------


class LPRMHead(nn.Module):
    """v_ψ(z) ∈ [0, 1] regressing on final-prediction accuracy. Reads h[:, 0, :]."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, L, D) → (B,) value in (0, 1)."""
        return torch.sigmoid(self.proj(h[:, 0, :]).squeeze(-1))


# Silence "imported but unused" if math is later removed — we keep it for the
# numerical-stability rationale in the docstrings.
_ = math
