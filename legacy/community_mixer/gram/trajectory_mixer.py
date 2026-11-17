"""Small residual mixers over the trajectory/particle axis.

The mixer is intentionally separate from the main GRAM model.  It lets us train
an inference-time interaction module on top of a frozen GRAM checkpoint and
compare it directly against the hand-written particle baselines.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


MIXER_ARCHITECTURES = (
    "local_only",
    "global_only",
    "local_global_uniform",
    "local_global_learned",
)
GLOBAL_CONTEXT_MODES = ("same_puzzle", "wrong_puzzle")
_GLOBAL_STATE_PREFIXES = (
    "norm_global.",
    "global_attn.",
    "norm_global_ff.",
    "global_ff.",
    "global_gate_logit",
)


def _is_global_state_key(key: str) -> bool:
    return any(key.startswith(prefix) for prefix in _GLOBAL_STATE_PREFIXES)


def load_trajectory_mixer_state_dict(
    mixer: "ResidualTrajectoryMixer",
    state_dict: Mapping[str, torch.Tensor],
) -> bool:
    """Load a mixer state, accepting legacy local-only checkpoints.

    Returns ``True`` only when all newly introduced global-summary parameters
    were absent and the checkpoint was therefore loaded as a legacy
    ``local_only`` mixer.  Any partial global state, unrelated missing key, or
    unexpected key remains a hard error.
    """

    incompatible = mixer.load_state_dict(dict(state_dict), strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    expected_global = {
        key for key in mixer.state_dict() if _is_global_state_key(key)
    }
    if not missing and not unexpected:
        return False
    if (
        mixer.architecture == "local_only"
        and not unexpected
        and missing == expected_global
    ):
        return True
    raise RuntimeError(
        "incompatible trajectory mixer state: "
        f"missing={sorted(missing)!r} unexpected={sorted(unexpected)!r}"
    )


class ResidualTrajectoryMixer(nn.Module):
    """Residual local/global attention over particles of the same puzzle.

    Input/outputs are flattened as ``(puzzles * particles, tokens, hidden)``.
    The legacy ``local_only`` architecture communicates across particles at
    each token position.  Global architectures additionally mean-pool each
    complete trajectory, communicate across those trajectory summaries, and
    broadcast the resulting residual to every token of the source trajectory.

    Every architecture instantiates the same parameters.  The four-arm gate
    can therefore disable branches without changing parameter count.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        gate_init: float = -4.0,
        ff_mult: int = 2,
        architecture: str = "local_only",
    ) -> None:
        super().__init__()
        if architecture not in MIXER_ARCHITECTURES:
            raise ValueError(f"unsupported mixer architecture: {architecture!r}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.architecture = architecture

        # Keep all legacy names unchanged so old local-only checkpoints can be
        # loaded with the narrowly validated compatibility helper above.
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(d_model)
        hidden = ff_mult * d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

        self.norm_global = nn.LayerNorm(d_model)
        self.global_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_global_ff = nn.LayerNorm(d_model)
        self.global_ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        self.global_gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    @property
    def global_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.global_gate_logit)

    def _local_delta(
        self,
        h: torch.Tensor,
        *,
        n_puzzles: int,
        n_particles: int,
        mode: str,
    ) -> torch.Tensor:
        _batch, seq_len, d_model = h.shape
        x = self.norm_attn(h).reshape(
            n_puzzles, n_particles, seq_len, d_model
        )
        x = x.permute(0, 2, 1, 3)
        queries = x.reshape(n_puzzles * seq_len, n_particles, d_model)

        if mode == "self_only":
            attn_mask = torch.ones(
                n_particles,
                n_particles,
                dtype=torch.bool,
                device=h.device,
            )
            attn_mask.fill_diagonal_(False)
            attn_out, _ = self.attn(
                queries,
                queries,
                queries,
                attn_mask=attn_mask,
                need_weights=False,
            )
        elif mode == "uniform_mean":
            attn_out = self._uniform_attention(self.attn, queries)
        else:
            context = queries
            if mode == "wrong_puzzle":
                context = x.roll(shifts=1, dims=0).reshape(
                    n_puzzles * seq_len, n_particles, d_model
                )
            attn_out, _ = self.attn(
                queries,
                context,
                context,
                need_weights=False,
            )

        attn_out = attn_out.reshape(n_puzzles, seq_len, n_particles, d_model)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape_as(h)
        gate = self.gate.to(dtype=h.dtype)
        y = h + gate * attn_out
        return gate * attn_out + gate * self.ff(self.norm_ff(y))

    @staticmethod
    def _uniform_attention(
        attention: nn.MultiheadAttention,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Use the trained V/output projections with uniform routing."""

        d_model = attention.embed_dim
        v_weight = attention.in_proj_weight[2 * d_model :]
        v_bias = (
            attention.in_proj_bias[2 * d_model :]
            if attention.in_proj_bias is not None
            else None
        )
        values = F.linear(context, v_weight, v_bias)
        pooled = values.mean(dim=1, keepdim=True).expand_as(values)
        return attention.out_proj(pooled)

    def _global_delta(
        self,
        h: torch.Tensor,
        *,
        n_puzzles: int,
        n_particles: int,
        routing: str,
        context_mode: str,
    ) -> torch.Tensor:
        batch, seq_len, d_model = h.shape
        summaries = self.norm_global(h).reshape(
            n_puzzles, n_particles, seq_len, d_model
        ).mean(dim=2)
        context = summaries
        if context_mode == "wrong_puzzle":
            context = summaries.roll(shifts=1, dims=0)

        if routing == "uniform":
            summary_attn = self._uniform_attention(self.global_attn, context)
        else:
            summary_attn, _ = self.global_attn(
                summaries,
                context,
                context,
                need_weights=False,
            )

        gate = self.global_gate.to(dtype=h.dtype)
        summary_y = summaries + gate * summary_attn
        summary_delta = gate * summary_attn
        summary_delta = summary_delta + gate * self.global_ff(
            self.norm_global_ff(summary_y)
        )
        return (
            summary_delta.unsqueeze(2)
            .expand(n_puzzles, n_particles, seq_len, d_model)
            .reshape(batch, seq_len, d_model)
        )

    def forward(
        self,
        h: torch.Tensor,
        n_particles: int,
        mode: str = "learned",
        global_context_mode: str = "same_puzzle",
    ) -> torch.Tensor:
        if mode not in {
            "disabled",
            "learned",
            "self_only",
            "uniform_mean",
            "wrong_puzzle",
        }:
            raise ValueError(f"unsupported mixer mode: {mode!r}")
        if global_context_mode not in GLOBAL_CONTEXT_MODES:
            raise ValueError(
                f"unsupported global context mode: {global_context_mode!r}"
            )
        if mode == "disabled" or n_particles <= 1:
            return h
        batch, _seq_len, d_model = h.shape
        if d_model != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {d_model}")
        if batch % n_particles != 0:
            raise ValueError(
                f"batch={batch} must be divisible by n_particles={n_particles}"
            )
        n_puzzles = batch // n_particles

        use_local = self.architecture != "global_only"
        use_global = self.architecture != "local_only"
        y = h
        if use_local:
            y = y + self._local_delta(
                h,
                n_puzzles=n_puzzles,
                n_particles=n_particles,
                mode=mode,
            )
        if use_global:
            routing = (
                "uniform"
                if self.architecture == "local_global_uniform"
                else "learned"
            )
            y = y + self._global_delta(
                h,
                n_puzzles=n_puzzles,
                n_particles=n_particles,
                routing=routing,
                context_mode=global_context_mode,
            )
        return y
