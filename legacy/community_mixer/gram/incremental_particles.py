"""Incremental particle acquisition for compute-elastic GRAM inference.

The legacy Budget-Elastic evaluation reruns complete forwards at every budget
(``4 + 8 + 16 + 32 + 64`` particles).  This module implements a different,
explicitly nested inference contract:

* previously computed particle logits and mixer contexts are immutable;
* only newly requested particles execute the recursive core;
* at mixer steps, new particles query the cached old-particle context;
* old particles are never recomputed or silently revised.

The query-only mixer is intentionally not claimed to be numerically equivalent
to a dense mixer over the final particle set.  It is the causal set-growth
architecture needed for real compute savings.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trajectory_mixer import ResidualTrajectoryMixer


ROUTER_FEATURE_NAMES = (
    "budget_fraction_log2",
    "decoder_confidence_mean",
    "duplicate_saturation",
    "novelty_rate",
    "constraint_residual",
)

_TRACE_FEATURE_KEYS = (
    "decoder_confidence_mean",
    "duplicate_saturation",
    "novelty_rate",
    "constraint_residual",
)


@dataclass(frozen=True)
class IncrementalParticleState:
    """Reusable state after one or more particle-acquisition rounds.

    ``logits`` has shape ``(n_puzzles, n_particles, seq_len, vocab)`` and
    ``lprm_value`` has shape ``(n_puzzles, n_particles)``.  Mixer contexts are
    stored only for supervision steps where communication is enabled.
    """

    puzzle_inputs: torch.Tensor
    n_particles: int
    logits: torch.Tensor
    lprm_value: torch.Tensor
    mixer_context: dict[int, torch.Tensor]
    computed_particles: int
    nsup: int
    mixer_every: int
    used_mixer: bool

    @property
    def transition_equivalent_particles(self) -> int:
        """Number of particles actually passed through the recursive core."""
        return self.computed_particles


class MarginalUtilityHead(nn.Module):
    """Predict expected new unique-valid solutions per additional particle."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.net = nn.Sequential(
            nn.Linear(len(ROUTER_FEATURE_NAMES), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != len(ROUTER_FEATURE_NAMES):
            raise ValueError(
                f"expected {len(ROUTER_FEATURE_NAMES)} router features, "
                f"got shape {tuple(features.shape)}"
            )
        return F.softplus(self.net(features)).squeeze(-1)


def router_features_from_trace_row(row: Mapping[str, Any]) -> dict[str, float]:
    """Build deployable router features from one current-budget trace row.

    Validity, coverage, solution counts, and any next-budget fields are
    deliberately excluded.  Those values may be used only to construct
    training labels.
    """
    n_samples = int(row["n_samples"])
    if n_samples < 1 or n_samples > 64:
        raise ValueError(f"n_samples must be in [1, 64], got {n_samples}")
    features = {
        "budget_fraction_log2": math.log2(n_samples) / math.log2(64),
        **{key: float(row[key]) for key in _TRACE_FEATURE_KEYS},
    }
    for name, value in features.items():
        if not math.isfinite(value):
            raise ValueError(f"non-finite router feature {name}={value}")
    return features


def marginal_utility_examples(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create adjacent-budget labels without leaking labels into features."""
    ordered = sorted(rows, key=lambda row: int(row["n_samples"]))
    budgets = [int(row["n_samples"]) for row in ordered]
    if len(set(budgets)) != len(budgets):
        raise ValueError(f"duplicate budgets in trace: {budgets}")
    examples: list[dict[str, Any]] = []
    for current, following in zip(ordered, ordered[1:]):
        current_n = int(current["n_samples"])
        next_n = int(following["n_samples"])
        delta_n = next_n - current_n
        if delta_n <= 0:
            raise ValueError("trace budgets must be strictly increasing")
        current_unique = int(current["n_unique_valid"])
        next_unique = int(following["n_unique_valid"])
        gain = max(0, next_unique - current_unique)
        examples.append(
            {
                "current_n": current_n,
                "next_n": next_n,
                "delta_n": delta_n,
                "features": router_features_from_trace_row(current),
                "target_new_unique_valid": gain,
                "target_marginal_utility": gain / delta_n,
            }
        )
    return examples


def should_expand_particles(
    predicted_marginal_utility: torch.Tensor,
    *,
    lambda_compute: float,
) -> torch.Tensor:
    """Return a label-free expand decision at a configured compute price."""
    if not math.isfinite(lambda_compute) or lambda_compute < 0.0:
        raise ValueError("lambda_compute must be finite and non-negative")
    return predicted_marginal_utility > lambda_compute


def _particle_view(
    tensor: torch.Tensor,
    *,
    n_puzzles: int,
    n_particles: int,
) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"expected rank-3 particle tensor, got {tensor.ndim}")
    if tensor.shape[0] != n_puzzles * n_particles:
        raise ValueError(
            f"batch={tensor.shape[0]} does not match "
            f"{n_puzzles} puzzles x {n_particles} particles"
        )
    return tensor.reshape(
        n_puzzles,
        n_particles,
        tensor.shape[1],
        tensor.shape[2],
    )


def _concat_flat_particles(
    old: torch.Tensor | None,
    new: torch.Tensor,
    *,
    n_puzzles: int,
    old_particles: int,
    new_particles: int,
) -> torch.Tensor:
    new_view = _particle_view(
        new,
        n_puzzles=n_puzzles,
        n_particles=new_particles,
    )
    if old is None:
        return new_view.reshape(-1, new.shape[1], new.shape[2])
    old_view = _particle_view(
        old,
        n_puzzles=n_puzzles,
        n_particles=old_particles,
    )
    return torch.cat((old_view, new_view), dim=1).reshape(
        -1,
        new.shape[1],
        new.shape[2],
    )


def incremental_query_mix(
    *,
    mixer: ResidualTrajectoryMixer,
    new_h: torch.Tensor,
    old_context_h: torch.Tensor,
    n_puzzles: int,
    old_particles: int,
    new_particles: int,
) -> torch.Tensor:
    """Update only new particles using old + new particles as KV context."""
    if old_particles < 1:
        raise ValueError("old_particles must be positive for query-only mixing")
    if new_particles < 1:
        raise ValueError("new_particles must be positive")
    new_raw = _particle_view(
        new_h,
        n_puzzles=n_puzzles,
        n_particles=new_particles,
    )
    old_raw = _particle_view(
        old_context_h,
        n_puzzles=n_puzzles,
        n_particles=old_particles,
    )
    if new_raw.shape[2:] != old_raw.shape[2:]:
        raise ValueError(
            "new and cached mixer states must share sequence/model dimensions"
        )
    seq_len, d_model = new_raw.shape[2:]
    if d_model != mixer.d_model:
        raise ValueError(f"expected d_model={mixer.d_model}, got {d_model}")

    new_norm = mixer.norm_attn(new_raw)
    old_norm = mixer.norm_attn(old_raw)
    new_queries = new_norm.permute(0, 2, 1, 3).reshape(
        n_puzzles * seq_len,
        new_particles,
        d_model,
    )
    context = torch.cat((old_norm, new_norm), dim=1)
    context = context.permute(0, 2, 1, 3).reshape(
        n_puzzles * seq_len,
        old_particles + new_particles,
        d_model,
    )
    attn_out, _ = mixer.attn(
        new_queries,
        context,
        context,
        need_weights=False,
    )
    attn_out = attn_out.reshape(
        n_puzzles,
        seq_len,
        new_particles,
        d_model,
    ).permute(0, 2, 1, 3)
    attn_out = attn_out.reshape_as(new_h)

    gate = mixer.gate.to(dtype=new_h.dtype)
    mixed = new_h + gate * attn_out
    return mixed + gate * mixer.ff(mixer.norm_ff(mixed))


def _validate_reuse(
    state: IncrementalParticleState,
    *,
    puzzle_inputs: torch.Tensor,
    nsup: int,
    mixer_every: int,
    used_mixer: bool,
) -> None:
    if state.puzzle_inputs.shape != puzzle_inputs.shape:
        raise ValueError("puzzle input shape changed across acquisition rounds")
    if state.puzzle_inputs.device != puzzle_inputs.device:
        raise ValueError("puzzle input device changed across acquisition rounds")
    if not torch.equal(state.puzzle_inputs, puzzle_inputs):
        raise ValueError("puzzle inputs changed across acquisition rounds")
    if state.nsup != nsup:
        raise ValueError("nsup changed across acquisition rounds")
    if state.mixer_every != mixer_every or state.used_mixer != used_mixer:
        raise ValueError("mixer policy changed across acquisition rounds")


@torch.no_grad()
def append_particles(
    *,
    base_model: nn.Module,
    puzzle_inputs: torch.Tensor,
    additional_particles: int,
    state: IncrementalParticleState | None = None,
    mixer: ResidualTrajectoryMixer | None = None,
    mixer_every: int = 1,
    prior_noise_scale: float = 1.0,
    amp_dtype: torch.dtype | None = None,
    nsup: int | None = None,
) -> IncrementalParticleState:
    """Append particles without recomputing any previously acquired particle."""
    if puzzle_inputs.ndim != 2:
        raise ValueError("puzzle_inputs must have shape (n_puzzles, seq_len)")
    if additional_particles < 1:
        raise ValueError("additional_particles must be positive")
    if mixer_every < 0:
        raise ValueError("mixer_every must be non-negative")
    n_puzzles, seq_len = puzzle_inputs.shape
    if seq_len != base_model.seq_len:
        raise ValueError(
            f"expected seq_len={base_model.seq_len}, got {seq_len}"
        )
    n_steps = base_model.nsup_train if nsup is None else int(nsup)
    if n_steps < 1:
        raise ValueError("nsup must be positive")
    used_mixer = mixer is not None and mixer_every > 0
    old_particles = 0 if state is None else state.n_particles
    if state is not None:
        _validate_reuse(
            state,
            puzzle_inputs=puzzle_inputs,
            nsup=n_steps,
            mixer_every=mixer_every,
            used_mixer=used_mixer,
        )

    expanded_inputs = puzzle_inputs.unsqueeze(1).expand(
        n_puzzles,
        additional_particles,
        seq_len,
    ).reshape(-1, seq_len)
    ex = base_model.encode_input(expanded_inputs)
    h, l = base_model.core.initial_state(expanded_inputs.shape[0])
    h = h.to(puzzle_inputs.device, non_blocking=True)
    l = l.to(puzzle_inputs.device, non_blocking=True)
    mixer_context = {} if state is None else dict(state.mixer_context)

    use_autocast = amp_dtype is not None and puzzle_inputs.device.type == "cuda"
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if use_autocast
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        for step_idx in range(1, n_steps + 1):
            h, l, _ = base_model.core.supervision_step(
                h,
                l,
                ex,
                y_pooled=None,
                truncated_grad=True,
                prior_noise_scale=prior_noise_scale,
            )
            if used_mixer and step_idx % mixer_every == 0:
                assert mixer is not None
                old_context = (
                    None if state is None else state.mixer_context.get(step_idx)
                )
                if old_context is None:
                    h = mixer(
                        h,
                        n_particles=additional_particles,
                        mode="learned",
                    )
                else:
                    h = incremental_query_mix(
                        mixer=mixer,
                        new_h=h,
                        old_context_h=old_context,
                        n_puzzles=n_puzzles,
                        old_particles=old_particles,
                        new_particles=additional_particles,
                    )
                mixer_context[step_idx] = _concat_flat_particles(
                    old_context,
                    h,
                    n_puzzles=n_puzzles,
                    old_particles=old_particles,
                    new_particles=additional_particles,
                ).detach()
            h = h.detach()
            l = l.detach()
        new_logits = base_model.lm_head(h)
        new_lprm = base_model.lprm_head(h)

    new_logits = new_logits.reshape(
        n_puzzles,
        additional_particles,
        seq_len,
        base_model.vocab_size,
    )
    new_lprm = new_lprm.reshape(n_puzzles, additional_particles)
    if state is None:
        logits = new_logits
        lprm_value = new_lprm
        computed_particles = additional_particles
    else:
        logits = torch.cat((state.logits, new_logits), dim=1)
        lprm_value = torch.cat((state.lprm_value, new_lprm), dim=1)
        computed_particles = state.computed_particles + additional_particles

    return IncrementalParticleState(
        puzzle_inputs=puzzle_inputs.detach(),
        n_particles=old_particles + additional_particles,
        logits=logits.detach(),
        lprm_value=lprm_value.detach(),
        mixer_context=mixer_context,
        computed_particles=computed_particles,
        nsup=n_steps,
        mixer_every=mixer_every,
        used_mixer=used_mixer,
    )
