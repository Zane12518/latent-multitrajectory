"""Exponential moving average for model parameters.

Paper §B.2 specifies EMA decay = 0.9999 across all tasks. EMA-of-parameters
is applied for evaluation; training-time gradient updates use the raw weights.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    """Standard parameter EMA. Shadow buffers track `decay`-weighted average."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }
        self._backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        """Swap model weights with the EMA shadow (call `restore` to undo)."""
        if self._backup is not None:
            raise RuntimeError("EMA.apply_to() already active; call restore() first")
        self._backup = {
            n: p.data.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module) -> None:
        if self._backup is None:
            return
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self._backup[n])
        self._backup = None

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": copy.deepcopy(self.shadow)}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]
