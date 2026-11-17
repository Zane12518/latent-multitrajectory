"""Top-level GRAMModel (Stage D).

Components per paper §B.1 / §2.1:
  Encoder: token embedding (scaled by √D) → ex
  Recursive core: T × N_sup transitions with stochastic guidance
  Decoder: linear head over h, vocab logits
  Halt head: q_halt(h[:, 0, :]) per §A.1 simplified variant
  LPRM head: v(h[:, 0, :]) per §A.2

Forward modes:
  * training=True, y provided: runs N_sup supervision steps with the posterior
    on every transition. Returns a list of `SupervisionStepOutput` (one per
    step) so the loss module can apply deep supervision.
  * training=False, y is None: same recursion but every transition samples
    from the prior. Returns the same shape of outputs; the loss is meaningless
    here. Use `generate()` for proper inference with ACT halting / parallel
    sampling.

Puzzle-embedding tokens (§B.1, ARC only) are not implemented in this Stage D
file — N-Queens has `puzzle_emb: ✗`. They will be added when we tackle ARC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .blocks import RotaryPositionalEmbedding
from .heads import HaltHead, LPRMHead
from .recursive import RecursiveCore, TransitionInfo


@dataclass
class SupervisionStepOutput:
    """One forward output of a supervision step (paper §2.1)."""

    logits: torch.Tensor          # (B, L, vocab) — decoder output at end of step
    halt_logit: torch.Tensor      # (B,) — pre-sigmoid halt score
    lprm_value: torch.Tensor      # (B,) — LPRM prediction in (0, 1)
    transition_info: TransitionInfo  # for the FINAL transition of this step
    h: torch.Tensor               # (B, L, D) terminal high-level state
    l: torch.Tensor               # (B, L, D) terminal low-level state


class GRAMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 512,
        fL_layers: int = 2,
        fH_layers: int = 2,
        K_inner: int = 4,
        T_outer: int = 3,
        nsup: int = 16,
        guidance_hidden_mult: int = 1,
        z0_seed: int = 1234,
        position_aware_token_embedding: bool = False,
        recursive_residual_scale_exponent: float = 0.0,
        guidance_mode: str = "full",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.nsup_train = nsup
        self.position_aware_token_embedding = (
            position_aware_token_embedding
        )
        self.recursive_residual_scale_exponent = float(
            recursive_residual_scale_exponent
        )

        # Shared RoPE (so fL and fH agree on positional encoding).
        self.rope = RotaryPositionalEmbedding(head_dim=d_model // n_heads)

        # Encoder: token embedding scaled by √D (paper §B.1).
        self.token_embed = nn.Embedding(vocab_size, d_model)
        nn.init.trunc_normal_(self.token_embed.weight, std=1.0 / math.sqrt(d_model))
        self._embed_scale = math.sqrt(d_model)
        if self.position_aware_token_embedding:
            # A direct (position, token) interaction is deliberately used
            # instead of a purely additive absolute-position embedding.
            # Additive positions would cancel under the posterior's mean
            # pooling and leave targets with the same token histogram
            # indistinguishable. The optional table preserves legacy
            # checkpoints because it is absent unless explicitly enabled.
            self.position_token_embed: nn.Embedding | None = nn.Embedding(
                seq_len * vocab_size, d_model
            )
            nn.init.trunc_normal_(
                self.position_token_embed.weight,
                std=1.0 / math.sqrt(d_model),
            )
        else:
            self.position_token_embed = None

        # Recursive core.
        self.core = RecursiveCore(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            seq_len=seq_len,
            fL_layers=fL_layers,
            fH_layers=fH_layers,
            K_inner=K_inner,
            T_outer=T_outer,
            guidance_hidden_mult=guidance_hidden_mult,
            rope=self.rope,
            z0_seed=z0_seed,
            residual_scale_exponent=self.recursive_residual_scale_exponent,
            residual_scale_nsup=nsup,
            guidance_mode=guidance_mode,
        )
        self.guidance_mode = self.core.guidance_mode

        # Decoder: Linear(D → vocab) per paper §B.1.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.trunc_normal_(self.lm_head.weight, std=1.0 / math.sqrt(d_model))

        # Auxiliary heads.
        self.halt_head = HaltHead(d_model)
        self.lprm_head = LPRMHead(d_model)

    # ------------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------------

    def _encode_position_aware_tokens(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Embed tokens with an optional learned (position, token) table."""
        token_emb = self.token_embed(tokens)
        if self.position_token_embed is None:
            return token_emb * self._embed_scale
        if tokens.ndim != 2 or tokens.shape[1] != self.seq_len:
            raise ValueError(
                "position-aware encoding requires tokens with shape "
                f"(B, {self.seq_len}), got {tuple(tokens.shape)}"
            )
        positions = torch.arange(
            self.seq_len, device=tokens.device, dtype=tokens.dtype
        ).unsqueeze(0)
        pair_index = positions * self.vocab_size + tokens
        pair_emb = self.position_token_embed(pair_index)
        return (token_emb + pair_emb) * (self._embed_scale / math.sqrt(2.0))

    def encode_input(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) long → ex: (B, L, D)."""
        return self._encode_position_aware_tokens(x)

    def encode_target_for_posterior(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, L) long → y_pooled: (B, D). Mean-pool target embeddings.

        Gap default #1 in BTN-method-detail-gaps. We share the token embedding
        table between input and target encodings (vocab is the same for these
        tasks).
        """
        y_emb = self._encode_position_aware_tokens(y)
        active = (y != 0).unsqueeze(-1)
        denom = active.sum(dim=1).clamp_min(1)
        return (y_emb * active).sum(dim=1) / denom

    # ------------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        nsup: int | None = None,
        truncated_grad: bool = True,
        prior_noise_scale: float = 1.0,
        posterior_noise_scale: float = 1.0,
    ) -> list[SupervisionStepOutput]:
        """Run N_sup supervision steps, returning per-step decoder outputs (deep supervision).

        x: (B, L) input token ids.
        y: (B, L) target token ids OR None.
           If provided, posterior is used at every transition; KL terms are well-defined.
           If None, prior is used (free-running / inference mode).
        nsup: override the default training N_sup; useful for ACT-style inference.
        """
        B = x.shape[0]
        ex = self.encode_input(x)
        y_pooled = self.encode_target_for_posterior(y) if y is not None else None

        h, l = self.core.initial_state(B)
        n_steps = nsup if nsup is not None else self.nsup_train
        outputs: list[SupervisionStepOutput] = []
        for _ in range(n_steps):
            h, l, info = self.core.supervision_step(
                h,
                l,
                ex,
                y_pooled=y_pooled,
                truncated_grad=truncated_grad,
                prior_noise_scale=prior_noise_scale,
                posterior_noise_scale=posterior_noise_scale,
            )
            logits = self.lm_head(h)
            # Paper §A.1 footnote: the ACT halt aux loss "does not propagate
            # gradients into the recursive core". Detach h going into the halt
            # head so BCE on halt only updates halt_head's parameters, not
            # fL / fH / guidance heads.
            # §A.2 is silent on LPRM gradient flow — keeping it connected for now;
            # can detach if EXP-8 convergence parity check shows regression.
            halt_logit = self.halt_head(h.detach())
            lprm_value = self.lprm_head(h)
            outputs.append(
                SupervisionStepOutput(
                    logits=logits,
                    halt_logit=halt_logit,
                    lprm_value=lprm_value,
                    transition_info=info,
                    h=h,
                    l=l,
                )
            )
            # Carry-over for next supervision step. Deep supervision: the next
            # step's gradient should not flow back through this step's terminal
            # state (paper §A.3 truncation policy). We detach h and l here.
            h = h.detach()
            l = l.detach()
        return outputs

    # ------------------------------------------------------------------------
    # Convenience: number of parameters
    # ------------------------------------------------------------------------

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
