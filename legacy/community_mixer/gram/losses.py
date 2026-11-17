"""Training losses (Stage E).

Combines per-supervision-step ELBO surrogate (paper Eq. 14), simplified ACT TD
loss (paper §A.1, released-code single-q_halt variant), and LPRM regression
loss (paper §A.2).

Per-supervision-step ELBO surrogate (Eq. 14):

    L_GRAM^(n) = E_q[ log p_θ(y | z_T^(n), x) ]
                 − β · KL( q_φ(ε_T^(n) | u_T^(n), y) ‖ p_θ(ε_T^(n) | u_T^(n)) )

Equivalent training objective (we minimize):

    L^(n) = CE( decoder(h_T^(n)), y ) + β · KL_balanced(q, p, α)

Total deep-supervision loss = mean over N_sup supervision steps of L^(n)
(per-step weighting; using sum vs mean only changes the effective learning
rate, mean is more stable).

ACT halt (simplified variant per §A.1 footnote): the binary cross-entropy of
σ(q_halt) against the actual exact-match accuracy of the step's prediction.
We train at every supervision step.

LPRM (§A.2): MSE between the head's prediction and r ∈ [0, 1] = exact-match
accuracy at this supervision step. We use the per-step terminal latent (the
only one accessible after truncation) instead of summing over t within a
step.

All losses keep their dimensions clear so the caller can re-weight if needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .heads import kl_balanced
from .model import SupervisionStepOutput


@dataclass
class GRAMLossBreakdown:
    """All loss components, per-step and aggregated.

    Shapes:
      ce_per_step, kl_per_step, halt_per_step, lprm_per_step: (N_sup,)
      total: scalar tensor
    """

    ce_per_step: torch.Tensor
    kl_per_step: torch.Tensor
    halt_per_step: torch.Tensor
    lprm_per_step: torch.Tensor
    graph_conflict_per_step: torch.Tensor
    graph_color_mass_per_step: torch.Tensor
    nqueen_count_per_step: torch.Tensor
    nqueen_row_per_step: torch.Tensor
    nqueen_col_per_step: torch.Tensor
    nqueen_diag_per_step: torch.Tensor
    sat_clause_per_step: torch.Tensor
    sudoku_constraint_per_step: torch.Tensor
    prior_path_ce_per_step: torch.Tensor
    valid_set_margin_per_step: torch.Tensor
    valid_set_margin_gap_per_step: torch.Tensor
    source_supported_valid_mode_lift_per_step: torch.Tensor
    source_supported_valid_mode_delta_per_step: torch.Tensor
    source_prior_exact_invalid_suppression_per_step: torch.Tensor
    source_prior_exact_invalid_delta_per_step: torch.Tensor
    valid_mode_bipartite_assignment_per_step: torch.Tensor
    accuracy_per_step: torch.Tensor  # (N_sup,) — exact-match accuracy
    total: torch.Tensor

    def as_dict_for_logging(self) -> dict[str, float]:
        return {
            "loss_total": self.total.item(),
            "loss_ce_mean": self.ce_per_step.mean().item(),
            "loss_kl_mean": self.kl_per_step.mean().item(),
            "loss_halt_mean": self.halt_per_step.mean().item(),
            "loss_lprm_mean": self.lprm_per_step.mean().item(),
            "loss_graph_conflict_mean": self.graph_conflict_per_step.mean().item(),
            "loss_graph_color_mass_mean": self.graph_color_mass_per_step.mean().item(),
            "loss_nqueen_count_mean": self.nqueen_count_per_step.mean().item(),
            "loss_nqueen_row_mean": self.nqueen_row_per_step.mean().item(),
            "loss_nqueen_col_mean": self.nqueen_col_per_step.mean().item(),
            "loss_nqueen_diag_mean": self.nqueen_diag_per_step.mean().item(),
            "loss_sat_clause_mean": self.sat_clause_per_step.mean().item(),
            "loss_sudoku_constraint_mean": self.sudoku_constraint_per_step.mean().item(),
            "loss_prior_path_ce_mean": self.prior_path_ce_per_step.mean().item(),
            "loss_valid_set_margin_mean": self.valid_set_margin_per_step.mean().item(),
            "valid_set_margin_gap_mean": self.valid_set_margin_gap_per_step.mean().item(),
            "loss_source_supported_valid_mode_lift_mean": self.source_supported_valid_mode_lift_per_step.mean().item(),
            "source_supported_valid_mode_delta_mean": self.source_supported_valid_mode_delta_per_step.mean().item(),
            "loss_source_prior_exact_invalid_suppression_mean": self.source_prior_exact_invalid_suppression_per_step.mean().item(),
            "source_prior_exact_invalid_delta_mean": self.source_prior_exact_invalid_delta_per_step.mean().item(),
            "loss_valid_mode_bipartite_assignment_mean": self.valid_mode_bipartite_assignment_per_step.mean().item(),
            "acc_last_step": self.accuracy_per_step[-1].item(),
            "acc_best_step": self.accuracy_per_step.max().item(),
        }


def cross_entropy_step(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Token-mean cross-entropy averaged over the batch.

    logits: (B, L, V), target: (B, L). Returns a scalar tensor.

    Token 0 is the repo-wide pad token. N-Queens targets do not contain pad, so
    this is a no-op there; Graph Coloring pads the color sequence to the input
    length and must not train on padded output positions.
    """
    B, L, V = logits.shape
    return F.cross_entropy(
        logits.view(B * L, V),
        target.view(B * L),
        weight=class_weights,
        reduction="mean",
        ignore_index=0,
    )


def exact_match_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-example exact-match (all-tokens-correct) accuracy.

    Returns (B,) tensor of 0/1 floats. The scalar batch accuracy is `.mean()`.
    """
    pred = logits.argmax(dim=-1)
    active = target != 0
    return ((pred == target) | ~active).all(dim=-1).float()


def kl_term_step(info, kl_balance_coef: float) -> torch.Tensor:
    """KL between posterior and prior at the FINAL transition of this step.

    info: TransitionInfo with mu_q/var_q set (training mode). Returns the
    mean KL across batch and sequence dims (one scalar per supervision step).
    """
    if not getattr(info, "kl_enabled", True):
        # Keep a differentiable zero on the same device/dtype while avoiding
        # an undefined KL for the deterministic repo-matched arms.
        return info.u_t.sum() * 0.0
    assert info.mu_q is not None and info.var_q is not None, (
        "kl_term_step called without a posterior (y was None)"
    )
    kl = kl_balanced(info.mu_q, info.var_q, info.mu_p, info.var_p, alpha=kl_balance_coef)
    # kl shape: (B, L) — sum over channel dim already done inside kl_balanced.
    return kl.mean()


def halt_bce_step(halt_logit: torch.Tensor, acc_per_example: torch.Tensor) -> torch.Tensor:
    """BCE of σ(q_halt) against (per-example exact-match accuracy).

    halt_logit: (B,) pre-sigmoid scores. acc_per_example: (B,) in {0, 1}.
    """
    return F.binary_cross_entropy_with_logits(halt_logit, acc_per_example, reduction="mean")


def lprm_mse_step(lprm_value: torch.Tensor, acc_per_example: torch.Tensor) -> torch.Tensor:
    """MSE between LPRM prediction (already in (0, 1)) and per-example accuracy."""
    return F.mse_loss(lprm_value, acc_per_example, reduction="mean")


def _graph_edges_from_upper_triangle(x: torch.Tensor, n_vertices: int) -> torch.Tensor:
    """Return a bool edge mask with shape (B, n_vertices, n_vertices)."""
    B = x.shape[0]
    edge_mask = torch.zeros((B, n_vertices, n_vertices), dtype=torch.bool, device=x.device)
    k = 0
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            edge = x[:, k] == 2
            edge_mask[:, i, j] = edge
            edge_mask[:, j, i] = edge
            k += 1
    return edge_mask


def graph_coloring_constraint_step(
    logits: torch.Tensor,
    x: torch.Tensor,
    *,
    n_vertices: int,
    vertex_offset: int,
    color_token_start: int = 3,
    n_colors: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft graph-coloring losses.

    conflict: expected same-color mass across graph edges:
      mean_edges sum_c p_i(c) p_j(c)

    color_mass: negative log probability assigned to the allowed color token
    set at vertex output positions. This keeps the model inside {3,4,5} while
    conflict teaches the actual edge constraint.
    """
    if n_vertices <= 0:
        z = logits.sum() * 0.0
        return z, z
    color_end = color_token_start + n_colors
    query_logits = logits[:, vertex_offset : vertex_offset + n_vertices, :]
    color_probs = query_logits.softmax(dim=-1)[..., color_token_start:color_end]
    color_mass = color_probs.sum(dim=-1).clamp_min(1e-8)
    color_mass_loss = -color_mass.log().mean()

    edges = _graph_edges_from_upper_triangle(x, n_vertices)
    upper = torch.triu(torch.ones(n_vertices, n_vertices, device=x.device, dtype=torch.bool), 1)
    pair_same = torch.einsum("bic,bjc->bij", color_probs, color_probs)
    edge_weights = edges[:, upper].to(pair_same.dtype)
    conflict_loss = (pair_same[:, upper] * edge_weights).sum() / edge_weights.sum().clamp_min(1.0)
    return conflict_loss, color_mass_loss


def nqueens_constraint_step(
    logits: torch.Tensor,
    *,
    board_size: int,
    queen_token: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft N-Queens validity losses on queen probability mass.

    These losses are intentionally smooth and task-structural:
      count: total queen mass should be N
      row: each row should contain one queen
      col: no column should contain more than one queen
      diag: no diagonal should contain more than one queen
    """
    if board_size <= 0:
        z = logits.sum() * 0.0
        return z, z, z, z
    B, L, _V = logits.shape
    n = int(board_size)
    if L != n * n:
        raise ValueError(f"N-Queens seq_len={L} is incompatible with board_size={n}")

    p = logits.softmax(dim=-1)[..., queen_token].reshape(B, n, n)
    count = p.sum(dim=(1, 2))
    count_loss = ((count - n) ** 2 / max(n, 1)).mean()

    row_mass = p.sum(dim=2)
    col_mass = p.sum(dim=1)
    row_loss = ((row_mass - 1.0) ** 2).mean()
    col_loss = (torch.relu(col_mass - 1.0) ** 2).mean()

    diag_losses = []
    for offset in range(-(n - 1), n):
        d = p.diagonal(offset=offset, dim1=1, dim2=2)
        if d.shape[1] > 1:
            diag_losses.append((torch.relu(d.sum(dim=1) - 1.0) ** 2).mean())
    flipped = p.flip(dims=(2,))
    for offset in range(-(n - 1), n):
        d = flipped.diagonal(offset=offset, dim1=1, dim2=2)
        if d.shape[1] > 1:
            diag_losses.append((torch.relu(d.sum(dim=1) - 1.0) ** 2).mean())
    diag_loss = torch.stack(diag_losses).mean() if diag_losses else p.sum() * 0.0
    return count_loss, row_loss, col_loss, diag_loss


def sat_clause_constraint_step(
    logits: torch.Tensor,
    x: torch.Tensor,
    *,
    n_variables: int,
    literal_slots: int,
    false_token: int,
    true_token: int,
) -> torch.Tensor:
    """Differentiable probability that every active 3-SAT clause is met.

    The answer logits are normalized within the boolean token pair.  This
    isolates the structural signal from the already-solved output-syntax
    problem and makes each clause loss depend on the literals in ``x``.
    Input literal tokens follow the SAT dataset contract: 1..N are positive,
    N+1..2N are negative, and all-zero triples are padded clauses.
    """
    if n_variables <= 0 or literal_slots <= 0 or literal_slots % 3 != 0:
        raise ValueError("invalid SAT layout")
    if logits.shape[1] < literal_slots + n_variables:
        raise ValueError("SAT logits are shorter than the registered layout")
    if x.shape[1] < literal_slots:
        raise ValueError("SAT inputs are shorter than the registered literal slots")

    answer_probs = logits.softmax(dim=-1)[
        :, literal_slots : literal_slots + n_variables, :
    ]
    false_prob = answer_probs[..., false_token]
    true_prob = answer_probs[..., true_token]
    boolean_mass = (false_prob + true_prob).clamp_min(1.0e-8)
    false_prob = false_prob / boolean_mass
    true_prob = true_prob / boolean_mass

    literals = x[:, :literal_slots].reshape(x.shape[0], -1, 3)
    active_clause = literals.ne(0).any(dim=-1)
    positive = literals.ge(1) & literals.le(n_variables)
    negative = literals.gt(n_variables) & literals.le(2 * n_variables)
    if bool((literals.ne(0) & ~(positive | negative)).any()):
        raise ValueError("SAT inputs contain an invalid literal token")
    variable_index = ((literals - 1) % n_variables).clamp_min(0)
    expanded_true = true_prob.unsqueeze(1).expand(-1, literals.shape[1], -1)
    expanded_false = false_prob.unsqueeze(1).expand(-1, literals.shape[1], -1)
    literal_true = torch.where(
        positive,
        expanded_true.gather(2, variable_index),
        expanded_false.gather(2, variable_index),
    )
    # Padding appears only in fully padded clauses and is ignored below.
    clause_unsatisfied = (1.0 - literal_true).prod(dim=-1)
    clause_satisfied = (1.0 - clause_unsatisfied).clamp_min(1.0e-8)
    if not bool(active_clause.any()):
        return logits.sum() * 0.0
    return -clause_satisfied[active_clause].log().mean()


def sudoku_constraint_step(
    logits: torch.Tensor,
    *,
    board_size: int = 4,
    box_size: int = 2,
    digit_token_start: int = 1,
) -> torch.Tensor:
    """Answer-independent soft Sudoku validity loss."""
    n = int(board_size)
    box = int(box_size)
    if n <= 0 or box <= 0 or box * box != n:
        raise ValueError("Sudoku requires board_size == box_size ** 2")
    if logits.shape[1] != n * n:
        raise ValueError("Sudoku logits have an incompatible sequence length")
    digit_end = digit_token_start + n
    if digit_token_start < 0 or digit_end > logits.shape[-1]:
        raise ValueError("Sudoku digit tokens are outside the vocabulary")

    token_probs = logits.softmax(dim=-1)
    digit_probs = token_probs[..., digit_token_start:digit_end]
    allowed_loss = -digit_probs.sum(dim=-1).clamp_min(1.0e-8).log().mean()
    grid = digit_probs.reshape(logits.shape[0], n, n, n)
    row_loss = ((grid.sum(dim=2) - 1.0) ** 2).mean()
    col_loss = ((grid.sum(dim=1) - 1.0) ** 2).mean()
    boxes = grid.reshape(logits.shape[0], box, box, box, box, n)
    box_mass = boxes.sum(dim=(2, 4))
    box_loss = ((box_mass - 1.0) ** 2).mean()
    return (allowed_loss + row_loss + col_loss + box_loss) / 4.0


def compute_gram_loss(
    outputs: list[SupervisionStepOutput],
    target: torch.Tensor,
    beta_kl: float,
    kl_balance_coef: float = 0.8,
    w_halt: float = 1.0,
    w_lprm: float = 1.0,
    class_weights: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
    graph_coloring: dict | None = None,
    nqueens: dict | None = None,
    sat: dict | None = None,
    sudoku: dict | None = None,
) -> GRAMLossBreakdown:
    """Full GRAM training loss, summed over supervision steps.

    target: (B, L) ground-truth tokens.
    beta_kl: paper β coefficient on the KL term per step (task-specific).
    kl_balance_coef: α in Dreamer-V2 KL balancing (paper § B.2 says 0.8).
    w_halt, w_lprm: auxiliary-loss weights; default 1.0 (no rescaling).
    """
    ce_list = []
    kl_list = []
    halt_list = []
    lprm_list = []
    graph_conflict_list = []
    graph_color_mass_list = []
    nqueen_count_list = []
    nqueen_row_list = []
    nqueen_col_list = []
    nqueen_diag_list = []
    sat_clause_list = []
    sudoku_constraint_list = []
    acc_list = []
    graph_cfg = graph_coloring or {}
    w_graph_conflict = float(graph_cfg.get("w_conflict", 0.0))
    w_graph_color_mass = float(graph_cfg.get("w_color_mass", 0.0))
    use_graph_loss = (
        x is not None
        and (w_graph_conflict != 0.0 or w_graph_color_mass != 0.0)
    )
    nqueen_cfg = nqueens or {}
    w_nqueen_count = float(nqueen_cfg.get("w_count", 0.0))
    w_nqueen_row = float(nqueen_cfg.get("w_row", 0.0))
    w_nqueen_col = float(nqueen_cfg.get("w_col", 0.0))
    w_nqueen_diag = float(nqueen_cfg.get("w_diag", 0.0))
    use_nqueen_loss = (
        w_nqueen_count != 0.0
        or w_nqueen_row != 0.0
        or w_nqueen_col != 0.0
        or w_nqueen_diag != 0.0
    )
    sat_cfg = sat or {}
    w_sat_clause = float(sat_cfg.get("w_clause", 0.0))
    use_sat_loss = x is not None and w_sat_clause != 0.0
    sudoku_cfg = sudoku or {}
    w_sudoku_constraint = float(sudoku_cfg.get("w_constraint", 0.0))
    use_sudoku_loss = w_sudoku_constraint != 0.0

    for step in outputs:
        # Per-example accuracy (used for halt + LPRM targets).
        acc_per_ex = exact_match_accuracy(step.logits.detach(), target)
        acc_list.append(acc_per_ex.mean())

        ce = cross_entropy_step(step.logits, target, class_weights=class_weights)
        kl = kl_term_step(step.transition_info, kl_balance_coef)
        halt = halt_bce_step(step.halt_logit, acc_per_ex)
        lprm = lprm_mse_step(step.lprm_value, acc_per_ex)
        if use_graph_loss:
            graph_conflict, graph_color_mass = graph_coloring_constraint_step(
                step.logits,
                x,
                n_vertices=int(graph_cfg["n_vertices"]),
                vertex_offset=int(graph_cfg["vertex_offset"]),
                color_token_start=int(graph_cfg.get("color_token_start", 3)),
                n_colors=int(graph_cfg.get("n_colors", 3)),
            )
        else:
            graph_conflict = ce * 0.0
            graph_color_mass = ce * 0.0
        if use_nqueen_loss:
            nqueen_count, nqueen_row, nqueen_col, nqueen_diag = nqueens_constraint_step(
                step.logits,
                board_size=int(nqueen_cfg["board_size"]),
                queen_token=int(nqueen_cfg.get("queen_token", 2)),
            )
        else:
            nqueen_count = ce * 0.0
            nqueen_row = ce * 0.0
            nqueen_col = ce * 0.0
            nqueen_diag = ce * 0.0
        if use_sat_loss:
            sat_clause = sat_clause_constraint_step(
                step.logits,
                x,
                n_variables=int(sat_cfg["n_variables"]),
                literal_slots=int(sat_cfg["literal_slots"]),
                false_token=int(sat_cfg["false_token"]),
                true_token=int(sat_cfg["true_token"]),
            )
        else:
            sat_clause = ce * 0.0
        if use_sudoku_loss:
            sudoku_constraint = sudoku_constraint_step(
                step.logits,
                board_size=int(sudoku_cfg.get("board_size", 4)),
                box_size=int(sudoku_cfg.get("box_size", 2)),
                digit_token_start=int(sudoku_cfg.get("digit_token_start", 1)),
            )
        else:
            sudoku_constraint = ce * 0.0

        ce_list.append(ce)
        kl_list.append(kl)
        halt_list.append(halt)
        lprm_list.append(lprm)
        graph_conflict_list.append(graph_conflict)
        graph_color_mass_list.append(graph_color_mass)
        nqueen_count_list.append(nqueen_count)
        nqueen_row_list.append(nqueen_row)
        nqueen_col_list.append(nqueen_col)
        nqueen_diag_list.append(nqueen_diag)
        sat_clause_list.append(sat_clause)
        sudoku_constraint_list.append(sudoku_constraint)

    ce_t = torch.stack(ce_list)
    kl_t = torch.stack(kl_list)
    halt_t = torch.stack(halt_list)
    lprm_t = torch.stack(lprm_list)
    graph_conflict_t = torch.stack(graph_conflict_list)
    graph_color_mass_t = torch.stack(graph_color_mass_list)
    nqueen_count_t = torch.stack(nqueen_count_list)
    nqueen_row_t = torch.stack(nqueen_row_list)
    nqueen_col_t = torch.stack(nqueen_col_list)
    nqueen_diag_t = torch.stack(nqueen_diag_list)
    sat_clause_t = torch.stack(sat_clause_list)
    sudoku_constraint_t = torch.stack(sudoku_constraint_list)
    acc_t = torch.stack(acc_list)

    total = (
        ce_t.mean()
        + beta_kl * kl_t.mean()
        + w_halt * halt_t.mean()
        + w_lprm * lprm_t.mean()
        + w_graph_conflict * graph_conflict_t.mean()
        + w_graph_color_mass * graph_color_mass_t.mean()
        + w_nqueen_count * nqueen_count_t.mean()
        + w_nqueen_row * nqueen_row_t.mean()
        + w_nqueen_col * nqueen_col_t.mean()
        + w_nqueen_diag * nqueen_diag_t.mean()
        + w_sat_clause * sat_clause_t.mean()
        + w_sudoku_constraint * sudoku_constraint_t.mean()
    )
    return GRAMLossBreakdown(
        ce_per_step=ce_t,
        kl_per_step=kl_t,
        halt_per_step=halt_t,
        lprm_per_step=lprm_t,
        graph_conflict_per_step=graph_conflict_t,
        graph_color_mass_per_step=graph_color_mass_t,
        nqueen_count_per_step=nqueen_count_t,
        nqueen_row_per_step=nqueen_row_t,
        nqueen_col_per_step=nqueen_col_t,
        nqueen_diag_per_step=nqueen_diag_t,
        sat_clause_per_step=sat_clause_t,
        sudoku_constraint_per_step=sudoku_constraint_t,
        prior_path_ce_per_step=torch.zeros_like(ce_t),
        valid_set_margin_per_step=torch.zeros_like(ce_t),
        valid_set_margin_gap_per_step=torch.zeros_like(ce_t),
        source_supported_valid_mode_lift_per_step=torch.zeros_like(ce_t),
        source_supported_valid_mode_delta_per_step=torch.zeros_like(ce_t),
        source_prior_exact_invalid_suppression_per_step=torch.zeros_like(ce_t),
        source_prior_exact_invalid_delta_per_step=torch.zeros_like(ce_t),
        valid_mode_bipartite_assignment_per_step=torch.zeros_like(ce_t),
        accuracy_per_step=acc_t,
        total=total,
    )
