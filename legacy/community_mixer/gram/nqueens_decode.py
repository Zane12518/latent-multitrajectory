"""Structured decoders for N-Queens token logits."""

from __future__ import annotations

import torch

EMPTY = 1
QUEEN = 2


def queen_margin(logits: torch.Tensor) -> torch.Tensor:
    return logits[..., QUEEN] - logits[..., EMPTY]


def decode_nqueens_logits(
    logits: torch.Tensor,
    x: torch.Tensor,
    *,
    board_size: int,
    mode: str = "argmax",
    beam_size: int = 64,
) -> torch.Tensor:
    """Decode `(B, N*N, V)` logits into N-Queens token boards.

    `beam` is intended as a diagnostic / controlled evaluator: it enforces
    one queen per row, preserves input clues, and avoids column/diagonal
    conflicts when the beam can find a consistent assignment.
    """
    if mode == "argmax":
        return logits.argmax(dim=-1)
    if mode == "topk":
        return _decode_topk(logits, x, board_size)
    if mode == "row_top1":
        return _decode_row_top1(logits, x, board_size)
    if mode == "beam":
        return _decode_beam(logits, x, board_size, beam_size)
    raise ValueError(f"unknown N-Queens decode mode: {mode}")


def _decode_topk(logits: torch.Tensor, x: torch.Tensor, n: int) -> torch.Tensor:
    scores = queen_margin(logits)
    preds = torch.full_like(x, EMPTY)
    forced = x == QUEEN
    preds[forced] = QUEEN
    for b in range(x.shape[0]):
        need = max(0, n - int(forced[b].sum().item()))
        if need == 0:
            continue
        row_scores = scores[b].clone()
        row_scores[forced[b]] = -torch.inf
        take = torch.topk(row_scores, k=need).indices
        preds[b, take] = QUEEN
    return preds


def _decode_row_top1(logits: torch.Tensor, x: torch.Tensor, n: int) -> torch.Tensor:
    scores = queen_margin(logits)
    preds = torch.full_like(x, EMPTY)
    for b in range(x.shape[0]):
        board_scores = scores[b].reshape(n, n)
        board_input = x[b].reshape(n, n)
        board_pred = preds[b].reshape(n, n)
        for r in range(n):
            forced_cols = torch.nonzero(board_input[r] == QUEEN, as_tuple=False).flatten()
            if forced_cols.numel() > 0:
                board_pred[r, forced_cols[0]] = QUEEN
            else:
                c = int(board_scores[r].argmax().item())
                board_pred[r, c] = QUEEN
    return preds


def _decode_beam(logits: torch.Tensor, x: torch.Tensor, n: int, beam_size: int) -> torch.Tensor:
    scores = queen_margin(logits)
    fallback = _decode_row_top1(logits, x, n)
    preds = torch.full_like(x, EMPTY)
    for b in range(x.shape[0]):
        board_scores = scores[b].reshape(n, n)
        board_input = x[b].reshape(n, n)
        beams: list[tuple[float, frozenset[int], frozenset[int], frozenset[int], tuple[int, ...]]] = [
            (0.0, frozenset(), frozenset(), frozenset(), tuple())
        ]
        ok = True
        for r in range(n):
            forced_cols = torch.nonzero(board_input[r] == QUEEN, as_tuple=False).flatten().tolist()
            candidates = forced_cols if forced_cols else torch.argsort(board_scores[r], descending=True).tolist()
            next_beams = []
            for score, cols, diag1, diag2, positions in beams:
                for c in candidates:
                    c = int(c)
                    d1 = r - c
                    d2 = r + c
                    if c in cols or d1 in diag1 or d2 in diag2:
                        continue
                    next_beams.append(
                        (
                            score + float(board_scores[r, c].item()),
                            frozenset((*cols, c)),
                            frozenset((*diag1, d1)),
                            frozenset((*diag2, d2)),
                            (*positions, c),
                        )
                    )
            if not next_beams:
                ok = False
                break
            next_beams.sort(key=lambda row: row[0], reverse=True)
            beams = next_beams[:beam_size]
        if not ok or not beams:
            preds[b] = fallback[b]
            continue
        for r, c in enumerate(beams[0][4]):
            preds[b, r * n + c] = QUEEN
    return preds
