"""Structured decoders for Maze shortest-path logits."""

from __future__ import annotations

from collections import deque

import torch

EMPTY = 1
WALL = 2
START = 3
GOAL = 4
PATH = 5


def decode_maze_logits(
    logits: torch.Tensor,
    x: torch.Tensor,
    *,
    grid_size: int,
    mode: str = "argmax",
    max_paths: int = 512,
) -> torch.Tensor:
    """Decode `(B, N*N, V)` Maze logits into token boards."""
    if mode == "argmax":
        return logits.argmax(dim=-1)
    preds = []
    for b in range(logits.shape[0]):
        inp = x[b].detach().cpu().tolist()
        if mode == "topk_len":
            pred = _decode_topk_len(logits[b], inp, grid_size)
        elif mode == "shortest_path":
            pred = _decode_shortest_path(logits[b], inp, grid_size, max_paths=max_paths)
        else:
            raise ValueError(f"unknown Maze decode mode: {mode}")
        preds.append(torch.tensor(pred, dtype=torch.long, device=logits.device))
    return torch.stack(preds, dim=0)


def _neighbors(pos: int, n: int) -> list[int]:
    r, c = divmod(pos, n)
    out = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n:
            out.append(rr * n + cc)
    return out


def _find_endpoints(inp: list[int]) -> tuple[int, int]:
    starts = [i for i, t in enumerate(inp) if t == START]
    goals = [i for i, t in enumerate(inp) if t == GOAL]
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError("maze input must contain exactly one start and one goal")
    return starts[0], goals[0]


def _distances(walls: set[int], start: int, n: int) -> dict[int, int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        pos = q.popleft()
        for nb in _neighbors(pos, n):
            if nb in walls or nb in dist:
                continue
            dist[nb] = dist[pos] + 1
            q.append(nb)
    return dist


def _shortest_paths(
    walls: set[int],
    start: int,
    goal: int,
    n: int,
    *,
    max_paths: int,
) -> list[tuple[int, ...]]:
    dist = _distances(walls, start, n)
    if goal not in dist:
        return []
    shortest = dist[goal]
    out: list[tuple[int, ...]] = []
    cur = [start]

    def dfs(pos: int) -> None:
        if len(out) >= max_paths:
            return
        if pos == goal:
            out.append(tuple(cur))
            return
        for nb in _neighbors(pos, n):
            if nb in walls:
                continue
            if dist.get(nb) != dist[pos] + 1:
                continue
            if dist[nb] > shortest:
                continue
            cur.append(nb)
            dfs(nb)
            cur.pop()

    dfs(start)
    return out


def _path_to_tokens(inp: list[int], path: tuple[int, ...]) -> list[int]:
    pred = list(inp)
    for i, tok in enumerate(pred):
        if tok == PATH:
            pred[i] = EMPTY
    for pos in path[1:-1]:
        pred[pos] = PATH
    return pred


def _path_margin(logits: torch.Tensor) -> torch.Tensor:
    return (logits[:, PATH] - logits[:, EMPTY]).float().detach().cpu()


def _decode_topk_len(logits: torch.Tensor, inp: list[int], n: int) -> list[int]:
    start, goal = _find_endpoints(inp)
    walls = {i for i, t in enumerate(inp) if t == WALL}
    dist = _distances(walls, start, n)
    if goal not in dist:
        return list(inp)
    k = max(0, dist[goal] - 1)
    scores = _path_margin(logits)
    pred = list(inp)
    for pos in walls | {start, goal}:
        scores[pos] = -torch.inf
    if k > 0:
        finite = int(torch.isfinite(scores).sum().item())
        take = torch.topk(scores, k=min(k, finite)).indices.tolist()
        for pos in take:
            pred[int(pos)] = PATH
    return pred


def _decode_shortest_path(logits: torch.Tensor, inp: list[int], n: int, *, max_paths: int) -> list[int]:
    start, goal = _find_endpoints(inp)
    walls = {i for i, t in enumerate(inp) if t == WALL}
    paths = _shortest_paths(walls, start, goal, n, max_paths=max_paths)
    if not paths:
        return list(inp)
    scores = _path_margin(logits)
    best = max(paths, key=lambda path: sum(float(scores[pos]) for pos in path[1:-1]))
    return _path_to_tokens(inp, best)
