"""Maze / shortest-path dataset loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class MazeItem:
    puzzle_id: int
    input_tokens: torch.Tensor
    target_tokens: torch.Tensor
    n_solutions: int
    all_completions: torch.Tensor


def _parse_token_seq(s: str) -> list[int]:
    return [int(t) for t in s.split(" ") if t]


def _parse_completions(s: str, n_expected_tokens: int) -> list[list[int]]:
    if not s:
        return []
    out = []
    for part in s.split("|"):
        toks = _parse_token_seq(part)
        if len(toks) != n_expected_tokens:
            raise ValueError(
                f"completion has {len(toks)} tokens, expected {n_expected_tokens}"
            )
        out.append(toks)
    return out


class MazeDataset(Dataset):
    """Loads one Maze TSV file into memory."""

    def __init__(self, tsv_path: str | Path, expected_seq_len: int | None = None):
        self.tsv_path = Path(tsv_path)
        if not self.tsv_path.exists():
            raise FileNotFoundError(self.tsv_path)
        self._items: list[MazeItem] = []
        with self.tsv_path.open() as f:
            header = f.readline().rstrip("\n").split("\t")
            expected_header = [
                "puzzle_id",
                "input_tokens",
                "target_tokens",
                "n_solutions",
                "all_completions_compact",
            ]
            if header != expected_header:
                raise ValueError(f"unexpected TSV header in {self.tsv_path}: {header}")
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 5:
                    raise ValueError(f"malformed row in {self.tsv_path}: {fields!r}")
                inp = _parse_token_seq(fields[1])
                tgt = _parse_token_seq(fields[2])
                if expected_seq_len is not None:
                    if len(inp) != expected_seq_len or len(tgt) != expected_seq_len:
                        raise ValueError(
                            f"seq_len mismatch in {self.tsv_path}: "
                            f"input={len(inp)} target={len(tgt)} expected={expected_seq_len}"
                        )
                comps = _parse_completions(fields[4], n_expected_tokens=len(inp))
                self._items.append(
                    MazeItem(
                        puzzle_id=int(fields[0]),
                        input_tokens=torch.tensor(inp, dtype=torch.long),
                        target_tokens=torch.tensor(tgt, dtype=torch.long),
                        n_solutions=int(fields[3]),
                        all_completions=torch.tensor(comps, dtype=torch.long),
                    )
                )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        it = self._items[idx]
        return it.input_tokens, it.target_tokens

    def item(self, idx: int) -> MazeItem:
        return self._items[idx]
