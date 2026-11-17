"""N-Queens dataset loader.

Reads TSV files produced by
`research_compiler/research/2026_05_21_gram/state/scripts/build_nqueens_dataset.py`.

TSV schema (header on line 1):
  puzzle_id <TAB> input_tokens <TAB> target_tokens <TAB> n_solutions <TAB> all_completions_compact

- input_tokens / target_tokens: space-separated ints (1=empty, 2=queen),
  length = N²  (64 for 8×8).
- all_completions_compact: '|'-separated list of N²-int space-separated strings.

This module exposes:
  - NQueensItem: dataclass holding one row (tensors + metadata).
  - NQueensDataset: torch.utils.data.Dataset returning (input, target) for training,
    with the n_solutions / all_completions kept on the item for eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class NQueensItem:
    puzzle_id: int
    input_tokens: torch.Tensor      # (L,) long
    target_tokens: torch.Tensor     # (L,) long
    n_solutions: int
    all_completions: torch.Tensor   # (n_solutions, L) long; eval-time use


def _parse_token_seq(s: str) -> list[int]:
    return [int(t) for t in s.split(" ") if t]


def _parse_completions(s: str, n_expected_tokens: int) -> list[list[int]]:
    if not s:
        return []
    comps = []
    for part in s.split("|"):
        toks = _parse_token_seq(part)
        assert len(toks) == n_expected_tokens, (
            f"completion has {len(toks)} tokens, expected {n_expected_tokens}"
        )
        comps.append(toks)
    return comps


class NQueensDataset(Dataset):
    """Loads one TSV file into memory (small datasets; 8×8 has ~5K rows max).

    __getitem__ returns a (input_long, target_long) tuple of (L,) tensors —
    the minimum needed for training. Use `item(i)` to get the full NQueensItem
    including completions for eval-time coverage scoring.
    """

    def __init__(self, tsv_path: str | Path, expected_seq_len: int | None = None):
        self.tsv_path = Path(tsv_path)
        if not self.tsv_path.exists():
            raise FileNotFoundError(self.tsv_path)
        self._items: list[NQueensItem] = []

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
                puzzle_id = int(fields[0])
                inp = _parse_token_seq(fields[1])
                tgt = _parse_token_seq(fields[2])
                n_sols = int(fields[3])
                if expected_seq_len is not None:
                    if len(inp) != expected_seq_len or len(tgt) != expected_seq_len:
                        raise ValueError(
                            f"seq_len mismatch in {self.tsv_path}: "
                            f"input={len(inp)} target={len(tgt)} expected={expected_seq_len}"
                        )
                comps = _parse_completions(fields[4], n_expected_tokens=len(inp))
                self._items.append(
                    NQueensItem(
                        puzzle_id=puzzle_id,
                        input_tokens=torch.tensor(inp, dtype=torch.long),
                        target_tokens=torch.tensor(tgt, dtype=torch.long),
                        n_solutions=n_sols,
                        all_completions=torch.tensor(comps, dtype=torch.long),
                    )
                )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        it = self._items[idx]
        return it.input_tokens, it.target_tokens

    def item(self, idx: int) -> NQueensItem:
        """Full record including completions; used by eval / coverage scoring."""
        return self._items[idx]

    def seq_len(self) -> int:
        return self._items[0].input_tokens.shape[0] if self._items else 0
