"""Trajectory-level duplicate metrics for multi-sample latent reasoning evals."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable


def parse_sample_sizes(text: str) -> list[int]:
    """Parse a comma-separated sample-size list, preserving sorted unique values."""
    sizes = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not sizes:
        raise ValueError("sample size list is empty")
    if sizes[0] <= 0:
        raise ValueError(f"sample sizes must be positive: {sizes}")
    return sizes


def safe_div(num: float, denom: float) -> float | None:
    return None if denom == 0 else num / denom


def valid_ndr(n_valid: int, n_unique_valid: int) -> float | None:
    """Valid normalized duplicate rate.

    Undefined when fewer than two valid trajectories are present, because a
    single valid sample cannot be duplicated among valid samples.
    """
    if n_valid < 2:
        return None
    return (n_valid - n_unique_valid) / (n_valid - 1)


def compute_prefix_metrics(
    *,
    puzzle_id: int,
    n_solutions: int,
    predictions: list[tuple[int, ...]],
    valid_set: set[tuple[int, ...]],
    sample_sizes: Iterable[int],
) -> list[dict]:
    """Compute per-puzzle duplicate metrics for each prefix size in sample_sizes."""
    rows: list[dict] = []
    for n_samples in sample_sizes:
        prefix = predictions[:n_samples]
        if len(prefix) != n_samples:
            raise ValueError(
                f"requested N={n_samples}, but only {len(prefix)} predictions are available"
            )
        valid_preds = [pred for pred in prefix if pred in valid_set]
        unique_valid = set(valid_preds)
        n_valid = len(valid_preds)
        n_unique_valid = len(unique_valid)
        n_invalid = n_samples - n_valid
        n_duplicated_valid = n_valid - n_unique_valid
        rows.append(
            {
                "puzzle_id": puzzle_id,
                "n_solutions": n_solutions,
                "n_samples": n_samples,
                "n_valid": n_valid,
                "n_unique_valid": n_unique_valid,
                "n_invalid": n_invalid,
                "n_duplicated_valid": n_duplicated_valid,
                "validity_rate": safe_div(n_valid, n_samples),
                "valid_ndr": valid_ndr(n_valid, n_unique_valid),
                "coverage": safe_div(n_unique_valid, n_solutions),
                "wasted_compute_rate": safe_div(n_invalid + n_duplicated_valid, n_samples),
            }
        )
    return rows


def aggregate_prefix_metrics(rows: Iterable[dict]) -> list[dict]:
    """Aggregate per-puzzle rows into one summary row per sample size."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[int(row["n_samples"])].append(row)

    out: list[dict] = []
    for n_samples in sorted(groups):
        group = groups[n_samples]
        n_examples = len(group)
        total_predictions = sum(int(row["n_samples"]) for row in group)
        total_valid = sum(int(row["n_valid"]) for row in group)
        total_unique_valid = sum(int(row["n_unique_valid"]) for row in group)
        total_possible_valid = sum(int(row["n_solutions"]) for row in group)
        total_invalid = sum(int(row["n_invalid"]) for row in group)
        total_duplicated_valid = sum(int(row["n_duplicated_valid"]) for row in group)

        ndr_rows = [row for row in group if row["valid_ndr"] is not None]
        weighted_ndr_num = sum(
            int(row["n_valid"]) - int(row["n_unique_valid"]) for row in ndr_rows
        )
        weighted_ndr_den = sum(int(row["n_valid"]) - 1 for row in ndr_rows)

        out.append(
            {
                "n_samples": n_samples,
                "n_examples": n_examples,
                "total_predictions": total_predictions,
                "total_valid_predictions": total_valid,
                "total_unique_valid_hits": total_unique_valid,
                "total_possible_valid": total_possible_valid,
                "total_invalid_predictions": total_invalid,
                "total_duplicated_valid_predictions": total_duplicated_valid,
                "validity_rate": safe_div(total_valid, total_predictions),
                "coverage": safe_div(total_unique_valid, total_possible_valid),
                "micro_wasted_compute_rate": safe_div(
                    total_invalid + total_duplicated_valid, total_predictions
                ),
                "mean_validity_rate": safe_div(
                    sum(float(row["validity_rate"]) for row in group), n_examples
                ),
                "mean_unique_valid": safe_div(total_unique_valid, n_examples),
                "mean_wasted_compute_rate": safe_div(
                    sum(float(row["wasted_compute_rate"]) for row in group), n_examples
                ),
                "mean_valid_ndr": safe_div(
                    sum(float(row["valid_ndr"]) for row in ndr_rows), len(ndr_rows)
                ),
                "weighted_valid_ndr": safe_div(weighted_ndr_num, weighted_ndr_den),
                "valid_ndr_defined_examples": len(ndr_rows),
            }
        )
    return out
