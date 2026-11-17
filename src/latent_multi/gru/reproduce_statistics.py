"""Recompute archived numeric summaries, without generation or new intervals."""
import argparse
import csv
from pathlib import Path

import numpy as np

from .io import write_json


def reproduce(metrics_path, changes_path, curves_path, table_path):
    with np.load(metrics_path, allow_pickle=False) as data:
        metrics = {k: data[k] for k in data.files}
    with np.load(changes_path, allow_pickle=False) as data:
        changes = {k: data[k] for k in data.files}
    for key in ("ids", "K", "rng"):
        if not np.array_equal(metrics[key], changes[key]):
            raise ValueError(f"Population, K or RNG mismatch: {key}")
        if len(np.unique(metrics[key])) != len(metrics[key]):
            raise ValueError(f"Duplicate {key}")
    fields, ks, seeds = (metrics[k].tolist() for k in ("fields", "K", "rng"))
    kinds, change_fields = (changes[k].tolist() for k in ("kinds", "fields"))
    models = [k for k in metrics if k not in ("ids", "K", "fields", "rng")]
    shape = (len(seeds), len(metrics["ids"]), len(ks), len(fields))
    for name in models:
        value = metrics[name]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid metrics: {name}")
        for u, v, d in (("U", "V_target", "D_target"), ("U_all", "V", "D")):
            if not np.array_equal(value[..., fields.index(u)], value[..., fields.index(v)] - value[..., fields.index(d)]):
                raise ValueError(f"Count identity failed: {name}, {u}")
    for name, value in changes.items():
        if "__minus__" not in name:
            continue
        candidate, control = name.split("__minus__")
        expected = shape[:3] + (len(kinds), len(change_fields))
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"Invalid changes: {name}")
        for kind, u in (("target", "U"), ("all_legal", "U_all")):
            selected = value[..., kinds.index(kind), :]
            added, lost, retained = (selected[..., change_fields.index(f)] for f in ("added", "lost", "retained"))
            if not (np.array_equal(added + retained, metrics[candidate][..., fields.index(u)])
                    and np.array_equal(lost + retained, metrics[control][..., fields.index(u)])):
                raise ValueError(f"Set-count identity failed: {name}, {kind}")
    counts, maximum = {"K_curve_rows": 0, "set_change_rows": 0}, 0.0
    with Path(curves_path).open() as handle:
        for row in csv.DictReader(handle):
            value = metrics[row["model"]]
            value = value.mean(axis=0) if row["evaluation_rng"] == "mean_of_three" else value[seeds.index(int(row["evaluation_rng"]))]
            mean = value[:, ks.index(int(row["K"])), :].mean(axis=0)
            error = float(np.max([abs(float(mean[i]) - float(row[f])) for i, f in enumerate(fields)]))
            if not np.isfinite(error) or error > 1e-10:
                raise ValueError("K-curve table differs from numeric arrays")
            maximum = max(maximum, error)
            counts["K_curve_rows"] += 1
    with Path(table_path).open() as handle:
        for row in csv.DictReader(handle):
            value = changes[row["candidate"] + "__minus__" + row["control"]]
            mean = value.mean(axis=(0, 1))[ks.index(int(row["K"])), kinds.index(row["set_kind"])]
            error = float(np.max([abs(float(mean[i]) - float(row[f])) for i, f in enumerate(change_fields)]))
            if not np.isfinite(error) or error > 1e-10:
                raise ValueError("Set-change table differs from numeric arrays")
            maximum = max(maximum, error)
            counts["set_change_rows"] += 1
    if not models or not all(counts.values()):
        raise ValueError("Empty model population or comparison tables")
    return {"scope": "saved numeric arrays only; no raw identities, generation or interval recomputation",
            "questions": len(metrics["ids"]), "models": len(models), "evaluation_rng": seeds,
            "K": ks, **counts, "maximum_absolute_difference": maximum,
            "count_identities": "PASS", "table_values": "PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("sat", "maze"), required=True)
    parser.add_argument("--statistics", type=Path, required=True, help="Task directory with two NPZ files")
    parser.add_argument("--tables", type=Path, required=True, help="Directory with public task CSV files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reproduce(args.statistics / "per_rng_metrics.npz", args.statistics / "per_rng_solution_changes.npz",
                       args.tables / f"{args.task}_all_models_K_curves.csv",
                       args.tables / f"{args.task}_legal_solution_changes.csv")
    write_json(args.output, {"task": args.task, **result})
    print(f"{args.task}: {result['K_curve_rows']} curve rows, {result['set_change_rows']} change rows PASS")


if __name__ == "__main__":
    main()
