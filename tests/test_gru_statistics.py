"""Small positive/corruption tests for saved-array reproduction, no model calls."""
import csv

import numpy as np
import pytest

from latent_multi.gru.reproduce_statistics import reproduce


def fixture(tmp_path):
    fields = ["U", "V_target", "D_target", "U_all", "V", "D"]
    metrics = np.array([[[[1., 2., 1., 1., 2., 1.]]]])
    common = {"ids": [1], "K": [32], "rng": [7]}
    m, c = tmp_path / "m.npz", tmp_path / "c.npz"
    np.savez(m, **common, fields=fields, model=metrics)
    np.savez(c, **common, kinds=["target", "all_legal"], fields=["added", "lost", "retained"],
             model__minus__model=np.array([[[[[0., 0., 1.], [0., 0., 1.]]]]]))
    curves, changes = tmp_path / "curves.csv", tmp_path / "changes.csv"
    with curves.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "evaluation_rng", "K"] + fields)
        writer.writeheader()
        writer.writerow({"model": "model", "evaluation_rng": 7, "K": 32, **dict(zip(fields, metrics.ravel()))})
    with changes.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate", "control", "K", "set_kind", "added", "lost", "retained"])
        writer.writeheader()
        for kind in ("target", "all_legal"):
            writer.writerow({"candidate": "model", "control": "model", "K": 32, "set_kind": kind,
                             "added": 0, "lost": 0, "retained": 1})
    return m, c, curves, changes


def test_numeric_reproduction(tmp_path):
    result = reproduce(*fixture(tmp_path))
    assert result["maximum_absolute_difference"] == 0
    assert result["K_curve_rows"] == 1 and result["set_change_rows"] == 2


def test_corrupted_counts_rejected(tmp_path):
    paths = fixture(tmp_path)
    with np.load(paths[0]) as source:
        data = {key: source[key] for key in source.files}
    data["model"][..., 0] += 1
    np.savez(paths[0], **data)
    with pytest.raises(ValueError, match="Count identity"):
        reproduce(*paths)


def test_nonfinite_table_rejected(tmp_path):
    paths = fixture(tmp_path)
    with paths[2].open() as f:
        rows = list(csv.DictReader(f))
    rows[0]["D"] = "nan"
    with paths[2].open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="table differs"):
        reproduce(*paths)
