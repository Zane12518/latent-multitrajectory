"""Tests only for the newly integrated GRU adapter; no historical retraining."""
import json
import math
from pathlib import Path

import pytest
import torch

from latent_multi.gru.evaluate import evaluate, read_predictions
from latent_multi.gru.io import check_weight_binding, content_crc32, read_config, validate_rows
from latent_multi.gru.models import build_maze, build_sat
from latent_multi.gru.native_metrics import check_metric_identities
from latent_multi.gru.objective import auxiliary, maze_exact_logmass, score_refs, subset
from latent_multi.gru.predict import generate
from latent_multi.gru.prepare_satlib import parse_dimacs
from latent_multi.gru.summarize import summarize

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def toy_maze():
    cells = [2] * 100
    cells[0], cells[1], cells[10], cells[11] = 3, 1, 1, 4
    return {"id": 0, "x": cells, "optimal": 2}


def test_saved_metric_identities():
    assert check_metric_identities()["status"] == "PASS"


@pytest.mark.parametrize("m", [0, 1, 7, 32])
def test_allocation_value_and_gradient(m):
    generator = torch.Generator().manual_seed(41)
    scores = torch.randn(32, m, generator=generator, dtype=torch.float64, requires_grad=True)
    multi, assigned, weights, error = auxiliary(scores, 20)
    assert not weights.requires_grad
    grad = torch.autograd.grad(assigned, scores, retain_graph=True)[0]
    # Autograd multiplies by the reciprocal; direct division can round differently.
    torch.testing.assert_close(grad, -weights / 20, rtol=4 * torch.finfo(scores.dtype).eps, atol=0)
    if m:
        assert abs(float(weights.sum()) - 1) < 1e-10
        assert error["col_err"] < 1e-10
    if m == 1:
        assert torch.equal(multi, assigned)
        assert torch.equal(grad, torch.autograd.grad(multi, scores)[0])


def test_reference_rotation_and_deduplication():
    bank = list(range(80)) + list(range(10))
    first = subset(bank, 9, 0, cap=8)
    second = subset(bank, 9, 1, cap=8)
    assert len(set(first)) == 8 and not set(first) & set(second)
    assert first == subset(list(reversed(bank)), 9, 0, cap=8)


def test_maze_complete_scores_and_dag():
    lp = torch.full((32, 100, 4), -math.log(4), dtype=torch.float64, requires_grad=True)
    row = toy_maze()
    references = (((0, 1), (3, 1)), ((0, 10), (1, 3)))
    scores = score_refs(lp, references, "maze")
    assert torch.allclose(scores, torch.full_like(scores, -math.log(16)))
    mass = maze_exact_logmass(lp, row)
    assert torch.allclose(mass, scores.logsumexp(1))
    assert torch.isfinite(torch.autograd.grad(mass.sum(), lp)[0]).all()


@pytest.mark.parametrize("task", ["sat", "maze"])
def test_native_generation_is_reproducible_and_ignores_references(task):
    model = build_sat() if task == "sat" else build_maze()
    row = {"id": 0, "n": 20, "clauses": [[1, 2, -3]]} if task == "sat" else toy_maze()
    a = generate(model, [row], task, 43)
    poisoned = {**row, "solution_bits": ["not used"], "path_indices": ["not used"], "optimal_actions": ["not used"]}
    b = generate(model, [poisoned], task, 43)
    assert a == b
    assert len(a[0]["candidates"]) == 32
    measured = evaluate(task, [row], [(0, a[0]["candidates"], a[0]["selector_scores"])])
    assert measured[0]["by_k"][32]["V"] == measured[0]["by_k"][32]["D"] + measured[0]["by_k"][32]["U_all"]


def test_missing_duplicate_and_changed_data_rejected():
    row = toy_maze()
    pred = (0, [[3] * 100] * 32, [0.] * 32)
    with pytest.raises(ValueError):
        evaluate("maze", [row, {**row, "id": 1}], [pred])
    with pytest.raises(ValueError):
        evaluate("maze", [row], [pred, pred])
    with pytest.raises(ValueError):
        validate_rows([{**row, "optimal": 3}], "maze", evaluation=True)
    with pytest.raises(ValueError):
        validate_rows([{"id": 0, "clauses": [[1, -1, 2]]}], "sat")


def test_prediction_protocol_and_summary(tmp_path):
    row = toy_maze()
    good = [3] * 100
    good[1] = 1
    scored = evaluate("maze", [row], [(0, [good] * 32, [0.] * 32)])
    paths = []
    for seed in [10, 11, 12]:
        path = tmp_path / f"metrics{seed}.json"
        path.write_text(json.dumps({"task": "maze", "K": 32, "depth": 16, "seed": seed, "per_question": scored}))
        paths.append(path)
    summary = summarize(paths, paths)
    assert summary["means"]["32"]["U"] == 1
    assert summary["delta_means"]["32"]["U"] == 0
    assert summary["paired_questions"][0]["by_k"]["32"]["finite_pool_changes"]["target_set"]["retained"] == 1
    with pytest.raises(ValueError):
        summarize([paths[0], paths[0]])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"task": "maze", "K": 2, "depth": 16, "predictions": []}))
    with pytest.raises(ValueError):
        read_predictions(bad, "maze")


def test_dimacs_and_config():
    formula = "c synthetic\np cnf 20 91\n" + "1 -2 3 0\n" * 91 + "%\n0\n"
    assert len(parse_dimacs(formula)) == 91
    with pytest.raises(ValueError):
        parse_dimacs(formula.replace("20 91", "20 90"))
    assert read_config(ROOT / "configs/gru/satlib20.json")["host"] == "sat20_gru"
    assert read_config(ROOT / "configs/gru/maze10.json")["host"] == "maze10_shared_gru"


def test_weight_content_binding(tmp_path):
    weights = tmp_path / "example.pt"
    weights.write_bytes(b"synthetic file, not a real checkpoint")
    manifest = tmp_path / "manifest.json"
    record = {"model_id": "synthetic", "file": "weights/example.pt", "bytes": weights.stat().st_size,
              "crc32": content_crc32(weights), "task": "maze"}
    manifest.write_text(json.dumps({"models": [record]}))
    assert check_weight_binding(weights, manifest, "maze") == "synthetic"
    with pytest.raises(ValueError):
        check_weight_binding(weights, manifest, "sat")
    weights.write_bytes(b"Synthetic file, not a real checkpoint")
    with pytest.raises(ValueError):
        check_weight_binding(weights, manifest, "maze")


def test_different_model_repeats_rejected(tmp_path):
    paths = []
    for seed in (10, 11):
        path = tmp_path / f"{seed}.json"
        path.write_text(json.dumps({"task": "maze", "K": 32, "depth": 16, "seed": seed,
                                    "binding": {"weights_crc32": str(seed), "input_crc32": "same"},
                                    "per_question": []}))
        paths.append(path)
    with pytest.raises(ValueError, match="same model"):
        summarize(paths)
