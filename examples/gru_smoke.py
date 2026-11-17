"""Two synthetic tasks; optionally load approved GRU weights supplied by the caller."""
import argparse
from pathlib import Path

import torch

from latent_multi.gru.evaluate import evaluate
from latent_multi.gru.io import write_json
from latent_multi.gru.models import build_maze, build_sat
from latent_multi.gru.predict import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cells = [2] * 100
    cells[0], cells[1], cells[10], cells[11] = 3, 1, 1, 4
    data = {"sat": [{"id": 0, "n": 20, "clauses": [[1, 2, -3], [-1, 4, 5]]}],
            "maze": [{"id": 0, "x": cells, "optimal": 2}]}
    report = {"scope": "synthetic CPU interface smoke, not benchmark performance",
              "weights": "caller supplied" if args.weights_dir else "random untrained initialization", "tasks": {}}
    for task, rows in data.items():
        torch.manual_seed(43)
        model = build_sat() if task == "sat" else build_maze()
        if args.weights_dir:
            name = "main_sat74_a_multi.pt" if task == "sat" else "main_maze77_a_multi.pt"
            model.load_state_dict(torch.load(args.weights_dir / name, weights_only=True, map_location="cpu"), strict=True)
        predictions = generate(model, rows, task, 43)
        metrics = evaluate(task, rows, [(p["id"], p["candidates"], p["selector_scores"]) for p in predictions])
        write_json(args.out / f"{task}_rows.json", rows)
        write_json(args.out / f"{task}_predictions.json", {"task": task, "K": 32, "depth": 16, "seed": 43,
                                                         "predictions": predictions})
        report["tasks"][task] = metrics[0]["by_k"][32]
    write_json(args.out / "result.json", report)
    print({task: {k: values[k] for k in ("U", "validity", "selected")} for task, values in report["tasks"].items()})


if __name__ == "__main__":
    main()
