"""Score saved candidates and retain per-question identities; no model calls."""
import argparse
import json
from pathlib import Path

import torch

from .io import content_crc32, validate_rows, write_json
from .native_metrics import curve

FIELDS = ("U", "U_all", "V", "D", "V_target", "D_target", "validity", "first", "first_valid",
          "atleastone", "atleastone_target", "selected", "selected_valid", "D_per_K")


def read_predictions(path, task):
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if data.get("task") != task or data.get("K") != 32 or data.get("depth") != 16:
            raise ValueError("Prediction task, K or depth does not match the protocol")
        return data, [(x["id"], x["candidates"], x["selector_scores"]) for x in data["predictions"]]
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("K") != 32 or data.get("depth") != 16:
        raise ValueError("Saved tensor prediction requires K32 / depth16")
    expected = (len(data["ids"]), 32, 20 if task == "sat" else 100, 2 if task == "sat" else 4)
    if tuple(data["typed_logits"].shape) != expected or not torch.isfinite(data["typed_logits"]).all():
        raise ValueError("Invalid saved logit shape or finite-value check")
    if not torch.equal(data["native"], data["typed_logits"].argmax(-1)):
        raise ValueError("Saved native decisions are not the logit argmax")
    if tuple(data["selector_scores"].shape) != expected[:2]:
        raise ValueError("Selector shape mismatch")
    return data, list(zip(data["ids"], data["native"].tolist(), data["selector_scores"].tolist()))


def evaluate(task, rows, predictions, allow_subset=False):
    validate_rows(rows, task, evaluation=True)
    lookup = {r["id"]: r for r in rows}
    seen, result = set(), []
    for ident, candidates, scores in predictions:
        if type(ident) is not int or ident not in lookup or ident in seen:
            raise ValueError("Unknown, repeated or malformed prediction ID")
        seen.add(ident)
        values = curve(task, lookup[ident], candidates, scores)
        for row in values.values():
            row["legal_set"] = sorted(row["legal_set"])
            row["target_set"] = sorted(row["target_set"])
        result.append({"id": ident, "by_k": values})
    if not seen or (not allow_subset and seen != set(lookup)):
        raise ValueError("Incomplete prediction population; explicitly use --allow-subset for a small check")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("sat", "maze"), required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = json.loads(args.rows.read_text())
    meta, pred = read_predictions(args.predictions, args.task)
    if meta.get("input_crc32") and meta["input_crc32"] != content_crc32(args.rows):
        raise ValueError("Input content differs from the generation-time binding")
    result = evaluate(args.task, rows, pred, args.allow_subset)
    summary = {k: {field: sum(q["by_k"][k][field] for q in result) / len(result) for field in FIELDS}
               for k in (1, 2, 4, 8, 16, 32)}
    write_json(args.output, {"task": args.task, "K": 32, "depth": 16, "seed": meta.get("seed"),
                            "binding": {key: meta.get(key) for key in
                                        ("weights_crc32", "input_crc32", "protocol", "questions_per_forward")},
                            "population_rows": len(rows), "scored_rows": len(result), "summary": summary,
                            "scope": "saved candidates; zero new model calls; no confidence interval",
                            "selector": "mean maximum token log probability; Maze excludes walls and goal",
                            "per_question": result})
    print(json.dumps({"rows": len(result), "K32": summary[32]}))


if __name__ == "__main__":
    main()
