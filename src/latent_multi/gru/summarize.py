"""Average evaluation RNG within questions; optional matched finite-pool changes."""
import argparse
import json
from pathlib import Path

from .evaluate import FIELDS
from .io import write_json
from .native_metrics import churn


def load_group(paths):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    if not runs or any(r.get("K") != 32 or r.get("depth") != 16 for r in runs):
        raise ValueError("Need at least one K32/depth16 scored run")
    if len({r["task"] for r in runs}) != 1:
        raise ValueError("Mixed tasks")
    seeds = [r["seed"] for r in runs]
    if None in seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation RNG identities must be present and distinct")
    if any(r.get("binding") != runs[0].get("binding") for r in runs):
        raise ValueError("RNG repeats must use the same model, input and generation protocol")
    groups = [{r["id"]: r for r in run["per_question"]} for run in runs]
    for run, group in zip(runs, groups):
        if len(group) != len(run["per_question"]) or set(group) != set(groups[0]):
            raise ValueError("Question identity mismatch")
    return runs, groups


def summarize(paths, baseline_paths=None):
    runs, groups = load_group(paths)
    ids = sorted(groups[0])
    output = {"task": runs[0]["task"], "questions": len(ids), "evaluation_rng": [r["seed"] for r in runs],
              "binding": runs[0].get("binding"),
              "model_input_binding_present": bool(runs[0].get("binding", {}).get("weights_crc32") and runs[0].get("binding", {}).get("input_crc32")),
              "scope": "RNG averaged within question; descriptive fixed-model means, no training-seed inference"}
    output["means"] = {k: {f: sum(g[i]["by_k"][k][f] for g in groups for i in ids) / (len(groups) * len(ids))
                           for f in FIELDS} for k in ("1", "2", "4", "8", "16", "32")}
    if baseline_paths:
        base_runs, base_groups = load_group(baseline_paths)
        if (base_runs[0]["task"] != runs[0]["task"] or set(base_groups[0]) != set(ids)
                or {r["seed"] for r in base_runs} != {r["seed"] for r in runs}):
            raise ValueError("Baseline task, population or evaluation RNG does not match")
        a_binding, b_binding = runs[0].get("binding", {}), base_runs[0].get("binding", {})
        for key in ("input_crc32", "protocol", "questions_per_forward"):
            if a_binding.get(key) != b_binding.get(key):
                raise ValueError("Baseline input content or generation protocol differs")
        base = {r["seed"]: g for r, g in zip(base_runs, base_groups)}
        def identities(values):
            return [tuple(v) if isinstance(v, list) else v for v in values]
        differences = []
        for ident in ids:
            by_k = {}
            for k in ("1", "2", "4", "8", "16", "32"):
                delta = {f: 0. for f in FIELDS}
                changes = {kind: {f: 0. for f in ("added", "lost", "retained")} for kind in ("legal_set", "target_set")}
                for run, group in zip(runs, groups):
                    a, b = group[ident]["by_k"][k], base[run["seed"]][ident]["by_k"][k]
                    for f in FIELDS:
                        delta[f] += (a[f] - b[f]) / len(groups)
                    for kind in changes:
                        for f, value in churn(identities(a[kind]), identities(b[kind])).items():
                            changes[kind][f] += value / len(groups)
                by_k[k] = {"delta": delta, "finite_pool_changes": changes}
            differences.append({"id": ident, "by_k": by_k})
        output["paired_questions"] = differences
        output["delta_means"] = {k: {f: sum(r["by_k"][k]["delta"][f] for r in differences) / len(ids)
                                     for f in FIELDS} for k in output["means"]}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, summarize(args.metrics, args.baseline))
    print(json.dumps({"output": str(args.output)}))


if __name__ == "__main__":
    main()
