"""Native GRU inference, without answer labels, filtering, repair or resampling."""
import argparse
import json
from pathlib import Path

import torch

from .io import check_weight_binding, content_crc32, read_config, validate_rows, write_json
from .models import build_maze, build_sat, encode


def generate(model, rows, task, seed, device="cpu"):
    validate_rows(rows, task)
    model.eval()
    torch.manual_seed(seed)
    predictions = []
    with torch.no_grad():
        for row in rows:
            # Match portable_v4: one question per forward, 32 ordered candidates.
            values = encode(row, 20, 91) if task == "sat" else row["x"]
            x = torch.tensor([values], device=device).repeat_interleave(32, 0)
            logits = model(x, 16)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite generated logits")
            confidence = logits.float().log_softmax(-1).amax(-1)
            if task == "maze":
                confidence = confidence[:, [v not in (2, 4) for v in row["x"]]]
            predictions.append({"id": row["id"], "candidates": logits.argmax(-1).cpu().tolist(),
                                "selector_scores": confidence.mean(-1).cpu().tolist()})
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Bind an archived weight to the reviewed model inventory")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026101201)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive; omit it for all rows")
    cfg = read_config(args.config)
    model_id = check_weight_binding(args.weights, args.manifest, cfg["task"]) if args.manifest else None
    rows = validate_rows(json.loads(args.rows.read_text()), cfg["task"])
    total = len(rows)
    if args.limit is not None:
        rows = rows[:args.limit]
    torch.set_num_threads(1)
    model = build_sat(device=args.device) if cfg["task"] == "sat" else build_maze(device=args.device)
    state = torch.load(args.weights, map_location=args.device, weights_only=True)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Expected tensor-only inference weights, not a training checkpoint")
    model.load_state_dict(state, strict=True)
    result = generate(model, rows, cfg["task"], args.seed, args.device)
    write_json(args.output, {
        "protocol": "portable-gru-inference-v1", "task": cfg["task"], "host": cfg["host"],
        "model_id": model_id,
        "K": 32, "depth": 16, "seed": args.seed, "device": args.device, "torch": torch.__version__,
        "questions_per_forward": 1, "population_rows": total, "generated_rows": len(rows),
        "weights_file": args.weights.name, "weights_crc32": content_crc32(args.weights),
        "input_crc32": content_crc32(args.rows), "labels_in_generation": False,
        "tool_assisted": False, "historical_cuda_bitwise_replay": False, "predictions": result,
    })
    print(json.dumps({"rows": len(result), "K": 32, "output": str(args.output)}))


if __name__ == "__main__":
    main()
