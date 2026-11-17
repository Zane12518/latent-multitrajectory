"""GRAM evaluation entry point — accuracy + coverage on a TSV split.

Usage:
    gram-eval --checkpoint runs/nqueens_8x8/seed42/final.pt \
              --split /path/to/cache/datasets/nqueens_8x8/test.tsv \
              --n-samples 20 \
              --output runs/nqueens_8x8/seed42/eval_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .datasets import load_dataset
from .ema import EMA
from .losses import exact_match_accuracy
from .model import GRAMModel


def _build_model_from_cfg(cfg: dict) -> GRAMModel:
    task = cfg["task"]
    m = cfg["model"]
    return GRAMModel(
        vocab_size=int(task["vocab_size"]),
        seq_len=int(task["seq_len"]),
        d_model=int(m["d_model"]),
        n_heads=int(m["n_heads"]),
        d_ff=int(m["ff_hidden"]),
        fL_layers=int(m["fL_layers"]),
        fH_layers=int(m["fH_layers"]),
        K_inner=int(m["K_inner"]),
        T_outer=int(m["T_outer"]),
        nsup=int(m["nsup"]),
        position_aware_token_embedding=bool(
            m.get("position_aware_token_embedding", False)
        ),
        recursive_residual_scale_exponent=float(
            m.get("recursive_residual_scale_exponent", 0.0)
        ),
        guidance_mode=str(m.get("guidance_mode", "full")),
    )


def select_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def _normalize_prediction(dataset, pred: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    normalizer = getattr(dataset, "normalize_prediction", None)
    if normalizer is None:
        return tuple(pred)
    return normalizer(pred)


@torch.no_grad()
def evaluate(
    model: GRAMModel,
    dataset,
    n_samples: int,
    device: torch.device,
    batch_size: int = 64,
    accuracy_samples: int = 1,
) -> dict:
    """Compute mean-per-draw accuracy + coverage.

    `accuracy_samples=1` preserves the historical evaluation behavior.
    The repo-matched Table 3b gate uses five independent draws and reports
    their arithmetic mean.  That aggregation is an explicit repository
    assumption, not a claim about the unpublished paper implementation.

    Two accuracy metrics are reported per output JSON:

      - `accuracy_valid_solution` (HEADLINE, matches paper Table 1 N-Queens
        accuracy definition: "output satisfies all constraints"). We check
        whether the argmax decoding lies in `dataset.item(i).all_completions`
        — the dataset enumerates every valid completion per puzzle, so
        membership IS constraint satisfaction.

      - `accuracy_target_exact_match` (archived, kept for continuity with
        the original RES-1 numbers). Argmax must exactly match the SINGLE
        `target_tokens` row that was carved from. On multi-solution puzzles
        the model can produce a different but equally valid completion and
        be marked wrong by this metric.

    Coverage is the same as before: across `n_samples` independent inference
    draws per puzzle, count how many unique elements of `all_completions`
    were hit. This is already a constraint-satisfaction metric.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if accuracy_samples < 1:
        raise ValueError(
            f"accuracy_samples must be positive, got {accuracy_samples}"
        )
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Accuracy pass over independent draws.  We average the per-draw
    # constraint-valid and target-exact indicators.
    correct_target = 0
    correct_valid = 0
    total_examples = 0
    total_accuracy_draws = 0
    # Pre-build per-puzzle valid sets once. Faster than rebuilding inside the
    # loop for each batch.
    valid_sets: list[set[tuple[int, ...]]] = []
    for i in range(len(dataset)):
        item = dataset.item(i)
        valid_sets.append({tuple(c.tolist()) for c in item.all_completions})

    idx = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        batch_examples, seq_len = x.shape
        x_draws = (
            x.unsqueeze(1)
            .expand(-1, accuracy_samples, -1)
            .reshape(batch_examples * accuracy_samples, seq_len)
        )
        y_draws = (
            y.unsqueeze(1)
            .expand(-1, accuracy_samples, -1)
            .reshape(batch_examples * accuracy_samples, seq_len)
        )
        outputs = model(x_draws, y=None)
        final_logits = outputs[-1].logits
        preds = final_logits.argmax(dim=-1).reshape(
            batch_examples, accuracy_samples, seq_len
        )

        correct_target += int(
            exact_match_accuracy(final_logits, y_draws).sum().item()
        )
        for b in range(batch_examples):
            for draw in range(accuracy_samples):
                pred = _normalize_prediction(dataset, preds[b, draw].tolist())
                if pred in valid_sets[idx + b]:
                    correct_valid += 1
        idx += batch_examples
        total_examples += batch_examples
        total_accuracy_draws += batch_examples * accuracy_samples

    accuracy_target = correct_target / max(total_accuracy_draws, 1)
    accuracy_valid = correct_valid / max(total_accuracy_draws, 1)

    # Coverage: per puzzle, draw n_samples and count unique valid completions hit.
    # Vectorized across `puzzles_per_forward` puzzles at once — one forward
    # at batch = puzzles_per_forward * n_samples instead of one per puzzle.
    # On a 4090 at L=64, batch=640 is healthy and ~30× faster than per-puzzle.
    cov_found = 0
    cov_possible = 0
    puzzles_per_forward = max(1, batch_size // n_samples) if n_samples > 0 else batch_size
    seq_len = dataset.item(0).input_tokens.shape[0]
    for start in range(0, len(dataset), puzzles_per_forward):
        end = min(start + puzzles_per_forward, len(dataset))
        chunk_items = [dataset.item(i) for i in range(start, end)]
        # (chunk_size, n_samples, L) → reshape to (chunk*n_samples, L)
        inp = torch.stack(
            [it.input_tokens.unsqueeze(0).expand(n_samples, -1) for it in chunk_items]
        ).to(device).reshape(-1, seq_len)
        outputs = model(inp, y=None)
        preds = outputs[-1].logits.argmax(dim=-1)  # (chunk*n_samples, L)
        preds = preds.reshape(len(chunk_items), n_samples, seq_len)
        for j, it in enumerate(chunk_items):
            hit = {
                _normalize_prediction(dataset, preds[j, k].tolist())
                for k in range(n_samples)
            } & valid_sets[start + j]
            cov_found += len(hit)
            cov_possible += it.n_solutions
    coverage = cov_found / max(cov_possible, 1)

    return {
        "n_examples": total_examples,
        "accuracy_valid_solution": accuracy_valid,
        "accuracy_target_exact_match": accuracy_target,
        "accuracy_samples": accuracy_samples,
        "accuracy_aggregation": "mean_independent_draws",
        "n_samples_for_coverage": n_samples,
        "coverage": coverage,
        "coverage_found": cov_found,
        "coverage_possible": cov_possible,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True, help="TSV file to evaluate on")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument(
        "--accuracy-samples",
        type=int,
        default=1,
        help="Independent draws averaged for accuracy. Default 1 preserves legacy behavior.",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument(
        "--use-act",
        action="store_true",
        help="Use ACT halting at inference (paper §A.1). Not yet wired in eval.py — "
             "raises NotImplementedError if set. Flag exists so the eval JSON records "
             "whether ACT was attempted.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for inference RNG. ε sampling at inference is stochastic; --seed "
             "makes accuracy/coverage numbers reproducible across runs.",
    )
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if args.use_act:
        raise NotImplementedError(
            "ACT halting at inference is not yet wired in eval.py. "
            "Tracked as a known TODO; see DEVIATIONS.md."
        )

    # Fix every RNG that affects inference ε sampling.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = _build_model_from_cfg(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    if args.use_ema:
        if "ema" not in ckpt:
            raise RuntimeError("--use-ema requested but checkpoint has no EMA state")
        ema = EMA(model, decay=float(cfg["train"]["ema_decay"]))
        ema.load_state_dict(ckpt["ema"])
        ema.apply_to(model)

    seq_len = int(cfg["task"]["seq_len"])
    task_name = str(cfg["task"].get("name", "nqueens"))
    ds = load_dataset(task_name, args.split, expected_seq_len=seq_len)
    print(f"[eval] {args.split.name}: {len(ds)} puzzles, n_samples={args.n_samples}, seed={args.seed}", file=sys.stderr)

    result = evaluate(
        model,
        ds,
        n_samples=args.n_samples,
        device=device,
        batch_size=args.batch_size,
        accuracy_samples=args.accuracy_samples,
    )
    result["split"] = str(args.split)
    result["checkpoint"] = str(args.checkpoint)
    result["seed"] = args.seed
    result["used_ema"] = args.use_ema
    result["used_act"] = args.use_act
    print(json.dumps(result, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
