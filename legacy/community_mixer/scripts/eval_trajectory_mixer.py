#!/usr/bin/env python3
"""Evaluate a learned trajectory-axis mixer with duplicate/coverage metrics."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from gram.datasets import load_dataset
from gram.ema import EMA
from gram.eval import _build_model_from_cfg, select_device
from gram.maze_decode import decode_maze_logits
from gram.nqueens_decode import decode_nqueens_logits
from gram.trajectory_metrics import (
    aggregate_prefix_metrics,
    compute_prefix_metrics,
    parse_sample_sizes,
)
from gram.trajectory_mixer import (
    GLOBAL_CONTEXT_MODES,
    ResidualTrajectoryMixer,
    load_trajectory_mixer_state_dict,
)


def _policy_name(mixer_mode: str) -> str:
    if mixer_mode == "disabled":
        return "independent"
    if mixer_mode == "learned":
        return "learned_mixer"
    return f"mixer_{mixer_mode}"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires non-empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _load_base_model(
    checkpoint: Path,
    device: torch.device,
    use_ema: bool,
) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = _build_model_from_cfg(cfg).to(device)
    state = ckpt.get("model", ckpt.get("base_model"))
    if not isinstance(state, dict):
        raise RuntimeError("Base checkpoint contains no model state")
    model.load_state_dict(state)
    if use_ema:
        if "ema" not in ckpt:
            raise RuntimeError("--use-ema requested but checkpoint has no EMA state")
        ema = EMA(model, decay=float(cfg["train"]["ema_decay"]))
        ema.load_state_dict(ckpt["ema"])
        ema.apply_to(model)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, cfg


def _load_mixer(
    checkpoint: Path,
    cfg: dict,
    device: torch.device,
) -> ResidualTrajectoryMixer:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    args = payload.get("args", {})
    architecture_from_args = args.get("mixer_architecture")
    architecture_from_checkpoint = payload.get("mixer_architecture")
    if (
        architecture_from_args is not None
        and architecture_from_checkpoint is not None
        and architecture_from_args != architecture_from_checkpoint
    ):
        raise ValueError(
            "mixer architecture mismatch between checkpoint fields: "
            f"{architecture_from_args!r} != {architecture_from_checkpoint!r}"
        )
    architecture = (
        architecture_from_checkpoint
        or architecture_from_args
        or "local_only"
    )
    model_cfg = cfg["model"]
    mixer = ResidualTrajectoryMixer(
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        dropout=float(args.get("dropout", 0.0)),
        gate_init=float(args.get("gate_init", -4.0)),
        architecture=architecture,
    ).to(device)
    mixer.legacy_checkpoint_loaded = load_trajectory_mixer_state_dict(
        mixer,
        payload["mixer"],
    )
    mixer.eval()
    return mixer


def _apply_joint_base_state(base_model: torch.nn.Module, payload: dict) -> bool:
    state = payload.get("base_model")
    if state is None:
        return False
    base_model.load_state_dict(state)
    base_model.eval()
    return True


def _normalize_prediction(dataset, pred: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    normalizer = getattr(dataset, "normalize_prediction", None)
    if normalizer is None:
        return tuple(pred)
    return normalizer(pred)


@torch.no_grad()
def run_mixer_batch(
    *,
    base_model,
    mixer: ResidualTrajectoryMixer,
    inputs: torch.Tensor,
    n_particles: int,
    prior_noise_scale: float,
    amp_dtype: torch.dtype | None,
    mixer_every: int,
    mixer_mode: str = "learned",
    mixer_global_context_mode: str = "same_puzzle",
) -> torch.Tensor:
    ex = base_model.encode_input(inputs)
    h, l = base_model.core.initial_state(inputs.shape[0])
    h = h.to(inputs.device, non_blocking=True)
    l = l.to(inputs.device, non_blocking=True)

    use_autocast = amp_dtype is not None and inputs.device.type == "cuda"
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if use_autocast
        else contextlib.nullcontext()
    )

    with autocast_ctx:
        for step_idx in range(base_model.nsup_train):
            h, l, _info = base_model.core.supervision_step(
                h,
                l,
                ex,
                y_pooled=None,
                truncated_grad=True,
                prior_noise_scale=prior_noise_scale,
            )
            if mixer_every > 0 and (step_idx + 1) % mixer_every == 0:
                h = mixer(
                    h,
                    n_particles=n_particles,
                    mode=mixer_mode,
                    global_context_mode=mixer_global_context_mode,
                )
            h = h.detach()
            l = l.detach()
        logits = base_model.lm_head(h)
    return logits


@torch.no_grad()
def evaluate_mixer(
    *,
    base_model,
    mixer: ResidualTrajectoryMixer,
    dataset,
    sample_sizes: list[int],
    n_particles: int,
    device: torch.device,
    batch_size: int,
    amp_dtype: torch.dtype | None,
    prior_noise_scale: float,
    mixer_every: int,
    limit: int | None,
    task_name: str,
    nqueens_decode: str,
    nqueens_beam_size: int,
    board_size: int | None,
    maze_decode: str,
    maze_max_paths: int,
    grid_size: int | None,
    mixer_mode: str = "learned",
    mixer_global_context_mode: str = "same_puzzle",
) -> tuple[list[dict], list[dict]]:
    n_items = min(len(dataset), limit) if limit is not None else len(dataset)
    seq_len = dataset.item(0).input_tokens.shape[0]
    puzzles_per_forward = max(1, batch_size // n_particles)
    per_puzzle_rows: list[dict] = []
    per_puzzle_latency_ms: list[float] = []
    evaluation_started = time.perf_counter()

    for start in range(0, n_items, puzzles_per_forward):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_started = time.perf_counter()
        end = min(start + puzzles_per_forward, n_items)
        chunk_items = [dataset.item(i) for i in range(start, end)]
        inputs = torch.stack(
            [item.input_tokens.unsqueeze(0).expand(n_particles, -1) for item in chunk_items]
        )
        inputs = inputs.reshape(-1, seq_len).to(device)

        logits = run_mixer_batch(
            base_model=base_model,
            mixer=mixer,
            inputs=inputs,
            n_particles=n_particles,
            prior_noise_scale=prior_noise_scale,
            amp_dtype=amp_dtype,
            mixer_every=mixer_every,
            mixer_mode=mixer_mode,
            mixer_global_context_mode=mixer_global_context_mode,
        )
        if task_name == "nqueens":
            n = board_size if board_size is not None else int(seq_len**0.5)
            preds = decode_nqueens_logits(
                logits,
                inputs,
                board_size=n,
                mode=nqueens_decode,
                beam_size=nqueens_beam_size,
            )
        elif task_name == "maze":
            n = grid_size if grid_size is not None else int(seq_len**0.5)
            preds = decode_maze_logits(
                logits,
                inputs,
                grid_size=n,
                mode=maze_decode,
                max_paths=maze_max_paths,
            )
        else:
            if nqueens_decode != "argmax":
                raise ValueError("--nqueens-decode is only valid for N-Queens checkpoints")
            if maze_decode != "argmax":
                raise ValueError("--maze-decode is only valid for Maze checkpoints")
            preds = logits.argmax(dim=-1)
        decoded_log_probs = (
            logits.float()
            .log_softmax(dim=-1)
            .gather(-1, preds.unsqueeze(-1))
            .squeeze(-1)
        )
        candidate_confidence = decoded_log_probs.mean(dim=-1).exp()
        candidate_confidence = candidate_confidence.reshape(
            len(chunk_items), n_particles
        ).cpu()
        preds = preds.reshape(len(chunk_items), n_particles, seq_len).cpu()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_latency_ms = 1000.0 * (time.perf_counter() - batch_started) / len(chunk_items)
        per_puzzle_latency_ms.extend([batch_latency_ms] * len(chunk_items))

        for local_idx, item in enumerate(chunk_items):
            valid_set = {tuple(row.tolist()) for row in item.all_completions}
            predictions = [
                _normalize_prediction(dataset, preds[local_idx, k].tolist())
                for k in range(n_particles)
            ]
            prefix_rows = compute_prefix_metrics(
                puzzle_id=item.puzzle_id,
                n_solutions=item.n_solutions,
                predictions=predictions,
                valid_set=valid_set,
                sample_sizes=sample_sizes,
            )
            for row in prefix_rows:
                prefix_n = int(row["n_samples"])
                prefix_predictions = predictions[:prefix_n]
                n_unique_all = len(set(prefix_predictions))
                row["decoder_confidence_mean"] = float(
                    candidate_confidence[local_idx, :prefix_n].mean().item()
                )
                # Router features must not depend on all_completions or validity labels.
                row["n_unique_all"] = n_unique_all
                row["duplicate_saturation"] = 1.0 - n_unique_all / prefix_n
                row["novelty_rate"] = n_unique_all / prefix_n
                # shortest_path decoding enforces Maze constraints by construction.
                # Keep this explicit, but non-discriminative, for schema stability.
                row["constraint_residual"] = 0.0
                row["batch_amortized_latency_ms"] = batch_latency_ms
                row["policy"] = _policy_name(mixer_mode)
                row["mixer_mode"] = mixer_mode
                row["mixer_architecture"] = mixer.architecture
                row["mixer_global_context_mode"] = mixer_global_context_mode
                row["mixer_every"] = mixer_every
                row["n_particles"] = n_particles
                per_puzzle_rows.append(row)

        print(f"[trajectory-mixer] {end}/{n_items} puzzles evaluated", file=sys.stderr)

    wall_s = time.perf_counter() - evaluation_started
    aggregate_rows = aggregate_prefix_metrics(per_puzzle_rows)
    for row in aggregate_rows:
        row["policy"] = _policy_name(mixer_mode)
        row["mixer_mode"] = mixer_mode
        row["mixer_architecture"] = mixer.architecture
        row["mixer_global_context_mode"] = mixer_global_context_mode
        row["mixer_every"] = mixer_every
        row["n_particles"] = n_particles
        row["wall_s"] = wall_s
        row["unique_valid_per_second"] = float(row["total_unique_valid_hits"]) / wall_s
        row["latency_ms_p50"] = _percentile(per_puzzle_latency_ms, 0.50)
        row["latency_ms_p95"] = _percentile(per_puzzle_latency_ms, 0.95)
    return aggregate_rows, per_puzzle_rows


def _evaluation_budget_plan(
    sample_sizes: list[int],
    *,
    n_particles: int,
    budget_conditioned: bool,
) -> list[tuple[int, tuple[int, ...]]]:
    """Plan faithful per-budget forwards or the legacy max-N prefix pass."""
    if not sample_sizes:
        raise ValueError("sample_sizes must not be empty")
    if max(sample_sizes) > n_particles:
        raise ValueError("--sample-sizes cannot exceed --n-particles")
    if budget_conditioned:
        return [(budget, (budget,)) for budget in sample_sizes]
    return [(n_particles, tuple(sample_sizes))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-checkpoint", type=Path, required=True)
    ap.add_argument("--mixer-checkpoint", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--n-particles", type=int, default=64)
    ap.add_argument("--sample-sizes", default="2,4,8,16,32,64")
    ap.add_argument(
        "--budget-conditioned",
        action="store_true",
        help=(
            "Run a separate forward at each sample size so mixer communication "
            "cannot use trajectories outside the evaluated budget."
        ),
    )
    ap.add_argument("--prior-noise-scale", type=float, default=1.0)
    ap.add_argument("--mixer-every", type=int, default=1)
    ap.add_argument(
        "--mixer-mode",
        choices=[
            "disabled",
            "learned",
            "self_only",
            "uniform_mean",
            "wrong_puzzle",
        ],
        default="learned",
        help="Inference-only communication ablation applied to the trained mixer.",
    )
    ap.add_argument(
        "--mixer-global-context-mode",
        choices=GLOBAL_CONTEXT_MODES,
        default="same_puzzle",
        help=(
            "Use same-puzzle global trajectory summaries, or roll summaries "
            "across puzzles as a preregistered negative control."
        ),
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp-dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument(
        "--nqueens-decode",
        choices=["argmax", "topk", "row_top1", "beam"],
        default="argmax",
        help="Structured N-Queens decoder for metrics. Non-N-Queens tasks require argmax.",
    )
    ap.add_argument("--nqueens-beam-size", type=int, default=64)
    ap.add_argument(
        "--maze-decode",
        choices=["argmax", "topk_len", "shortest_path"],
        default="argmax",
        help="Structured Maze decoder for metrics. Non-Maze tasks require argmax.",
    )
    ap.add_argument("--maze-max-paths", type=int, default=512)
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--output-jsonl", type=Path, default=None)
    args = ap.parse_args()

    sample_sizes = parse_sample_sizes(args.sample_sizes)
    budget_plan = _evaluation_budget_plan(
        sample_sizes,
        n_particles=args.n_particles,
        budget_conditioned=args.budget_conditioned,
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)
    amp_dtype = {"fp32": None, "bf16": torch.bfloat16}[args.amp_dtype]
    base_model, cfg = _load_base_model(args.base_checkpoint, device, args.use_ema)
    mixer_payload = torch.load(
        args.mixer_checkpoint, map_location=device, weights_only=True
    )
    mixer = _load_mixer(args.mixer_checkpoint, cfg, device)
    joint_base_loaded = _apply_joint_base_state(base_model, mixer_payload)

    task_name = str(cfg["task"].get("name", "nqueens"))
    dataset = load_dataset(task_name, args.split, expected_seq_len=int(cfg["task"]["seq_len"]))
    aggregates: list[dict] = []
    per_puzzle: list[dict] = []
    for run_particles, run_sample_sizes in budget_plan:
        if args.budget_conditioned:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        run_aggregates, run_per_puzzle = evaluate_mixer(
            base_model=base_model,
            mixer=mixer,
            dataset=dataset,
            sample_sizes=list(run_sample_sizes),
            n_particles=run_particles,
            device=device,
            batch_size=args.batch_size,
            amp_dtype=amp_dtype,
            prior_noise_scale=args.prior_noise_scale,
            mixer_every=args.mixer_every,
            limit=args.limit,
            task_name=task_name,
            nqueens_decode=args.nqueens_decode,
            nqueens_beam_size=args.nqueens_beam_size,
            board_size=int(cfg["task"].get("board_size", int(cfg["task"]["seq_len"] ** 0.5))),
            maze_decode=args.maze_decode,
            maze_max_paths=args.maze_max_paths,
            grid_size=int(cfg["task"].get("grid_size", int(cfg["task"]["seq_len"] ** 0.5))),
            mixer_mode=args.mixer_mode,
            mixer_global_context_mode=args.mixer_global_context_mode,
        )
        for row in [*run_aggregates, *run_per_puzzle]:
            row["budget_conditioned"] = args.budget_conditioned
        aggregates.extend(run_aggregates)
        per_puzzle.extend(run_per_puzzle)
    result = {
        "split": str(args.split),
        "base_checkpoint": str(args.base_checkpoint),
        "mixer_checkpoint": str(args.mixer_checkpoint),
        "seed": args.seed,
        "amp_dtype": args.amp_dtype,
        "prior_noise_scale": args.prior_noise_scale,
        "nqueens_decode": args.nqueens_decode if task_name == "nqueens" else None,
        "nqueens_beam_size": args.nqueens_beam_size if task_name == "nqueens" else None,
        "maze_decode": args.maze_decode if task_name == "maze" else None,
        "maze_max_paths": args.maze_max_paths if task_name == "maze" else None,
        "policy": _policy_name(args.mixer_mode),
        "mixer_mode": args.mixer_mode,
        "mixer_architecture": mixer.architecture,
        "mixer_global_context_mode": args.mixer_global_context_mode,
        "mixer_every": args.mixer_every,
        "n_particles": args.n_particles,
        "sample_sizes": sample_sizes,
        "budget_conditioned": args.budget_conditioned,
        "n_examples": len({row["puzzle_id"] for row in per_puzzle}),
        "diagnostics": {
            "gate": float(mixer.gate.detach().item()),
            "global_gate": float(mixer.global_gate.detach().item()),
            "legacy_mixer_checkpoint": bool(
                getattr(mixer, "legacy_checkpoint_loaded", False)
            ),
            "base_train_mode": mixer_payload.get("base_train_mode", "frozen"),
            "joint_base_loaded": joint_base_loaded,
        },
        "aggregates": aggregates,
    }
    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        _write_json(args.output_json, result)
    if args.output_jsonl is not None:
        _write_jsonl(args.output_jsonl, per_puzzle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
