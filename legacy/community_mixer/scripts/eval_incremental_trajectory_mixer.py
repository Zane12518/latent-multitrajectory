#!/usr/bin/env python3
"""Evaluate causal incremental particle acquisition with cached mixer context."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from gram.datasets import load_dataset
from gram.incremental_particles import append_particles
from gram.maze_decode import decode_maze_logits
from gram.nqueens_decode import decode_nqueens_logits
from gram.trajectory_metrics import (
    aggregate_prefix_metrics,
    compute_prefix_metrics,
    parse_sample_sizes,
)
from scripts.eval_trajectory_mixer import (
    _apply_joint_base_state,
    _load_base_model,
    _load_mixer,
    _normalize_prediction,
    _percentile,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    temporary.replace(path)


def _decode(
    *,
    logits: torch.Tensor,
    inputs: torch.Tensor,
    task_name: str,
    cfg: dict,
    maze_decode: str,
    maze_max_paths: int,
    nqueens_decode: str,
    nqueens_beam_size: int,
) -> torch.Tensor:
    if task_name == "maze":
        grid_size = int(
            cfg["task"].get(
                "grid_size",
                int(cfg["task"]["seq_len"] ** 0.5),
            )
        )
        return decode_maze_logits(
            logits,
            inputs,
            grid_size=grid_size,
            mode=maze_decode,
            max_paths=maze_max_paths,
        )
    if task_name == "nqueens":
        board_size = int(
            cfg["task"].get(
                "board_size",
                int(cfg["task"]["seq_len"] ** 0.5),
            )
        )
        return decode_nqueens_logits(
            logits,
            inputs,
            board_size=board_size,
            mode=nqueens_decode,
            beam_size=nqueens_beam_size,
        )
    if maze_decode != "argmax" or nqueens_decode != "argmax":
        raise ValueError("structured decoding is only supported for Maze/N-Queens")
    return logits.argmax(dim=-1)


@torch.no_grad()
def evaluate_incremental(
    *,
    base_model,
    mixer,
    cfg: dict,
    dataset,
    budgets: list[int],
    device: torch.device,
    batch_size: int,
    amp_dtype: torch.dtype | None,
    prior_noise_scale: float,
    mixer_every: int,
    limit: int | None,
    task_name: str,
    maze_decode: str,
    maze_max_paths: int,
    nqueens_decode: str,
    nqueens_beam_size: int,
) -> tuple[list[dict], list[dict], dict]:
    if budgets != sorted(set(budgets)):
        raise ValueError("budgets must be unique and increasing")
    if budgets[0] < 1:
        raise ValueError("budgets must be positive")
    n_items = min(len(dataset), limit) if limit is not None else len(dataset)
    seq_len = dataset.item(0).input_tokens.shape[0]
    max_particles = budgets[-1]
    puzzles_per_forward = max(1, batch_size // max_particles)
    per_puzzle_rows: list[dict] = []
    stage_wall_s: dict[int, float] = defaultdict(float)
    model_decode_stage_wall_s: dict[int, float] = defaultdict(float)
    cumulative_latency_ms: dict[int, list[float]] = defaultdict(list)
    model_decode_cumulative_latency_ms: dict[int, list[float]] = defaultdict(list)
    total_particle_forwards = 0
    evaluation_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for start in range(0, n_items, puzzles_per_forward):
        end = min(start + puzzles_per_forward, n_items)
        items = [dataset.item(index) for index in range(start, end)]
        puzzle_inputs = torch.stack(
            [item.input_tokens for item in items]
        ).to(device)
        state = None
        previous_budget = 0
        batch_cumulative_ms = 0.0
        batch_cumulative_model_decode_ms = 0.0

        for budget in budgets:
            additional = budget - previous_budget
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            stage_started = time.perf_counter()
            state = append_particles(
                base_model=base_model,
                puzzle_inputs=puzzle_inputs,
                additional_particles=additional,
                state=state,
                mixer=mixer,
                mixer_every=mixer_every,
                prior_noise_scale=prior_noise_scale,
                amp_dtype=amp_dtype,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - stage_started
            stage_wall_s[budget] += elapsed
            per_puzzle_stage_ms = 1000.0 * elapsed / len(items)
            batch_cumulative_ms += per_puzzle_stage_ms
            cumulative_latency_ms[budget].extend(
                [batch_cumulative_ms] * len(items)
            )
            total_particle_forwards += len(items) * additional

            flat_logits = state.logits.reshape(
                -1,
                seq_len,
                base_model.vocab_size,
            )
            expanded_inputs = puzzle_inputs.unsqueeze(1).expand(
                len(items),
                budget,
                seq_len,
            ).reshape(-1, seq_len)
            predictions = _decode(
                logits=flat_logits,
                inputs=expanded_inputs,
                task_name=task_name,
                cfg=cfg,
                maze_decode=maze_decode,
                maze_max_paths=maze_max_paths,
                nqueens_decode=nqueens_decode,
                nqueens_beam_size=nqueens_beam_size,
            )
            decoded_log_probs = (
                flat_logits.float()
                .log_softmax(dim=-1)
                .gather(-1, predictions.unsqueeze(-1))
                .squeeze(-1)
            )
            confidence = decoded_log_probs.mean(dim=-1).exp().reshape(
                len(items),
                budget,
            ).cpu()
            predictions = predictions.reshape(
                len(items),
                budget,
                seq_len,
            ).cpu()
            lprm = state.lprm_value.cpu()
            model_decode_elapsed = time.perf_counter() - stage_started
            model_decode_stage_wall_s[budget] += model_decode_elapsed
            per_puzzle_model_decode_ms = (
                1000.0 * model_decode_elapsed / len(items)
            )
            batch_cumulative_model_decode_ms += per_puzzle_model_decode_ms
            model_decode_cumulative_latency_ms[budget].extend(
                [batch_cumulative_model_decode_ms] * len(items)
            )

            for local_index, item in enumerate(items):
                valid_set = {
                    tuple(row.tolist()) for row in item.all_completions
                }
                normalized = [
                    _normalize_prediction(
                        dataset,
                        predictions[local_index, particle].tolist(),
                    )
                    for particle in range(budget)
                ]
                row = compute_prefix_metrics(
                    puzzle_id=item.puzzle_id,
                    n_solutions=item.n_solutions,
                    predictions=normalized,
                    valid_set=valid_set,
                    sample_sizes=[budget],
                )[0]
                n_unique_all = len(set(normalized))
                row.update(
                    {
                        "decoder_confidence_mean": float(
                            confidence[local_index].mean().item()
                        ),
                        "lprm_value_mean": float(
                            lprm[local_index].mean().item()
                        ),
                        "n_unique_all": n_unique_all,
                        "duplicate_saturation": 1.0
                        - n_unique_all / budget,
                        "novelty_rate": n_unique_all / budget,
                        "constraint_residual": 0.0,
                        "incremental_stage_particles": additional,
                        "incremental_cumulative_particles": budget,
                        "stage_latency_ms": per_puzzle_stage_ms,
                        "batch_amortized_latency_ms": batch_cumulative_ms,
                        "model_decode_stage_latency_ms": (
                            per_puzzle_model_decode_ms
                        ),
                        "model_decode_cumulative_latency_ms": batch_cumulative_model_decode_ms,
                        "policy": "incremental_query_mixer",
                        "mixer_mode": "incremental_query_only",
                        "mixer_every": mixer_every,
                        "n_particles": budget,
                    }
                )
                per_puzzle_rows.append(row)
            previous_budget = budget

        print(
            f"[incremental-mixer] {end}/{n_items} puzzles evaluated",
            file=sys.stderr,
        )

    total_wall_s = time.perf_counter() - evaluation_started
    aggregate_rows = aggregate_prefix_metrics(per_puzzle_rows)
    cumulative_stage_wall = 0.0
    cumulative_model_decode_wall = 0.0
    for budget in budgets:
        cumulative_stage_wall += stage_wall_s[budget]
        cumulative_model_decode_wall += model_decode_stage_wall_s[budget]
        matches = [
            row
            for row in aggregate_rows
            if int(row["n_samples"]) == budget
        ]
        if len(matches) != 1:
            raise RuntimeError(f"missing N={budget} aggregate")
        row = matches[0]
        row.update(
            {
                "policy": "incremental_query_mixer",
                "mixer_mode": "incremental_query_only",
                "mixer_every": mixer_every,
                "n_particles": budget,
                "wall_s": cumulative_stage_wall,
                "model_decode_wall_s": cumulative_model_decode_wall,
                "unique_valid_per_second": float(
                    row["total_unique_valid_hits"]
                )
                / cumulative_stage_wall,
                "latency_ms_p50": _percentile(
                    cumulative_latency_ms[budget],
                    0.50,
                ),
                "latency_ms_p95": _percentile(
                    cumulative_latency_ms[budget],
                    0.95,
                ),
                "unique_valid_per_model_decode_second": float(
                    row["total_unique_valid_hits"]
                )
                / cumulative_model_decode_wall,
                "model_decode_latency_ms_p50": _percentile(
                    model_decode_cumulative_latency_ms[budget],
                    0.50,
                ),
                "model_decode_latency_ms_p95": _percentile(
                    model_decode_cumulative_latency_ms[budget],
                    0.95,
                ),
            }
        )

    legacy_staged_particle_forwards = n_items * sum(budgets)
    diagnostics = {
        "n_examples": n_items,
        "budgets": budgets,
        "actual_particle_forwards": total_particle_forwards,
        "legacy_full_recompute_particle_forwards": (
            legacy_staged_particle_forwards
        ),
        "particle_forward_reduction_pct": 100.0
        * (
            1.0
            - total_particle_forwards
            / legacy_staged_particle_forwards
        ),
        "terminal_particle_forwards_per_puzzle": (
            total_particle_forwards / n_items
        ),
        "wall_s_including_decode": total_wall_s,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "semantics": (
            "old particles are frozen; newly appended particles query cached "
            "old+new mixer context"
        ),
    }
    return aggregate_rows, per_puzzle_rows, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--mixer-checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--budgets", default="4,8,16,32,64")
    parser.add_argument("--prior-noise-scale", type=float, default=1.0)
    parser.add_argument("--mixer-every", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp-dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument(
        "--maze-decode",
        choices=["argmax", "topk_len", "shortest_path"],
        default="argmax",
    )
    parser.add_argument("--maze-max-paths", type=int, default=512)
    parser.add_argument(
        "--nqueens-decode",
        choices=["argmax", "topk", "row_top1", "beam"],
        default="argmax",
    )
    parser.add_argument("--nqueens-beam-size", type=int, default=64)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    budgets = parse_sample_sizes(args.budgets)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    amp_dtype = {"fp32": None, "bf16": torch.bfloat16}[args.amp_dtype]
    base_model, cfg = _load_base_model(
        args.base_checkpoint,
        device,
        args.use_ema,
    )
    mixer_payload = torch.load(
        args.mixer_checkpoint,
        map_location=device,
        weights_only=False,
    )
    mixer = _load_mixer(args.mixer_checkpoint, cfg, device)
    joint_base_loaded = _apply_joint_base_state(base_model, mixer_payload)
    task_name = str(cfg["task"].get("name", "nqueens"))
    dataset = load_dataset(
        task_name,
        args.split,
        expected_seq_len=int(cfg["task"]["seq_len"]),
    )
    aggregates, per_puzzle, diagnostics = evaluate_incremental(
        base_model=base_model,
        mixer=mixer,
        cfg=cfg,
        dataset=dataset,
        budgets=budgets,
        device=device,
        batch_size=args.batch_size,
        amp_dtype=amp_dtype,
        prior_noise_scale=args.prior_noise_scale,
        mixer_every=args.mixer_every,
        limit=args.limit,
        task_name=task_name,
        maze_decode=args.maze_decode,
        maze_max_paths=args.maze_max_paths,
        nqueens_decode=args.nqueens_decode,
        nqueens_beam_size=args.nqueens_beam_size,
    )
    result = {
        "schema": 1,
        "split": str(args.split),
        "base_checkpoint": str(args.base_checkpoint),
        "mixer_checkpoint": str(args.mixer_checkpoint),
        "seed": args.seed,
        "amp_dtype": args.amp_dtype,
        "prior_noise_scale": args.prior_noise_scale,
        "task_name": task_name,
        "maze_decode": args.maze_decode if task_name == "maze" else None,
        "maze_max_paths": (
            args.maze_max_paths if task_name == "maze" else None
        ),
        "policy": "incremental_query_mixer",
        "mixer_every": args.mixer_every,
        "budgets": budgets,
        "n_examples": diagnostics["n_examples"],
        "diagnostics": {
            **diagnostics,
            "gate": float(mixer.gate.detach().item()),
            "base_train_mode": mixer_payload.get(
                "base_train_mode",
                "frozen",
            ),
            "joint_base_loaded": joint_base_loaded,
        },
        "aggregates": aggregates,
    }
    _write_json(args.output_json, result)
    _write_jsonl(args.output_jsonl, per_puzzle)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
