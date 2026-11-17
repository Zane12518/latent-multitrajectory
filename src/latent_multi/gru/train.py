"""Bounded continuation of the archived SAT20/Maze10 GRU main-stage protocol.

Not the official GRAM host. Legacy pickle loading requires explicit trust and
cannot validate data identities absent from the original checkpoint.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .io import validate_rows, write_json
from .models import build_maze, build_sat, encode
from .objective import K, terms, combine


def content_identity(rows):
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": (n[0], n[1].tolist(), *n[2:]),
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["cpu"].cpu())
    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("CUDA device count differs from saved RNG state")
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def maze_quality(logits, rows):
    # Original foundation_scale.loss: probability mass of optimal actions,
    # only at active cells. This is added to complete shortest-path DAG mass.
    lp = logits.float().log_softmax(-1).reshape(len(rows), K, 100, 4)
    values = []
    for row, a in zip(rows, lp, strict=True):
        target = torch.tensor(row["optimal_actions"], device=logits.device, dtype=torch.bool)
        active = target.any(-1)
        if not active.any():
            raise ValueError("No supervised optimal-action cell")
        values.append(-a[:, active].masked_fill(~target[None, active], -torch.inf).logsumexp(-1).mean())
    return torch.stack(values).mean()


def protocol(task, arm, lr):
    if task not in ("maze", "sat") or arm not in ("mass", "multi", "set"):
        raise ValueError("Only main-stage mass/multi/set are supported")
    return dict(task=task, arm=arm, lr=lr, K=32, depths=[8, 16], microbatch=2,
                accumulation=4, aux_coef=0.1, cap=32, phase="main", clip=1.0,
                precision="fp32", weight_decay=0.01, scheduler=None, scaler=None,
                carry="none; state is initialized per forward")


class Session:
    def __init__(self, rows, task, arm="multi", lr=1e-4, device="cuda"):
        self.rows = validate_rows(rows, task)
        self.cfg = protocol(task, arm, lr)
        self.data_identity = content_identity(rows)
        self.device = device
        self.model = build_sat(device=device) if task == "sat" else build_maze(device=device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=.01)
        self.step = 0
        self.order_rng = random.Random(0)
        self.order, self.cursor = [], 0
        self.visits = Counter()
        self.lineage = []

    def restore(self, cp, *, legacy=False, fork_arm=False):
        required = {"model", "optimizer", "full_rng", "model_training", "scheduler", "scaler",
                    "step", "order_rng", "order", "cursor", "reference_visits", "spec"}
        missing = required - cp.keys()
        if missing:
            raise ValueError(f"Missing full recovery state: {sorted(missing)}")
        if cp["scheduler"] is not None or cp["scaler"] is not None:
            raise ValueError("This protocol has no scheduler or AMP scaler")
        old_arm = cp["spec"]["arm"]
        if old_arm != self.cfg["arm"] and not fork_arm:
            raise ValueError("Arm change requires --fork-arm; it is not exact continuation")
        if cp.get("schema") == "gru-continuation-v1":
            expected = dict(self.cfg, arm=old_arm) if fork_arm else self.cfg
            if cp["protocol"] != expected or cp["data_identity"] != self.data_identity:
                raise ValueError("Scientific configuration or input/reference content changed")
            if cp["row_count"] != len(self.rows):
                raise ValueError("Population length changed")
            self.lineage = list(cp["lineage"])
        elif legacy:
            spec = cp["spec"]
            if (spec.get("phase") != "main" or spec.get("regime") is not None
                    or spec.get("lr") != self.cfg["lr"]
                    or not spec["base"].startswith(self.cfg["task"])):
                raise ValueError("Legacy checkpoint does not match the bound main protocol")
            self.lineage = [dict(kind="legacy_state_adoption", base=spec["base"],
                flow=spec["flow"], step=cp["step"],
                warning="Original file has no data identity or complete scientific config; externally bound inputs required")]
        else:
            raise ValueError("Legacy state requires explicit --trusted-legacy adoption")
        if not 0 <= cp["cursor"] <= len(cp["order"]) or any(
                type(i) is not int or not 0 <= i < len(self.rows) for i in cp["order"]):
            raise ValueError("Invalid data order/cursor")
        ids = {r["id"] for r in self.rows}
        if any(k not in ids for k in cp["reference_visits"]):
            raise ValueError("Reference cursor contains unknown question IDs")
        self.model.load_state_dict(cp["model"], strict=True)
        self.optimizer.load_state_dict(cp["optimizer"])
        for group in self.optimizer.param_groups:
            if group["lr"] != self.cfg["lr"] or group["weight_decay"] != .01:
                raise ValueError("Optimizer configuration differs")
        if not self.optimizer.state or any(int(s["step"]) != cp["step"] for s in self.optimizer.state.values()):
            raise ValueError("Incomplete optimizer step state")
        self.model.train(cp["model_training"])
        self.step = cp["step"]
        self.order_rng.setstate(cp["order_rng"])
        self.order, self.cursor = list(cp["order"]), cp["cursor"]
        self.visits = Counter(cp["reference_visits"])
        if fork_arm and old_arm != self.cfg["arm"]:
            self.lineage.append(dict(kind="explicit_arm_fork", previous=old_arm,
                                     current=self.cfg["arm"], step=self.step))
        set_rng(cp["full_rng"])

    def state(self):
        return dict(schema="gru-continuation-v1", protocol=self.cfg,
            data_identity=self.data_identity, row_count=len(self.rows), lineage=self.lineage,
            model={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()},
            optimizer=self.optimizer.state_dict(), full_rng=rng_state(),
            model_training=self.model.training, scheduler=None, scaler=None, step=self.step,
            order_rng=self.order_rng.getstate(), order=self.order[:], cursor=self.cursor,
            reference_visits=dict(self.visits), spec=dict(arm=self.cfg["arm"]))

    def update(self):
        step = self.step + 1
        depth = self.cfg["depths"][(step-1) % 2]
        self.optimizer.zero_grad(set_to_none=True)
        ids, diagnostics, losses = [], [], []
        for _ in range(4):
            batch, visits = [], []
            for _ in range(2):
                if self.cursor >= len(self.order):
                    self.order = list(range(len(self.rows)))
                    self.order_rng.shuffle(self.order)
                    self.cursor = 0
                row = self.rows[self.order[self.cursor]]
                self.cursor += 1
                batch.append(row)
                ids.append(row["id"])
                visits.append(self.visits[row["id"]])
                self.visits[row["id"]] += 1
            x = [encode(r, 20, 91) if self.cfg["task"] == "sat" else r["x"] for r in batch]
            x = torch.tensor(x, device=self.device).repeat_interleave(K, 0)
            z = self.model(x, depth, True)
            parts, info = terms(z, batch, self.cfg["task"], visits, cap=32)
            native = maze_quality(z, batch) if self.cfg["task"] == "maze" else None
            loss = combine(parts, self.cfg["arm"], self.cfg["task"], native)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            (loss / 4).backward()
            losses.append(float(loss.detach()))
            diagnostics.extend(info)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise FloatingPointError("Missing or nonfinite parameter gradient")
        norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1, error_if_nonfinite=True))
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise FloatingPointError("Non-finite update")
        self.step = step
        return dict(step=step, depth=depth, ids=ids, cursor=self.cursor,
                    loss=sum(losses)/4, grad_preclip=norm, references=diagnostics,
                    trajectories=256, backward_calls=4)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["sat", "maze"], required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, required=True)
    p.add_argument("--arm", choices=["mass", "multi", "set"], default="multi")
    p.add_argument("--device", default="cuda")
    p.add_argument("--trusted-legacy", action="store_true")
    p.add_argument("--fork-arm", action="store_true")
    a = p.parse_args()
    if a.updates < 1 or a.output.exists():
        raise ValueError("Require positive updates and a new output directory")
    rows = json.loads(a.rows.read_text())
    session = Session(rows, a.task, a.arm, device=a.device)
    cp = torch.load(a.checkpoint, map_location="cpu", weights_only=not a.trusted_legacy)
    session.restore(cp, legacy=a.trusted_legacy, fork_arm=a.fork_arm)
    a.output.mkdir(parents=True)
    start = time.monotonic()
    for _ in range(a.updates):
        result = session.update()
        with (a.output / "updates.jsonl").open("a") as f:
            f.write(json.dumps(result, allow_nan=False) + "\n")
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    torch.save(session.state(), a.output / "full_state.pt")
    write_json(a.output / "RESULT.json", dict(status="updates_saved", updates=a.updates,
        last_step=session.step, seconds=time.monotonic()-start, torch=torch.__version__,
        cuda=torch.version.cuda, device=a.device, precision="fp32", protocol=session.cfg,
        legacy_adoption=a.trusted_legacy, full_historical_training_reproduction=False))


if __name__ == "__main__":
    main()
