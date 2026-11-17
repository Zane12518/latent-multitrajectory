"""Small strict interfaces for the two archived GRU tasks."""
import json
import zlib
from pathlib import Path

from .objective import shortest_dag


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never replace an existing result.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def content_crc32(path):
    value = 0
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value = zlib.crc32(block, value)
    return f"{value:08x}"


def check_weight_binding(weights, manifest, task):
    entries = json.loads(Path(manifest).read_text())["models"]
    matches = [r for r in entries if Path(r["file"]).name == Path(weights).name]
    if len(matches) != 1:
        raise ValueError("Weight filename is missing or ambiguous in the approved manifest")
    record = matches[0]
    if record["task"] != task or Path(weights).stat().st_size != record["bytes"]:
        raise ValueError("Weight task or byte count differs from its binding")
    if content_crc32(weights) != record["crc32"]:
        raise ValueError("Weight content differs from its CRC32 binding")
    return record["model_id"]


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg.get("task") not in ("sat", "maze"):
        raise ValueError("Only the project SAT20 and Maze10 GRU hosts are supported")
    if cfg.get("K") != 32 or cfg.get("depth") != 16:
        raise ValueError("This published-result adapter requires K32 and depth16")
    expected = "sat20_gru" if cfg["task"] == "sat" else "maze10_shared_gru"
    if cfg.get("host") != expected:
        raise ValueError("Task and GRU host disagree")
    return cfg


def validate_rows(rows, task, evaluation=False):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty row list")
    seen = set()
    for row in rows:
        ident = row.get("id")
        if type(ident) is not int or ident in seen:
            raise ValueError("Row IDs must be unique integers")
        seen.add(ident)
        if task == "sat":
            clauses = row.get("clauses", [])
            if row.get("n", 20) != 20 or not 1 <= len(clauses) <= 91:
                raise ValueError("Expected SAT20 with at most 91 clauses")
            for clause in clauses:
                if (len(clause) != 3 or any(type(v) is not int or not 1 <= abs(v) <= 20 for v in clause)
                        or len({abs(v) for v in clause}) != 3):
                    raise ValueError("Each clause needs three distinct, non-tautological variables")
        elif task == "maze":
            cells = row.get("x", [])
            if (len(cells) != 100 or any(type(v) is not int or v not in (1, 2, 3, 4) for v in cells)
                    or cells.count(3) != 1 or cells.count(4) != 1):
                raise ValueError("Expected a 10x10 grid with one start and one goal")
            if evaluation:
                levels, count = shortest_dag(tuple(cells))
                if not count or row.get("optimal") != len(levels):
                    raise ValueError("Evaluation shortest-path length disagrees with the input grid")
        else:
            raise ValueError(task)
    return rows
