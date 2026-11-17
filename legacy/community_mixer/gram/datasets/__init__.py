"""Historical Maze/N-Queens loaders only; no other benchmark is included."""
from .maze import MazeDataset
from .nqueens import NQueensDataset

def load_dataset(task_name, tsv_path, expected_seq_len=None):
    cls = {"maze": MazeDataset, "nqueens": NQueensDataset}.get(task_name)
    if cls is None:
        raise ValueError(f"This historical adapter does not include {task_name!r}")
    return cls(tsv_path, expected_seq_len=expected_seq_len)
