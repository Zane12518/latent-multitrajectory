"""One-question interface run, not historical benchmark/throughput reproduction."""
import argparse
from pathlib import Path
import runpy
import sys
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--case", choices=["maze12","nq8","query_only"], required=True)
p.add_argument("--base", type=Path, required=True)
p.add_argument("--mixer", type=Path, required=True)
p.add_argument("--data", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.output.exists() or a.output.with_suffix(".jsonl").exists():
    raise FileExistsError("Use a new output path")
torch.set_num_threads(1)
script = "eval_incremental_trajectory_mixer.py" if a.case=="query_only" else "eval_trajectory_mixer.py"
args = ["--base-checkpoint",str(a.base),"--mixer-checkpoint",str(a.mixer),"--split",str(a.data),
    "--device","cpu","--batch-size","1","--seed","0","--amp-dtype","fp32","--limit","1",
    "--output-json",str(a.output),"--output-jsonl",str(a.output.with_suffix(".jsonl"))]
args += ["--maze-decode","shortest_path","--maze-max-paths","64"] if a.case=="maze12" else ["--nqueens-decode","beam","--nqueens-beam-size","64"]
args += ["--budgets","1,2"] if a.case=="query_only" else ["--n-particles","2","--sample-sizes","1,2"]
sys.argv = [script]+args
runpy.run_path(str(Path(__file__).parent/"scripts"/script),run_name="__main__")
