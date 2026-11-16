"""用伪谱求解器复现 PDEBench 样本，找出数据真实的 (扩散系数, 时间范围)。x 固定为 [0,1) 周期。"""
import math
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from burgers_solver import load_pdebench_samples, relative_l2, solve  # noqa: E402

PATH = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "datasets") + "/pdebench_burgers/burgers_nu0.01_test.parquet"
samples = list(load_pdebench_samples(PATH, 3))
CASES = [
    ("代码现用: nu=0.01,     T=1", 0.01, 1.0),
    ("nu=0.01,     T=2", 0.01, 2.0),
    ("PDEBench: nu=0.01/pi, T=2", 0.01 / math.pi, 2.0),
    ("nu=0.01/pi, T=1", 0.01 / math.pi, 1.0),
]
for label, nu, T in CASES:
    t0 = time.time()
    errs = [relative_l2(solve(f[:, 0], nu=nu, nt=f.shape[1], duration=T, dt=5e-4), f) for f in samples]
    print(f"{label:30} 相对 L2 = {np.mean(errs):.2e}  (各样本 {', '.join(f'{e:.1e}' for e in errs)}; {time.time()-t0:.0f}s)")
