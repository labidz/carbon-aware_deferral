import numpy as np, json, ukciv as U, pandas as pd, warnings; warnings.filterwarnings("ignore")
from gate import *
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); W = np.load("/home/claude/data/wf_main.npz")
s, L, f0, y = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], M["y"]; blk = W["blk"]; pm, p50 = W["mean"], W["q50"]
lam = gate_lambdas(s, L, y, f0, np.nan_to_num(pm), blk)
sel = (blk >= 0) & np.isfinite(y)
res = y - f0; names = ["2023-06","2023-07","2023-08","2023-09","2023-10","2023-11","2023-12","2024-01/02","2026-07","2026-08","2026-09","2026-10"]
rows = []
b = lead_bucket(L)
for k in range(12):
    m = sel & (blk == k)
    r = {"block": names[k], "n": int(m.sum()), "NESO": np.mean(np.abs(res[m])), "ML_mean": np.mean(np.abs(res[m] - pm[m])), "ML_median": np.mean(np.abs(res[m] - p50[m])),
         "ML_gated": np.mean(np.abs(res[m] - lam[m] * pm[m])), "mean_lambda": float(np.mean(lam[m]))}
    rows.append(r)
df = pd.DataFrame(rows); print(df.round(2).to_string(index=False))
for nm, mk in [("ALL", sel), ("2023-24", sel & (blk <= 7)), ("2026", sel & (blk >= 8)), ("ex-first-block(Jun)", sel & (blk >= 1))]:
    print(nm, {k: round(float(np.mean(np.abs(res[mk] - v[mk]))), 2) for k, v in {"NESO": 0*res, "ML_mean": pm, "ML_median": p50, "ML_gated": lam*pm}.items()})
print("\nBy lead bucket (blocks>=1):")
for bb, nm in enumerate(["0-2h", "2-6h", "6-12h", "12-24h", "24-48h"]):
    mk = sel & (blk >= 1) & (b == bb)
    print(nm, {k: round(float(np.mean(np.abs(res[mk] - v[mk]))), 2) for k, v in {"NESO": 0*res, "ML_mean": pm, "ML_gated": lam*pm}.items()})
