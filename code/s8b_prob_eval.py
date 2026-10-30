import numpy as np, json, pandas as pd, ukciv as U, warnings, sys; warnings.filterwarnings("ignore")
from conformal import *
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); W = np.load("/home/claude/data/wf_main.npz")
s, L, f0, y = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], M["y"]; blk = W["blk"]
q10, q50, q90 = np.nan_to_num(W["q10"]), np.nan_to_num(W["q50"]), np.nan_to_num(W["q90"])
lo = np.minimum(q10, q90); hi = np.maximum(q10, q90)            # non-crossing
sel = (blk >= 0) & np.isfinite(y); ev = sel & (blk >= 1)         # evaluation blocks (block 0 = calibration for static CQR)
res = y - f0
LO, HI = f0 + lo, f0 + hi
# (a) naive empirical NESO interval: residual quantiles per lead bucket from block 0 only (static, uses same data budget as static CQR)
b = bucket(L); Qe = np.array([[np.quantile(res[sel & (blk == 0) & (b == k)], p) for p in (0.1, 0.9)] for k in range(NB)])
NLO, NHI = f0 + Qe[b, 0], f0 + Qe[b, 1]
# (b) static CQR on block 0
Qs = static_cqr(L, res, lo, hi, sel & (blk == 0))
# (c) rolling CQR
Qr = rolling_cqr(s, L, res, lo, hi, sel, min_n=1500)
names = ["2023-06","2023-07","2023-08","2023-09","2023-10","2023-11","2023-12","2024-01/02","2026-07","2026-08","2026-09","2026-10"]
methods = {"NESO+empirical residual quantiles (static)": (NLO, NHI), "Raw quantile model (q10,q90)": (LO, HI), "Static CQR": (LO - Qs, HI + Qs), "Rolling CQR (14 d)": (LO - Qr, HI + Qr)}
out = {"overall": {}, "by_block": {}, "by_lead": {}, "by_period": {}}
roll_ok = np.isfinite(Qr)
for nm, (l_, h_) in methods.items():
    m = ev & roll_ok if "Rolling" in nm else ev
    out["overall"][nm] = interval_metrics(y[m], l_[m], h_[m])
    out["by_period"][nm] = {p: interval_metrics(y[m & mk], l_[m & mk], h_[m & mk]) for p, mk in [("2023-24", blk <= 7), ("2026", blk >= 8)]}
    out["by_block"][nm] = {names[k]: interval_metrics(y[ev & (blk == k) & (roll_ok if 'Rolling' in nm else True)], l_[ev & (blk == k) & (roll_ok if 'Rolling' in nm else True)], h_[ev & (blk == k) & (roll_ok if 'Rolling' in nm else True)]) for k in range(1, 12) if (ev & (blk == k)).any()}
    out["by_lead"][nm] = {["0-2h","2-6h","6-12h","12-24h","24-48h"][k]: interval_metrics(y[m & (b == k)], l_[m & (b == k)], h_[m & (b == k)]) for k in range(5)}
out["rolling_available_share"] = float(roll_ok[ev].mean())
json.dump(out, open("/home/claude/results/prob_main.json", "w"), indent=1)
for nm in methods: print(f"{nm:46s}", {k: round(v, 3) for k, v in out["overall"][nm].items()}, "| 2023-24 cov %.3f | 2026 cov %.3f" % (out["by_period"][nm]["2023-24"]["coverage"], out["by_period"][nm]["2026"]["coverage"]))
print("rolling share", out["rolling_available_share"])
