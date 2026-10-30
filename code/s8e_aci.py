"""Adaptive conformal inference with delayed feedback (Gibbs & Candes 2021) on top of the walk-forward q10/q90 intervals, per lead bucket."""
import numpy as np, json, ukciv as U, warnings, time; warnings.filterwarnings("ignore")
from conformal import bucket, scores, interval_metrics, NB
t00 = time.time()
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); Wm = np.load("/home/claude/data/wf_main.npz")
s, L, f0, y = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], M["y"]; blk = Wm["blk"]
q10, q90 = np.nan_to_num(Wm["q10"]), np.nan_to_num(Wm["q90"]); lo = np.minimum(q10, q90); hi = np.maximum(q10, q90); res = y - f0
sel = (blk >= 0) & np.isfinite(y); w = s + L; b = bucket(L); sc = np.where(sel, scores(res, lo, hi), np.nan); ok = np.isfinite(sc)
order = np.argsort(w[ok], kind="stable"); W = w[ok][order]; B_ = b[ok][order]; SC = sc[ok][order]; S_iss = s[ok][order]
per_b = [np.where(B_ == k)[0] for k in range(NB)]; Wb = [W[i] for i in per_b]; Sb = [SC[i] for i in per_b]; Ib = [S_iss[i] for i in per_b]
hours = np.unique(s[sel] // 2); H0 = hours.min(); NH = hours.max() - H0 + 1
def run(gamma, K=672, min_n=1500, alpha=0.2):
    a = np.full(NB, alpha); Qs = np.full((NH, NB), np.nan); prev = None
    for hi_ in range(NH):
        t = (H0 + hi_) * 2; hi_w, lo_w = t - 49, t - 49 - K
        for k in range(NB):
            # feedback from targets newly settled since previous update (their stored Q at issue time)
            if gamma > 0 and prev is not None:
                i0, i1 = np.searchsorted(Wb[k], [prev - 49 + 1, t - 49 + 1])
                if i1 > i0:
                    iss = Ib[k][i0:i1] // 2 - H0; Qi = Qs[np.clip(iss, 0, NH - 1), k]; okq = np.isfinite(Qi) & (iss >= 0)
                    if okq.any(): miss = np.mean(Sb[k][i0:i1][okq] > Qi[okq]); a[k] = np.clip(a[k] + gamma * (alpha - miss), 0.01, 0.6)
            i0, i1 = np.searchsorted(Wb[k], [lo_w + 1, hi_w + 1]); n = i1 - i0
            if n >= min_n: Qs[hi_, k] = np.quantile(Sb[k][i0:i1], min(1.0, np.ceil((n + 1) * (1 - a[k])) / n))
        prev = t
    h = s // 2 - H0; Q = np.where((h >= 0) & (h < NH), Qs[np.clip(h, 0, NH - 1), b], np.nan); return Q
out = {}; names = ["2023-06","2023-07","2023-08","2023-09","2023-10","2023-11","2023-12","2024-01/02","2026-07","2026-08","2026-09","2026-10"]
ev = sel & (blk >= 1)
for nm, g in [("rolling CQR (gamma=0)", 0.0), ("ACI gamma=0.005", 0.005), ("ACI gamma=0.02", 0.02), ("ACI gamma=0.05", 0.05)]:
    Q = run(g); m = ev & np.isfinite(Q); met = interval_metrics(res[m], (lo - Q)[m], (hi + Q)[m])
    by_blk = {names[k]: float(np.mean(((res >= (lo - Q)) & (res <= (hi + Q)))[m & (blk == k)])) for k in range(1, 12) if (m & (blk == k)).any()}
    met.update(by_block=by_blk, cov_sd_across_blocks=float(np.std(list(by_blk.values()))), cov_min=float(min(by_blk.values())), cov_max=float(max(by_blk.values())), share=float(m.sum() / ev.sum())); out[nm] = met
    print(nm, {k: (round(v, 3) if not isinstance(v, dict) else "...") for k, v in met.items()}, f"{time.time()-t00:.0f}s", flush=True)
json.dump(out, open("/home/claude/results/aci.json", "w"), indent=1)
