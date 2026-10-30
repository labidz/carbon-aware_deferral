"""Accuracy of naive / climatology / stand-alone / causal-tuned forecasts vs NESO and ML-median (paired day-block bootstrap)."""
import numpy as np, json, ukciv as U, sim, warnings; warnings.filterwarnings("ignore")
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); X = np.load("/home/claude/data/g_X.npy", mmap_mode="r"); names = list(M["names"])
s, L, f0, y = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], M["y"]; Wm = np.load("/home/claude/data/wf_main.npz"); blk = Wm["blk"]
ev = (blk >= 1) & np.isfinite(y); col = lambda n: np.asarray(X[:, names.index(n)])
clim = np.load("/home/claude/results/clim.npy"); ci = np.searchsorted(sim.CAPS, s); climv = clim[np.clip(ci, 0, sim.NC - 1), L]
cand = {"NESO (operator)": f0, "Persistence (latest actual)": np.where(np.isfinite(col("a_last")), col("a_last"), np.nan), "Seasonal naive (yesterday)": np.where(np.isfinite(col("a_yday")), col("a_yday"), col("a_2d")),
        "Weekly naive (last week)": col("a_lw"), "Climatology (14-day mean by time of day)": climv,
        "Stand-alone LightGBM (no operator forecast)": f0 + np.load("/home/claude/data/wf_standalone.npz")["q50"], "ML median, fixed hyper-parameters": f0 + np.nan_to_num(Wm["q50"]),
        "ML median, causally tuned": f0 + np.load("/home/claude/data/wf_causal.npz")["q50"]}
common = ev.copy()
for v in cand.values(): common &= np.isfinite(v)
day = (s + L) // 48; ud, inv = np.unique(day[common], return_inverse=True); rng = np.random.default_rng(9); draws = rng.integers(0, len(ud), (1000, len(ud)))
def agg(v): e = np.abs(y - v)[common]; return np.bincount(inv, weights=e, minlength=len(ud)), np.bincount(inv, minlength=len(ud)).astype(float)
AG = {k: agg(v) for k, v in cand.items()}; LB = np.digitize(L, [4, 12, 24, 48]); out = {"n_common": int(common.sum())}
for k, v in cand.items():
    S_, C_ = AG[k]; Sn, Cn = AG["NESO (operator)"]; Sm, Cm = AG["ML median, fixed hyper-parameters"]
    rel = lambda Sb, Cb: (S_[draws].sum(1) / C_[draws].sum(1)) / (Sb[draws].sum(1) / Cb[draws].sum(1)) - 1
    d1, d2 = rel(Sn, Cn), rel(Sm, Cm)
    out[k] = dict(mae=float(S_.sum() / C_.sum()), vs_neso_pct=float(100 * ((S_.sum() / C_.sum()) / (Sn.sum() / Cn.sum()) - 1)), vs_neso_ci=[float(100 * np.percentile(d1, 2.5)), float(100 * np.percentile(d1, 97.5))],
                  vs_ml_pct=float(100 * ((S_.sum() / C_.sum()) / (Sm.sum() / Cm.sum()) - 1)), vs_ml_ci=[float(100 * np.percentile(d2, 2.5)), float(100 * np.percentile(d2, 97.5))],
                  by_lead={n: float(np.mean(np.abs(y - v)[common & (LB == i)])) for i, n in enumerate(["0-2h", "2-6h", "6-12h", "12-24h", "24-48h"])},
                  by_period={"2023-24": float(np.mean(np.abs(y - v)[common & (blk <= 7)])), "2026": float(np.mean(np.abs(y - v)[common & (blk >= 8)]))})
json.dump(out, open("/home/claude/results/forecast_extras.json", "w"), indent=1)
for k in cand: print(f"{k:46s} MAE {out[k]['mae']:6.2f}  vs NESO {out[k]['vs_neso_pct']:+6.1f}% [{out[k]['vs_neso_ci'][0]:+.1f},{out[k]['vs_neso_ci'][1]:+.1f}]  by lead", {a: round(b, 1) for a, b in out[k]['by_lead'].items()})
