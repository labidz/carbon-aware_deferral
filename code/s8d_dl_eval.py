"""VintageNet (DL) vs static LightGBM trained on identical rows vs walk-forward LightGBM vs NESO; accuracy + interval calibration."""
import numpy as np, pandas as pd, json, lightgbm as lgb, ukciv as U, warnings; warnings.filterwarnings("ignore")
from conformal import rolling_cqr, interval_metrics, bucket
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); X = np.load("/home/claude/data/g_X.npy", mmap_mode="r"); names = list(M["names"])
s, L, f0, y = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], M["y"]; Wm = np.load("/home/claude/data/wf_main.npz"); blk = Wm["blk"]; fi = np.isfinite(y)
res = y - f0; tis = U.TS[s]; T = pd.Timestamp
P = dict(num_leaves=31, min_child_samples=1000, colsample_bytree=0.8, reg_lambda=30, max_bin=63, learning_rate=0.03, subsample=0.7, subsample_freq=1, n_jobs=1, verbose=-1)
NT = {"q10": 140, "q50": 120, "q90": 140}; OBJ = {"q10": 0.1, "q50": 0.5, "q90": 0.9}
out = {}
for mode, tr_m, test_blocks in [("A", (tis >= T("2023-03-13")) & (tis < T("2023-10-27")) & fi & (s % 2 == 0), [6, 7]), ("B", (tis >= T("2023-03-13")) & (tis < T("2024-01-05")) & fi & (s % 2 == 0), [8, 9, 10, 11])]:
    dl = np.load(f"/home/claude/data/dl_{mode}.npz"); te = np.isin(blk, test_blocks) & fi
    Xtr = np.asarray(X[np.where(tr_m)[0]]); st_pred = {}
    for q in ["q10", "q50", "q90"]:
        m = lgb.LGBMRegressor(n_estimators=NT[q], objective="quantile", alpha=OBJ[q], **P).fit(Xtr, res[tr_m]); st_pred[q] = m.predict(np.asarray(X[np.where(te)[0]]))
    idx = np.where(te)[0]; r = {}
    cands = {"NESO": np.zeros(len(idx)), "LightGBM-static (same training rows)": st_pred["q50"], "VintageNet (DL, static)": dl["q50"][idx], "LightGBM-walk-forward (monthly retrain)": Wm["q50"][idx]}
    for nm, v in cands.items(): r[nm] = dict(mae=float(np.mean(np.abs(res[idx] - v))), by_block={str(b): float(np.mean(np.abs(res[idx][blk[idx] == b] - v[blk[idx] == b]))) for b in test_blocks})
    # calibration: raw 80% quantile intervals and rolling CQR (history only inside the evaluation blocks => min_n relaxed)
    for nm, (lo, hi) in {"LightGBM-static": (st_pred["q10"], st_pred["q90"]), "VintageNet": (dl["q10"][idx], dl["q90"][idx]), "LightGBM-walk-forward": (Wm["q10"][idx], Wm["q90"][idx])}.items():
        lo_, hi_ = np.minimum(lo, hi), np.maximum(lo, hi); raw = interval_metrics(res[idx], lo_, hi_)
        Q = rolling_cqr(s[idx], L[idx], res[idx], lo_, hi_, np.ones(len(idx), bool), min_n=1500); okq = np.isfinite(Q)
        rc = interval_metrics(res[idx][okq], (lo_ - Q)[okq], (hi_ + Q)[okq]); r[f"interval|{nm}"] = dict(raw=raw, rolling_cqr=rc, rolling_share=float(okq.mean()))
    out[mode] = r
    print(mode, {k: round(v["mae"], 3) for k, v in r.items() if "mae" in v})
    for k, v in r.items():
        if k.startswith("interval"): print("  ", k, "raw cov %.3f width %.1f | rollCQR cov %.3f width %.1f IS %.1f" % (v["raw"]["coverage"], v["raw"]["width"], v["rolling_cqr"]["coverage"], v["rolling_cqr"]["width"], v["rolling_cqr"]["interval_score"]))
json.dump(out, open("/home/claude/results/dl_eval.json", "w"), indent=1)
