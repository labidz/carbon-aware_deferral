"""Regularised VintageNet (3 seeds, relative-level inputs, dropout 0.3, L2) - seed spread and ensemble vs same-row LightGBM and NESO."""
import numpy as np, json, os, ukciv as U, lightgbm as lgb, pandas as pd, warnings; warnings.filterwarnings("ignore")
from conformal import rolling_cqr, interval_metrics
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); X = np.load("/home/claude/data/g_X.npy", mmap_mode="r")
s, L, f0, y = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], M["y"]; Wm = np.load("/home/claude/data/wf_main.npz"); blk = Wm["blk"]; fi = np.isfinite(y); res = y - f0; tis = U.TS[s]; T = pd.Timestamp
old = json.load(open("/home/claude/results/dl_eval.json")); out = {}
for mode, blocks in [("A", [6, 7]), ("B", [8, 9, 10, 11])]:
    te = np.isin(blk, blocks) & fi; idx = np.where(te)[0]; seeds = [k for k in (1, 2, 3) if os.path.exists(f"/home/claude/data/dl_{mode}_r{k}.npz")]
    if not seeds: continue
    Q = {q: np.stack([np.load(f"/home/claude/data/dl_{mode}_r{k}.npz")[q][idx] for k in seeds]) for q in ["q10", "q50", "q90"]}
    r = {"seeds": seeds, "NESO": float(np.mean(np.abs(res[idx]))), "member_mae": [float(np.mean(np.abs(res[idx] - Q["q50"][i]))) for i in range(len(seeds))]}
    ens = {q: Q[q].mean(0) for q in Q}; r["ensemble_mae"] = float(np.mean(np.abs(res[idx] - ens["q50"])))
    r["lgbm_static_mae"] = old[mode]["LightGBM-static (same training rows)"]["mae"]; r["vintagenet_v1_mae"] = old[mode]["VintageNet (DL, static)"]["mae"]; r["lgbm_walkforward_mae"] = old[mode]["LightGBM-walk-forward (monthly retrain)"]["mae"]
    lo, hi = np.minimum(ens["q10"], ens["q90"]), np.maximum(ens["q10"], ens["q90"]); r["raw_interval"] = interval_metrics(res[idx], lo, hi)
    Qc = rolling_cqr(s[idx], L[idx], res[idx], lo, hi, np.ones(len(idx), bool), min_n=1500); ok = np.isfinite(Qc); r["rolling_cqr"] = interval_metrics(res[idx][ok], (lo - Qc)[ok], (hi + Qc)[ok]); r["rolling_share"] = float(ok.mean())
    out[mode] = r; print(mode, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)}, "| raw cov %.3f | rollCQR cov %.3f width %.1f IS %.1f" % (r["raw_interval"]["coverage"], r["rolling_cqr"]["coverage"], r["rolling_cqr"]["width"], r["rolling_cqr"]["interval_score"]))
json.dump(out, open("/home/claude/results/dl_ensemble.json", "w"), indent=1)
