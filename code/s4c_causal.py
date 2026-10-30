"""Causal (no look-ahead) hyper-parameter selection: for each block, choose among 4 configs by validation MAE on the PREVIOUS block
(models trained strictly before that block), then retrain the chosen config on all data before the target block (median model)."""
import numpy as np, pandas as pd, lightgbm as lgb, time, json, warnings; warnings.filterwarnings("ignore")
import ukciv as U
X = np.load("/home/claude/data/g_X.npy", mmap_mode="r"); M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True)
s, L, f0, y = M["s"], M["L"], M["f0"], M["y"]; res = y - f0; tis = U.TS[s]; fi = np.isfinite(y); even = (s % 2 == 0)
BLOCKS = [("2023-06-01","2023-07-01"),("2023-07-01","2023-08-01"),("2023-08-01","2023-09-01"),("2023-09-01","2023-10-01"),("2023-10-01","2023-11-01"),("2023-11-01","2023-12-01"),
          ("2023-12-01","2024-01-01"),("2024-01-01","2024-02-06"),("2026-07-01","2026-08-01"),("2026-08-01","2026-09-01"),("2026-09-01","2026-10-01"),("2026-10-01","2026-10-26")]
base = dict(learning_rate=0.03, subsample=0.7, subsample_freq=1, n_jobs=1, verbose=-1, objective="quantile", alpha=0.5)
CFG = {"A_default": dict(num_leaves=31, min_child_samples=1000, colsample_bytree=0.8, reg_lambda=30, max_bin=63, n_estimators=120),
       "B_regularised": dict(num_leaves=15, min_child_samples=3000, colsample_bytree=0.6, reg_lambda=300, max_bin=63, n_estimators=150),
       "C_flexible": dict(num_leaves=63, min_child_samples=300, colsample_bytree=0.8, reg_lambda=1, max_bin=63, n_estimators=80),
       "D_shallow": dict(num_leaves=7, min_child_samples=3000, colsample_bytree=0.6, reg_lambda=30, max_bin=63, n_estimators=200)}
Xall = np.asarray(X)                                    # 380 MB
def rows(a, b, train=False):
    m = (tis >= pd.Timestamp(a)) & (tis < pd.Timestamp(b)) & fi
    return np.where(m & even)[0] if train else np.where(m)[0]
def fit(idx, cfg): return lgb.LGBMRegressor(**base, **CFG[cfg]).fit(Xall[idx], res[idx])
import os
PART = "/home/claude/data/wf_causal_partial.npz"; PJ = "/home/claude/results/causal_choice.json"
out = np.full(len(s), np.nan, np.float32); chosen = {}; t0 = time.time()
if os.path.exists(PART) and os.path.exists(PJ): out = np.load(PART)["q50"].copy(); chosen = json.load(open(PJ)); print("resuming; done blocks:", list(chosen), flush=True)
for k in range(1, len(BLOCKS)):
    a, b = BLOCKS[k]; pa, pb = BLOCKS[k - 1]
    if a in chosen: continue
    tr_v = rows("2023-03-01", str((pd.Timestamp(pa) - pd.Timedelta(days=2)).date()), train=True); va = rows(pa, pb)
    scores = {c: float(np.mean(np.abs(res[va] - fit(tr_v, c).predict(Xall[va])))) for c in CFG}
    best = min(scores, key=scores.get); chosen[a] = dict(best=best, val_mae=scores)
    tr = rows("2023-03-01", str((pd.Timestamp(a) - pd.Timedelta(days=2)).date()), train=True); te = rows(a, b)
    out[te] = fit(tr, best).predict(Xall[te])
    np.savez(PART, q50=out); json.dump(chosen, open(PJ, "w"), indent=1)
    print(f"block {a}: chosen {best}  val {json.dumps({c: round(v, 2) for c, v in scores.items()})}  test MAE {np.mean(np.abs(res[te] - out[te])):.2f}  ({time.time()-t0:.0f}s)", flush=True)
np.savez("/home/claude/data/wf_causal.npz", q50=out); json.dump(chosen, open("/home/claude/results/causal_choice.json", "w"), indent=1)
print("DONE")
