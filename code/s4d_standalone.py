"""Stand-alone LightGBM (no operator-forecast features) walk-forward; predicts y directly; stored as residual vs f0 for comparability."""
import numpy as np, pandas as pd, lightgbm as lgb, time, warnings; warnings.filterwarnings("ignore")
import ukciv as U
X = np.load("/home/claude/data/g_X.npy", mmap_mode="r"); M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True)
s, L, f0, y, names = M["s"], M["L"], M["f0"], M["y"], list(M["names"]); tis = U.TS[s]; fi = np.isfinite(y); even = (s % 2 == 0)
KEEP = ["lead", "tod", "tod_sin", "tod_cos", "dow", "weekend", "a_last", "a_mean4", "a_mean12", "a_mean48", "a_trend", "a_yday", "a_2d", "a_lw"] + [f"mix_{c}" for c in ["wind","solar","gas","coal","imports","nuclear","biomass","hydro"]] + ["mix_wind_d6", "mix_gas_d6"]
ci = [names.index(n) for n in KEEP]; Xs = np.asarray(X)[:, ci]
BLOCKS = [("2023-06-01","2023-07-01"),("2023-07-01","2023-08-01"),("2023-08-01","2023-09-01"),("2023-09-01","2023-10-01"),("2023-10-01","2023-11-01"),("2023-11-01","2023-12-01"),
          ("2023-12-01","2024-01-01"),("2024-01-01","2024-02-06"),("2026-07-01","2026-08-01"),("2026-08-01","2026-09-01"),("2026-09-01","2026-10-01"),("2026-10-01","2026-10-26")]
P = dict(num_leaves=31, min_child_samples=1000, colsample_bytree=0.8, reg_lambda=30, max_bin=63, learning_rate=0.03, subsample=0.7, subsample_freq=1, n_jobs=1, verbose=-1, objective="quantile", alpha=0.5, n_estimators=250)
out = np.full(len(s), np.nan, np.float32); t0 = time.time()
for k, (a, b) in enumerate(BLOCKS):
    te = np.where((tis >= pd.Timestamp(a)) & (tis < pd.Timestamp(b)) & fi)[0]; tr = np.where((tis < pd.Timestamp(a) - pd.Timedelta(days=2)) & fi & even)[0]
    m = lgb.LGBMRegressor(**P).fit(Xs[tr], y[tr]); out[te] = m.predict(Xs[te]) - f0[te]
    print(k, a, "standalone MAE", round(float(np.mean(np.abs(y[te] - f0[te] - out[te]))), 2), "NESO", round(float(np.mean(np.abs(y[te] - f0[te]))), 2), f"{time.time()-t0:.0f}s", flush=True)
np.savez("/home/claude/data/wf_standalone.npz", q50=out); print("DONE")
