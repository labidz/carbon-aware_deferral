"""Walk-forward (expanding-window, monthly retrain) evaluation of the vintage-aware post-processor.
usage: python s4b_walkforward.py VARIANT
"""
import numpy as np, pandas as pd, lightgbm as lgb, sys, time, json, warnings; warnings.filterwarnings("ignore")
import ukciv as U
variant = sys.argv[1]
X = np.load("/home/claude/data/g_X.npy", mmap_mode="r"); M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True)
s, L, f0, y, names = M["s"], M["L"], M["f0"], M["y"], list(M["names"])
res = (y - f0); tis = U.TS[s]
BLOCKS = [("2023-06-01","2023-07-01"),("2023-07-01","2023-08-01"),("2023-08-01","2023-09-01"),("2023-09-01","2023-10-01"),("2023-10-01","2023-11-01"),
          ("2023-11-01","2023-12-01"),("2023-12-01","2024-01-01"),("2024-01-01","2024-02-06"),
          ("2026-07-01","2026-08-01"),("2026-08-01","2026-09-01"),("2026-09-01","2026-10-01"),("2026-10-01","2026-10-26")]
GROUPS = {
 "base": ["f0", "lead", "tod", "tod_sin", "tod_cos", "dow", "weekend"],
 "path": ["path_mean", "path_min", "path_max", "path_sd", "f0_rel", "f_prev", "f_next", "f_slope", "f_loc7", "f_rank48"],
 "real": ["a_last", "a_mean4", "a_mean12", "a_mean48", "a_trend", "nowc_err4", "nowc_err24", "f0_minus_alast", "e_same4", "e_same24", "e_same_last", "a_yday", "a_2d", "a_lw", "e_yday", "f0_minus_yday"],
 "adapt": ["e_wk", "e_mo", "e_tod_wk", "e_tod_mo"],
 "vint": [f"lag{k}" for k in [1,2,3,4,6,8,12,16,24,32,48]] + [f"rev{k}" for k in [1,2,4,8,12,24,48]] + ["vint_mean", "vint_std", "vint_n", "rev_total"],
 "mix":  [f"mix_{c}" for c in ["wind","solar","gas","coal","imports","nuclear","biomass","hydro"]] + ["mix_wind_d6", "mix_gas_d6"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(names)
ABS = [f"lag{k}" for k in [1,2,3,4,6,8,12,16,24,32,48]] + ["path_mean","path_min","path_max","f_prev","f_next","f_loc7","a_last","a_mean4","a_mean12","a_mean48","a_yday","a_2d","a_lw","vint_mean"]
V = {  # variant -> (groups, relative?, halflife_days, models)
 "main":       (["base","path","real","adapt","vint","mix"], False, None, ["mean","q10","q50","q90"]),
 "main_w60":   (["base","path","real","adapt","vint","mix"], False, 60,   ["mean"]),
 "no_vint":    (["base","path","real","adapt","mix"], False, None, ["mean"]),
 "no_adapt":   (["base","path","real","vint","mix"], False, None, ["mean"]),
 "no_real":    (["base","path","adapt","vint","mix"], False, None, ["mean"]),
 "no_mix":     (["base","path","real","adapt","vint"], False, None, ["mean"]),
 "no_path":    (["base","real","adapt","vint","mix"], False, None, ["mean"]),
 "base_only":  (["base"], False, None, ["mean"]),
 "base_adapt": (["base","adapt"], False, None, ["mean"]),
 "base_vint":  (["base","vint"], False, None, ["mean"]),
 "base_real":  (["base","real"], False, None, ["mean"]),
 "base_vint_path": (["base","path","vint"], False, None, ["mean"]),
 "relative":   (["base","path","real","adapt","vint","mix"], True, None, ["mean"]),
}
groups, rel, hl, models = V[variant]
cols = [n for g in groups for n in GROUPS[g]]; ci = [names.index(n) for n in cols]
P = dict(num_leaves=31, min_child_samples=1000, colsample_bytree=0.8, reg_lambda=30, max_bin=63, learning_rate=0.03, subsample=0.7, subsample_freq=1, n_jobs=1, verbose=-1)
NT = {"mean": 110, "q10": 140, "q50": 120, "q90": 140}
OBJ = {"mean": dict(objective="l2"), "q10": dict(objective="quantile", alpha=0.1), "q50": dict(objective="quantile", alpha=0.5), "q90": dict(objective="quantile", alpha=0.9)}
n = len(s); OUT = {m: np.full(n, np.nan, np.float32) for m in models}; blk = np.full(n, -1, np.int8)
fi = np.isfinite(y); even = (s % 2 == 0)
def feats(idx):
    Xi = np.asarray(X[idx][:, ci]).astype(np.float32)
    if rel:
        for j, nm in enumerate(cols):
            if nm in ABS: Xi[:, j] -= f0[idx]
    return Xi
t00 = time.time(); log = []
for k, (a, b) in enumerate(BLOCKS):
    ta, tb = pd.Timestamp(a), pd.Timestamp(b)
    test = np.where((tis >= ta) & (tis < tb) & fi)[0]
    tr = np.where((tis < ta - pd.Timedelta(days=2)) & fi & even)[0]
    if len(test) == 0 or len(tr) == 0: continue
    blk[test] = k
    Xtr, ytr = feats(tr), res[tr]; Xte = feats(test)
    w = None if hl is None else 0.5 ** (((ta - tis[tr]).total_seconds() / 86400.0) / hl)
    for m in models:
        mdl = lgb.LGBMRegressor(n_estimators=NT[m], **OBJ[m], **P); mdl.fit(Xtr, ytr, sample_weight=w)
        OUT[m][test] = mdl.predict(Xte)
    neso = float(np.mean(np.abs(res[test]))); mm = float(np.mean(np.abs(res[test] - OUT["mean"][test])))
    log.append((a, len(tr), len(test), neso, mm)); print(f"block {k} {a} train {len(tr)} test {len(test)}  NESO {neso:.2f}  {variant}-mean {mm:.2f}  ({time.time()-t00:.0f}s)", flush=True)
np.savez(f"/home/claude/data/wf_{variant}.npz", blk=blk, **OUT)
json.dump(log, open(f"/home/claude/results/wf_{variant}.json", "w"))
print("DONE", variant, f"{time.time()-t00:.0f}s")
