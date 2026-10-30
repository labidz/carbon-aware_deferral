"""Transfer test: pooled LightGBM post-processor for CarbonCast forecasts on 13 grids (train Jul-Sep 2021, test Oct-Dec); accuracy and decision value."""
import numpy as np, pandas as pd, json, time, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from ext_data import load_region, REGIONS
import ext_sim as X
t00 = time.time(); TSPLIT = 92 * 24; H = 96
wl = json.load(open("/home/claude/results/workload.json"))["all_gpu_jobs"]["bins"]; sh = [b["gpu_hour_share"] for b in wl if b["slots"]]
WD = {1: sh[0] + sh[1], 2: sh[2], 4: sh[3], 8: sh[4], 12: sh[5]}; DS = [1, 2, 4, 8, 12]; SS = [3, 6, 12, 24, 48]
def build(R, rid):
    A, bs, F = R["A"], R["bstart"], R["F"]; T = len(A); rows = []; 
    idx_of = {int(b): i for i, b in enumerate(bs)}
    for k in range(len(bs)):
        t = int(bs[k]); lead = np.arange(H); w = t + lead; ok = w < T
        f0 = F[k]; A_w = np.where(ok, A[np.clip(w, 0, T - 1)], np.nan)
        prev = idx_of.get(t - 24)
        r1 = np.full(H, np.nan); 
        if prev is not None: r1[:H - 24] = f0[:H - 24] - F[prev, 24:]
        # previous day's realised error of the previous issue (windows t-24..t-1 have leads 0..23 in block k-1)
        ea = np.nan; a24 = np.nan; alast = np.nan
        if prev is not None and t >= 24:
            h = np.arange(t - 24, t); ea = np.nanmean(A[h] - F[prev, h - (t - 24)]); a24 = np.nanmean(A[h]); alast = A[t - 1]
        e_lag = np.full(H, np.nan); a_lag = np.full(H, np.nan)
        for m in range(1, 5):
            pk = idx_of.get(t - 24 * m); sel = np.arange(24 * (m - 1), 24 * m)
            if pk is not None and t - 24 * m >= 0:
                wl_ = w[sel] - 24 * m; e_lag[sel] = A[np.clip(wl_, 0, T - 1)] - F[pk, sel]; a_lag[sel] = A[np.clip(wl_, 0, T - 1)]
        fprev = np.r_[np.nan, f0[:-1]]; fnext = np.r_[f0[1:], np.nan]
        for l in range(H):
            rows.append((rid, k, t, l, (t + l) % 24, ((t + l) // 24) % 7, f0[l], r1[l], ea, a24, alast, e_lag[l], a_lag[l], f0[l] - a_lag[l], fprev[l], fnext[l], np.nanmean(f0[:24]), np.nanstd(f0[:24]), A_w[l] - f0[l]))
    return pd.DataFrame(rows, columns=["rid", "k", "t", "lead", "hod", "dow", "f0", "r1", "ea", "a24", "alast", "e_lag", "a_lag", "f0_m_alag", "fprev", "fnext", "pm24", "psd24", "y_res"])
D = pd.concat([build(load_region(r), i) for i, r in enumerate(REGIONS)], ignore_index=True); print("samples", D.shape, f"{time.time()-t00:.0f}s", flush=True)
FEATS = ["rid", "lead", "hod", "dow", "f0", "r1", "ea", "a24", "alast", "e_lag", "a_lag", "f0_m_alag", "fprev", "fnext", "pm24", "psd24"]
VARS = {"full": FEATS, "no_revision": [f for f in FEATS if f != "r1"], "no_recent_errors": [f for f in FEATS if f not in ("ea", "e_lag", "a_lag", "f0_m_alag", "alast", "a24")], "calendar_only": ["rid", "lead", "hod", "dow", "f0"]}
tr = D[(D.t <= 2088) & D.y_res.notna()]; te = D[(D.t >= 2184) & D.y_res.notna()]
P = dict(objective="quantile", alpha=0.5, n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=200, colsample_bytree=0.8, reg_lambda=10, subsample=0.8, subsample_freq=1, n_jobs=1, verbose=-1)
preds = {}; out = {"accuracy": {}, "decisions": {}}
for nm, fs in VARS.items():
    m = lgb.LGBMRegressor(**P).fit(tr[fs], tr.y_res); preds[nm] = m.predict(te[fs])
    out["accuracy"][nm] = {r: dict(raw=float(np.mean(np.abs(te.y_res[te.rid == i]))), corrected=float(np.mean(np.abs(te.y_res[te.rid == i] - preds[nm][(te.rid == i).values]))), n=int((te.rid == i).sum())) for i, r in enumerate(REGIONS)}
    out["accuracy"][nm]["_pooled"] = dict(raw=float(np.mean(np.abs(te.y_res))), corrected=float(np.mean(np.abs(te.y_res - preds[nm]))))
    print(nm, {k: round(v, 2) for k, v in out["accuracy"][nm]["_pooled"].items()}, f"{time.time()-t00:.0f}s", flush=True)
# decision value per region: corrected path for test blocks (blocks before TSPLIT-24 keep raw forecast, they are never used for test arrivals)
def capture(env, ps, ref, pol, S):
    num = den = 0
    for d in DS:
        a = ref[(d, S)]["arr"]; off = X.run_policies(env, a, d, S, ps)[pol]; c = env.realised(a + off, d); asap = ref[(d, S)]["asap"]; orc = ref[(d, S)]["orc"]
        num += WD[d] * (1 - c.sum() / asap.sum()); den += WD[d] * (1 - orc.sum() / asap.sum())
    return num / den
for i, r in enumerate(REGIONS):
    R = load_region(r); env = X.Env(R["A"], R["bstart"]); O = X.oracle_matrix(env); orc_ps = X.PathSet(env, O); real_ps = X.PathSet(env, R["F"]); arrs = np.arange(TSPLIT, env.T); ref = {}
    sub = te[te.rid == i]; mapk = {int(t): k for k, t in enumerate(R["bstart"])}
    paths = {}
    for nm in VARS:
        Fc = R["F"].copy(); pr = pd.Series(preds[nm], index=te.index)[sub.index]
        for (k_, l_, p_) in zip(sub.k.values, sub.lead.values, pr.values): Fc[k_, l_] = R["F"][k_, l_] + p_
        paths[nm] = X.PathSet(env, Fc)
    for d in DS:
        for S in SS:
            ok = X.valid_arrivals(env, arrs, d, S, [real_ps, orc_ps] + list(paths.values())); a = arrs[ok]
            ref[(d, S)] = dict(arr=a, asap=env.realised(a, d), orc=env.realised(a + X.run_policies(env, a, d, S, orc_ps)["static"], d))
    out["decisions"][r] = {"raw": {pol: {S: float(capture(env, real_ps, ref, pol, S)) for S in SS} for pol in ["static", "replan"]}}
    for nm in VARS: out["decisions"][r][nm] = {pol: {S: float(capture(env, paths[nm], ref, pol, S)) for S in SS} for pol in ["static", "replan"]}
    print(r, "raw replan S12 %.3f  full %.3f  no_rev %.3f  no_recent %.3f  cal %.3f" % tuple(out["decisions"][r][k]["replan"][12] for k in ["raw", "full", "no_revision", "no_recent_errors", "calendar_only"]), f"{time.time()-t00:.0f}s", flush=True)
json.dump(out, open("/home/claude/results/external_ml.json", "w"), indent=1); print("DONE")
