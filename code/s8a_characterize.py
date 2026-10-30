import numpy as np, pandas as pd, json, ukciv as U, statsmodels.api as sm, warnings; warnings.filterwarnings("ignore")
X = np.load("/home/claude/data/g_X.npy", mmap_mode="r"); M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True)
s, L, f0, y, names = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], M["y"], list(M["names"])
col = lambda n: np.asarray(X[:, names.index(n)])
e = y - f0; w = s + L; day = (w // 48); per = np.where(U.TS[np.clip(w, 0, U.T-1)] < pd.Timestamp("2025-01-01"), "2023-24", "2026")
fi = np.isfinite(y); out = {}
# ---- (1) error by lead & period, with day-block bootstrap CIs for MAE
rng = np.random.default_rng(0)
leads = [0, 1, 2, 4, 8, 12, 24, 36, 48, 72, 95]
tab = []
for p in ["2023-24", "2026"]:
    for l in leads:
        m = fi & (per == p) & (L == l); ee = e[m]; dd = day[m]
        ud = np.unique(dd); idx = {d: np.where(dd == d)[0] for d in ud}
        bs = [np.mean(np.abs(ee[np.concatenate([idx[d] for d in rng.choice(ud, len(ud))])])) for _ in range(100)]
        tab.append(dict(period=p, lead_h=l/2, n=int(m.sum()), mae=float(np.mean(np.abs(ee))), mae_lo=float(np.percentile(bs, 2.5)), mae_hi=float(np.percentile(bs, 97.5)), bias=float(ee.mean()), rmse=float(np.sqrt((ee**2).mean())),
                        mean_actual=float(y[m].mean())))
out["error_by_lead"] = tab
# ---- (2) monthly bias for L>=12
tw = pd.DataFrame({"m": U.TS[np.clip(w, 0, U.T-1)].to_period("M").astype(str), "e": e, "L": L})[fi]
mm = tw[tw.L >= 12].groupby("m").e.agg(["mean", "count", lambda x: np.abs(x).mean()]); mm.columns = ["bias", "n", "mae"]
out["monthly_bias_L12plus"] = mm.round(2).reset_index().to_dict("records")
# ---- (3) revision efficiency: e = a + b * r4 (rev over last 4 captures = 2h), cluster by target day
def cl_ols(xv, yv, g):
    ok = np.isfinite(xv) & np.isfinite(yv); Xc = sm.add_constant(xv[ok]); r = sm.OLS(yv[ok], Xc).fit(cov_type="cluster", cov_kwds={"groups": g[ok]})
    return dict(slope=float(r.params[1]), se=float(r.bse[1]), t=float(r.tvalues[1]), corr=float(np.corrcoef(xv[ok], yv[ok])[0, 1]), n=int(ok.sum()))
rev = {}
for p in ["2023-24", "2026"]:
    for l in [2, 6, 12, 24, 40, 60]:
        m = fi & (per == p) & (L == l); rev[f"{p}|L={l/2}h"] = cl_ols(col("rev4")[m], e[m], day[m])
out["revision_efficiency_rev4"] = rev
# ---- (4) error persistence: e on mean same-lead error of last 12h (e_same24) ; and on the 7-day (e_wk)
pers = {}
for feat in ["e_same24", "e_wk"]:
    for p in ["2023-24", "2026"]:
        for l in [2, 12, 24, 48, 80]:
            m = fi & (per == p) & (L == l); pers[f"{feat}|{p}|L={l/2}h"] = cl_ols(col(feat)[m], e[m], day[m])
out["error_persistence"] = pers
# ---- (5) diurnal bias at L=24 by period (for heatmap)
tod = col("tod").astype(int); dd = {}
for p in ["2023-24", "2026"]:
    m = fi & (per == p) & (L == 24); dd[p] = [float(e[m & (tod == t)].mean()) for t in range(48)]
out["diurnal_bias_L24"] = dd
# ---- (6) forecast shrinkage vs truth (explains synthetic-noise gap): corr(f, e), corr(A, e), slope of f on A (L=24)
shr = {}
for l in [4, 12, 24, 48]:
    m = fi & (L == l); f_, a_, e_ = f0[m], y[m], e[m]
    shr[f"L={l/2}h"] = dict(corr_f_e=float(np.corrcoef(f_, e_)[0, 1]), corr_A_e=float(np.corrcoef(a_, e_)[0, 1]), slope_f_on_A=float(np.polyfit(a_, f_, 1)[0]), sd_f=float(f_.std()), sd_A=float(a_.std()))
out["shrinkage"] = shr
# ---- (7) heteroscedasticity: |e| by forecast tercile (L>=12)
m = fi & (L >= 12); q = np.quantile(f0[m], [1/3, 2/3]); b = np.digitize(f0[m], q)
out["abs_error_by_forecast_tercile"] = {f"T{i+1}": float(np.abs(e[m][b == i]).mean()) for i in range(3)} | {"cutoffs": q.tolist()}
json.dump(out, open("/home/claude/results/characterize.json", "w"), indent=1)
print(pd.DataFrame(tab).round(2).to_string(index=False)); print(mm.round(1).T.to_string())
for k, v in rev.items(): print("REV", k, {a: round(b, 3) for a, b in v.items()})
for k, v in pers.items(): print("PERS", k, {a: round(b, 3) for a, b in v.items()})
for k, v in shr.items(): print("SHR", k, {a: round(b, 3) for a, b in v.items()})
print(out["abs_error_by_forecast_tercile"])
