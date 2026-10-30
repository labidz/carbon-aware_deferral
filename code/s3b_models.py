"""Step 3b: final LightGBM models (mean, median, quantiles), feature-group ablations, blocked cross-fitting."""
import numpy as np, time, json, lightgbm as lgb, warnings, pickle, sys; warnings.filterwarnings("ignore")
def load(n): return dict(np.load(f"/home/claude/data/samples_{n}.npz", allow_pickle=True))
D = {n: load(n) for n in ["train", "val", "cal", "testA", "testB"]}
names = list(D["train"]["names"])
tune = json.load(open("/home/claude/results/tuning.json")); best = min(tune, key=lambda r: r["val_mae"])
P = dict(num_leaves=int(best["num_leaves"]), min_child_samples=int(best["min_child_samples"]), colsample_bytree=float(best["colsample_bytree"]),
         reg_lambda=float(best["reg_lambda"]), max_bin=int(best["max_bin"]), learning_rate=0.03, subsample=0.7, subsample_freq=1, n_jobs=1, verbose=-1)
print("params", P, flush=True)

GROUPS = {
 "base": ["f0", "lead", "tod", "tod_sin", "tod_cos", "dow", "weekend"],
 "path": ["path_mean", "path_min", "path_max", "path_sd", "f0_rel", "f_prev", "f_next", "f_slope", "f_loc7", "f_rank48"],
 "real": ["a_last", "a_mean4", "a_mean12", "a_mean48", "a_trend", "nowc_err4", "nowc_err24", "f0_minus_alast", "e_same4", "e_same24", "e_same_last", "a_yday", "a_2d", "a_lw", "e_yday", "f0_minus_yday"],
 "vint": [f"lag{k}" for k in [1,2,3,4,6,8,12,16,24,32,48]] + [f"rev{k}" for k in [1,2,4,8,12,24,48]] + ["vint_mean", "vint_std", "vint_n", "rev_total"],
 "mix":  [f"mix_{c}" for c in ["wind","solar","gas","coal","imports","nuclear","biomass","hydro"]] + ["mix_wind_d6", "mix_gas_d6"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(names), set(names) ^ set(sum(GROUPS.values(), []))
ABS = [f"lag{k}" for k in [1,2,3,4,6,8,12,16,24,32,48]] + ["path_mean","path_min","path_max","f_prev","f_next","f_loc7","a_last","a_mean4","a_mean12","a_mean48","a_yday","a_2d","a_lw","vint_mean"]
def cols(gs, drop=()): return [n for g in gs for n in GROUPS[g] if n not in drop]
def sel(d, c, rel=False):
    X = d["X"][:, [names.index(n) for n in c]].copy()
    if rel:
        f0 = d["f0"]
        for j, n in enumerate(c):
            if n in ABS: X[:, j] = X[:, j] - f0
    return X

def fit(X, y, Xv, yv, obj="l2", alpha=None, n_est=None, params=P):
    kw = dict(params); 
    if obj == "quantile": kw.update(objective="quantile", alpha=alpha)
    elif obj == "l1": kw.update(objective="l1")
    else: kw.update(objective="l2")
    if n_est is None:
        m = lgb.LGBMRegressor(n_estimators=2500, **kw)
        met = "quantile" if obj == "quantile" else "l1"
        m.fit(X, y, eval_set=[(Xv, yv)], eval_metric=met, callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        m = lgb.LGBMRegressor(n_estimators=n_est, **kw); m.fit(X, y)
    return m

R = {"params": P}; PRED = {}
resid = {n: D[n]["y"] - D[n]["f0"] for n in D}
def mae_on(n, pred_resid): return float(np.mean(np.abs(resid[n] - pred_resid)))
t00 = time.time()

# ---------- A. final model family (full features) ----------
cf = cols(list(GROUPS)); final = {}
for tag, (obj, a) in {"mean": ("l2", None), "q50": ("quantile", 0.5), "q05": ("quantile", .05), "q10": ("quantile", .10), "q25": ("quantile", .25),
                      "q75": ("quantile", .75), "q90": ("quantile", .90), "q95": ("quantile", .95)}.items():
    t = time.time()
    m = fit(sel(D["train"], cf), resid["train"], sel(D["val"], cf), resid["val"], obj, a)
    final[tag] = m
    for n in ["val", "cal", "testA", "testB"]:
        PRED[(tag, n)] = m.predict(sel(D[n], cf)).astype(np.float32)
    R[f"iters_{tag}"] = int(m.best_iteration_); print(tag, "iters", m.best_iteration_, f"{time.time()-t:.0f}s", flush=True)
pickle.dump({k: v.booster_.model_to_string() for k, v in final.items()}, open("/home/claude/data/lgbm_final.pkl", "wb"))
R["final"] = {n: {"NESO": float(np.mean(np.abs(resid[n]))), "lgbm_mean": mae_on(n, PRED[("mean", n)]), "lgbm_median": mae_on(n, PRED[("q50", n)])} for n in ["val", "cal", "testA", "testB"]}
print(json.dumps(R["final"], indent=1), flush=True)

# ---------- B. ablations (mean model; early stopping on val) ----------
abl = {}
def run_abl(name, c, rel=False):
    m = fit(sel(D["train"], c, rel), resid["train"], sel(D["val"], c, rel), resid["val"], "l2")
    abl[name] = {n: mae_on(n, m.predict(sel(D[n], c, rel))) for n in ["val", "testA", "testB"]} | {"iters": int(m.best_iteration_), "n_feat": len(c)}
    print("ABL", name, {k: round(v, 3) for k, v in abl[name].items()}, flush=True)
order = ["base", "path", "real", "vint", "mix"]
for i in range(1, 6): run_abl("cum:" + "+".join(order[:i]), cols(order[:i]))
for g in order: run_abl("loo:no_" + g, cols([x for x in order if x != g]))
run_abl("only:vint+base", cols(["base", "vint"]))
run_abl("rel:full_relative", cf, rel=True)
c_norel_f0 = [n for n in cf if n != "f0"]; run_abl("rel:full_relative_nof0", c_norel_f0, rel=True)
c_nolevel = [n for n in cf if n not in ABS and n not in ("f0", "path_mean")]; run_abl("rel:no_absolute_levels", c_nolevel)
R["ablation"] = abl
json.dump(R, open("/home/claude/results/forecast_models.json", "w"), indent=1)
pickle.dump(PRED, open("/home/claude/data/pred_lgbm.pkl", "wb"))

# ---------- C. blocked cross-fitting on train (for decision-policy training) ----------
import pandas as pd, ukciv as U
iss = pd.to_datetime(U.t0 + 1800 * D["train"]["s"], unit="s")
bounds = ["2023-03-01", "2023-05-01", "2023-07-01", "2023-09-01", "2023-11-01"]
fold = np.digitize(iss.values.astype("datetime64[s]").astype(np.int64), [pd.Timestamp(b).value // 10**9 for b in bounds[1:-1]])
OOF = {t: np.full(len(fold), np.nan, np.float32) for t in ["mean", "q10", "q50", "q90"]}
emb = pd.Timedelta(days=2)
for k in range(4):
    lo, hi = pd.Timestamp(bounds[k]) - emb, pd.Timestamp(bounds[k + 1]) + emb
    test_m = fold == k; train_m = ~((iss >= lo) & (iss < hi))
    for tag, (obj, a) in {"mean": ("l2", None), "q10": ("quantile", .10), "q50": ("quantile", .5), "q90": ("quantile", .90)}.items():
        m = fit(sel(D["train"], cf)[train_m], resid["train"][train_m], None, None, obj, a, n_est=max(30, int(0.9 * R[f"iters_{tag}"])))
        OOF[tag][test_m] = m.predict(sel(D["train"], cf)[test_m])
    print("fold", k, "done", f"{time.time()-t00:.0f}s", flush=True)
R["oof_mae_mean"] = float(np.mean(np.abs(resid["train"] - OOF["mean"]))); R["oof_neso_mae"] = float(np.mean(np.abs(resid["train"])))
pickle.dump(OOF, open("/home/claude/data/oof_lgbm.pkl", "wb"))
json.dump(R, open("/home/claude/results/forecast_models.json", "w"), indent=1)
print("ALL DONE", f"{time.time()-t00:.0f}s", flush=True)
