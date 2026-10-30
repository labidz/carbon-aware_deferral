import numpy as np, time, json, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
def load(n): return dict(np.load(f"/home/claude/data/samples_{n}.npz", allow_pickle=True))
tr, va, ta, tb = load("train"), load("val"), load("testA"), load("testB")
names = list(tr["names"])
LB = [(0,4),(4,12),(12,24),(24,48),(48,96)]
def bymae(d, pred):
    r = {}
    for a,b in LB:
        m=(d["L"]>=a)&(d["L"]<b); r[f"{a}-{b}"]=float(np.mean(np.abs(d["y"][m]-pred[m])))
    r["all"]=float(np.mean(np.abs(d["y"]-pred))); return r
res = {}
for nm, d in [("testA", ta), ("testB", tb)]:
    X = d["X"]; g = lambda n: X[:, names.index(n)]
    base = {"NESO": d["f0"], "persist_last_actual": np.where(np.isfinite(g("a_last")), g("a_last"), d["f0"]),
            "seasonal_naive_yday": np.where(np.isfinite(g("a_yday")), g("a_yday"), np.where(np.isfinite(g("a_2d")), g("a_2d"), d["f0"]))}
    res[nm] = {k: bymae(d, v) for k, v in base.items()}
# bias table: lead bucket x tod mean residual (train)
lb = lambda L: np.digitize(L, [4,12,24,48])
key = lambda d: lb(d["L"])*48 + d["X"][:, names.index("tod")].astype(int)
resid = tr["y"] - tr["f0"]; k = key(tr); tab = np.array([resid[k==i].mean() if (k==i).any() else 0 for i in range(5*48)])
for nm, d in [("testA", ta), ("testB", tb)]:
    res[nm]["bias_table"] = bymae(d, d["f0"] + tab[key(d)])
# LightGBM pilot (mean residual, fast)
t = time.time()
ytr = tr["y"] - tr["f0"]; yva = va["y"] - va["f0"]
m = lgb.LGBMRegressor(objective="l2", n_estimators=300, learning_rate=0.1, num_leaves=63, min_child_samples=200, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10, n_jobs=1, verbose=-1)
m.fit(tr["X"], ytr, eval_set=[(va["X"], yva)], eval_metric="l1", callbacks=[lgb.early_stopping(50, verbose=False)])
print("pilot fit time", round(time.time()-t,1), "s; best_iter", m.best_iteration_)
for nm, d in [("testA", ta), ("testB", tb)]:
    res[nm]["lgbm_mean_pilot"] = bymae(d, d["f0"] + m.predict(d["X"]))
for nm in res:
    print("\n", nm)
    for k, v in res[nm].items(): print(f"  {k:24s}", {a: round(b,2) for a,b in v.items()})
json.dump(res, open("/home/claude/results/pilot.json","w"), indent=1)
