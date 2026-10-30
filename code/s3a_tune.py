import numpy as np, time, json, lightgbm as lgb, warnings, itertools; warnings.filterwarnings("ignore")
def load(n): return dict(np.load(f"/home/claude/data/samples_{n}.npz", allow_pickle=True))
tr, va = load("train"), load("val")
ytr = tr["y"] - tr["f0"]; yva = va["y"] - va["f0"]
rng = np.random.default_rng(42)
space = dict(num_leaves=[7,15,31,63], min_child_samples=[300,1000,3000], colsample_bytree=[0.4,0.6,0.8], reg_lambda=[1,30,300], max_bin=[63,255], learning_rate=[0.03])
cfgs = [{k: rng.choice(v).item() for k, v in space.items()} for _ in range(14)]
out = []
for i, c in enumerate(cfgs):
    t = time.time()
    m = lgb.LGBMRegressor(objective="l2", n_estimators=1500, subsample=0.7, subsample_freq=1, n_jobs=1, verbose=-1, **c)
    m.fit(tr["X"], ytr, eval_set=[(va["X"], yva)], eval_metric="l1", callbacks=[lgb.early_stopping(80, verbose=False)])
    p = m.predict(va["X"]); mae = float(np.mean(np.abs(yva - p)))
    out.append({**c, "best_iter": int(m.best_iteration_), "val_mae": mae, "neso_val_mae": float(np.mean(np.abs(yva)))})
    print(i, c, "iters", m.best_iteration_, "val MAE", round(mae, 3), f"{time.time()-t:.0f}s", flush=True)
json.dump(out, open("/home/claude/results/tuning.json", "w"), indent=1)
best = min(out, key=lambda r: r["val_mae"]); print("BEST", best)
