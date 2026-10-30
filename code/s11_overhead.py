"""Computational overhead audit of the forecasting pipeline: measured CPU time, assumed per-core power -> energy/carbon (an ESTIMATE)."""
import numpy as np, time, json, lightgbm as lgb, ukciv as U, warnings, resource; warnings.filterwarnings("ignore")
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); X = np.load("/home/claude/data/g_X.npy", mmap_mode="r")
s, L, f0, y = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], M["y"]; fi = np.isfinite(y) & (s % 2 == 0); tis = U.TS[s]
out = {}
# 1. feature construction for one capture (96 leads)
caps = U.split_slots("testA")[:60]; t = time.process_time()
for c in caps: U.make_samples(np.array([c]))
out["feature_cpu_ms_per_capture"] = 1000 * (time.process_time() - t) / len(caps)
# 2. training of one median model on all data to Feb-2024 (hourly captures, 120 trees) and inference per capture
import pandas as pd
tr = np.where(fi & (tis < pd.Timestamp("2024-02-06")))[0]; Xtr = np.asarray(X[tr]); P = dict(num_leaves=31, min_child_samples=1000, colsample_bytree=0.8, reg_lambda=30, max_bin=63, learning_rate=0.03, subsample=0.7, subsample_freq=1, n_jobs=1, verbose=-1)
t = time.process_time(); m = lgb.LGBMRegressor(n_estimators=120, objective="quantile", alpha=0.5, **P).fit(Xtr, (y - f0)[tr]); out["train_cpu_s_one_model_%d_rows" % len(tr)] = time.process_time() - t
xc = np.asarray(X[np.where(s == U.split_slots("testA")[100])[0]][:96]); reps = 300; t = time.process_time()
for _ in range(reps): m.predict(xc)
out["gbm_inference_cpu_ms_per_capture_per_model"] = 1000 * (time.process_time() - t) / reps
out["model_trees"] = 120; out["gbm_model_bytes"] = len(m.booster_.model_to_string())
# 3. DL numbers recorded by s7_dl.py
for mode in ["A", "B"]:
    try: j = json.load(open(f"/home/claude/results/dl_{mode}.json")); out[f"dl_{mode}"] = dict(params=j["params"], inference_ms_per_capture=1000 * 96 * j["inference_seconds_per_sample"], train_seconds=j["train_seconds"])
    except Exception: pass
# 4. walk-forward totals from logs
wf = json.load(open("/home/claude/results/wf_main.json")); out["walk_forward_main_blocks"] = len(wf)
# 5. energy / carbon ESTIMATE under stated assumptions
refresh_per_year = 365 * 48
assump = {"cpu_watts_per_core_low_mid_high": [10, 20, 40], "grid_g_per_kwh": 150, "refresh_per_year": refresh_per_year}
per_refresh_cpu_s = (out["feature_cpu_ms_per_capture"] + 4 * out["gbm_inference_cpu_ms_per_capture_per_model"]) / 1000
train_cpu_s_year = 12 * 4 * out["train_cpu_s_one_model_%d_rows" % len(tr)]               # monthly retrain of 4 models
res = {}
for w in assump["cpu_watts_per_core_low_mid_high"]:
    e_wh_year = (per_refresh_cpu_s * refresh_per_year + train_cpu_s_year) * w / 3600
    res[f"{w}W_per_core"] = dict(energy_Wh_per_year=e_wh_year, gCO2_per_year=e_wh_year / 1000 * assump["grid_g_per_kwh"], energy_Wh_per_refresh=per_refresh_cpu_s * w / 3600)
out["assumptions"] = assump; out["per_refresh_cpu_s"] = per_refresh_cpu_s; out["train_cpu_s_year"] = train_cpu_s_year; out["energy_estimate"] = res
# 6. break-even: carbon saved by ML correction over operational forecast for one 8-GPU, 4-h job (0.4 kW/GPU incl. host share; PUE 1.2)
job_kwh = 8 * 0.4 * 4 * 1.2; asap_ci = 148.0; gain_pp = 0.74 / 100                      # weighted absolute saving gain of ML re-plan vs NESO re-plan (12 h slack); intensity-weighted
out["breakeven"] = dict(job_kwh=job_kwh, extra_saving_g_per_job=job_kwh * asap_ci * gain_pp * (1 / 0.82 if False else 1.0))
out["breakeven"]["jobs_per_year_to_repay_overhead_at_20W"] = res["20W_per_core"]["gCO2_per_year"] / max(out["breakeven"]["extra_saving_g_per_job"], 1e-9)
out["rss_mb_peak"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
json.dump(out, open("/home/claude/results/overhead.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:2500])
