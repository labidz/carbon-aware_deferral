import numpy as np, pandas as pd, json, ukciv as U, warnings; warnings.filterwarnings("ignore")
from gate import gate_lambdas, lead_bucket
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); X = np.load("/home/claude/data/g_X.npy", mmap_mode="r")
s, L, f0, y, names = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], M["y"], list(M["names"])
Wm = np.load("/home/claude/data/wf_main.npz"); blk = Wm["blk"]; fi = np.isfinite(y); sel = (blk >= 0) & fi
col = lambda n: np.asarray(X[:, names.index(n)])
res = y - f0; day = (s + L) // 48
q50, mean = np.nan_to_num(Wm["q50"]), np.nan_to_num(Wm["mean"])
lam = gate_lambdas(s, L, y, f0, q50, blk, K=336)
corr = {"NESO": 0 * res, "NESO+RollBias7d": np.nan_to_num(col("e_wk")), "NESO+RollBias7d(tod)": np.nan_to_num(col("e_tod_wk")),
        "ML-mean": mean, "ML-median": q50, "ML-gated": lam * q50}
VAR = ["no_vint", "no_adapt", "base_only", "base_adapt", "relative", "no_real", "no_mix", "main_w60", "base_vint", "base_real", "base_vint_path"]
for v in VAR:
    try: corr["abl:" + v] = np.nan_to_num(np.load(f"/home/claude/data/wf_{v}.npz")["mean"])
    except Exception: pass
names_blk = ["2023-06","2023-07","2023-08","2023-09","2023-10","2023-11","2023-12","2024-01/02","2026-07","2026-08","2026-09","2026-10"]
ev = sel & (blk >= 1)
ud, inv = np.unique(day[ev], return_inverse=True)
def day_agg(err_abs): return np.bincount(inv, weights=err_abs[ev], minlength=len(ud)), np.bincount(inv, minlength=len(ud)).astype(float)
AGG = {k: day_agg(np.abs(res - v)) for k, v in corr.items()}
rng = np.random.default_rng(5); B = 2000; draws = rng.integers(0, len(ud), (B, len(ud)))
def mae_ci(k): S_, C_ = AGG[k]; v = S_[draws].sum(1) / C_[draws].sum(1); return float(S_.sum() / C_.sum()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
def diff_ci(k, base="NESO"):
    Sk, Ck = AGG[k]; Sb, Cb = AGG[base]; d = (Sk[draws].sum(1) / Ck[draws].sum(1)) / (Sb[draws].sum(1) / Cb[draws].sum(1)) - 1
    return float(100 * ((Sk.sum() / Ck.sum()) / (Sb.sum() / Cb.sum()) - 1)), float(100 * np.percentile(d, 2.5)), float(100 * np.percentile(d, 97.5))
out = {"overall": {}, "by_block": {}, "by_period": {}, "by_lead": {}}
for k in corr:
    m, lo_, hi_ = mae_ci(k); dm, dlo, dhi = diff_ci(k)
    out["overall"][k] = dict(mae=m, mae_lo=lo_, mae_hi=hi_, rel_change_vs_NESO_pct=dm, rel_lo=dlo, rel_hi=dhi)
    out["by_block"][k] = {names_blk[b]: float(np.mean(np.abs(res[sel & (blk == b)] - corr[k][sel & (blk == b)]))) for b in range(12)}
    for p, mk in [("2023-24", blk <= 7), ("2026", blk >= 8)]:
        e_ = np.abs(res - corr[k])[ev & mk]; n_ = np.abs(res)[ev & mk]; out["by_period"].setdefault(k, {})[p] = dict(mae=float(e_.mean()), neso=float(n_.mean()), rel_pct=float(100 * (e_.mean() / n_.mean() - 1)))
    bk = lead_bucket(L)
    out["by_lead"][k] = {["0-2h","2-6h","6-12h","12-24h","24-48h"][i]: float(np.mean(np.abs(res[ev & (bk == i)] - corr[k][ev & (bk == i)]))) for i in range(5)}
# win counts
for k in ["ML-median", "ML-mean", "ML-gated", "NESO+RollBias7d"]:
    out["overall"][k]["blocks_better_than_NESO"] = int(sum(out["by_block"][k][n] < out["by_block"]["NESO"][n] for n in names_blk[1:]))
# paired contrasts among ablations (mean models vs ML-mean)
cont = {}
for v in ["abl:no_vint", "abl:no_adapt", "abl:base_only", "abl:base_adapt", "abl:relative", "abl:no_real", "abl:no_mix", "abl:base_vint", "abl:base_real", "abl:base_vint_path", "abl:main_w60"]:
    if v in AGG:
        Sk, Ck = AGG[v]; Sm, Cm = AGG["ML-mean"]; d = (Sk[draws].sum(1) / Ck[draws].sum(1)) / (Sm[draws].sum(1) / Cm[draws].sum(1)) - 1
        cont[v] = dict(rel_vs_full_pct=float(100 * ((Sk.sum() / Ck.sum()) / (Sm.sum() / Cm.sum()) - 1)), lo=float(100 * np.percentile(d, 2.5)), hi=float(100 * np.percentile(d, 97.5)))
out["ablation_vs_full_mean"] = cont
json.dump(out, open("/home/claude/results/forecast_eval.json", "w"), indent=1)
print(pd.DataFrame({k: v for k, v in out["overall"].items()}).T.round(2).to_string())
print("\nblocks beating NESO (of 11):", {k: out["overall"][k].get("blocks_better_than_NESO") for k in ["ML-median", "ML-mean", "ML-gated", "NESO+RollBias7d"]})
print("\nablation vs full mean:", json.dumps({k: {a: round(b, 2) for a, b in v.items()} for k, v in cont.items()}))
print(pd.DataFrame(out["by_lead"]).T.round(2).to_string())
