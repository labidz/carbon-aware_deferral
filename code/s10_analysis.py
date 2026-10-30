"""Aggregate decision results: reference scenarios, GPU-hour-weighted outcomes, paired day-block bootstrap, synthetic-vs-real gap, skill-vs-value."""
import numpy as np, pandas as pd, json, sim, ukciv as U, warnings; warnings.filterwarnings("ignore")
rng = np.random.default_rng(2024)
st = np.load("/home/claude/results/decisions_store.npz"); S_ = pd.read_parquet("/home/claude/results/decisions_summary.parquet")
wl = json.load(open("/home/claude/results/workload.json"))["all_gpu_jobs"]["bins"]
DS = [1, 2, 4, 8, 16, 24]; SS = [2, 6, 12, 24, 48]
wgt = {b["slots"]: b["gpu_hour_share"] for b in wl if b["slots"]}; addressable = sum(wgt.values())
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); Wm = np.load("/home/claude/data/wf_main.npz")
s, L, blk = M["s"].astype(np.int64), M["L"].astype(np.int64), Wm["blk"]
cap_blk = np.full(sim.NC, -1); ci = np.searchsorted(sim.CAPS, s); cap_blk[ci[blk >= 0]] = blk[blk >= 0]
day_all = sim.CAPS // 48; uday, day_id = np.unique(day_all, return_inverse=True); ND = len(uday)
B = 1000; CNT = rng.multinomial(ND, np.ones(ND) / ND, size=B).astype(float)          # day-resampling weights
def dayvec(costs, a, mask=None):
    m = np.ones(len(a), bool) if mask is None else mask
    return np.bincount(day_id[a[m]], weights=np.nan_to_num(costs[m].astype(float)), minlength=ND), np.bincount(day_id[a[m]], minlength=ND).astype(float)
def keyc(fc, pol, d, S): return f"{fc}|{pol}|d{d}|S{S}"
def vecs(fc, pol, d, S, mask_fn=None):
    a = st[f"a|d{d}|S{S}"]; m = None if mask_fn is None else mask_fn(a)
    return dayvec(st[keyc(fc, pol, d, S)], a, m), dayvec(st[f"asap|d{d}|S{S}"], a, m)[0], dayvec(st[f"oracle|d{d}|S{S}"], a, m)[0]
def wfrac_boot(fc, pol, S, mask_fn=None, dlist=DS):
    """GPU-hour-weighted saving and oracle-fraction with day-block bootstrap (resampling common days)."""
    num_c = np.zeros(B + 1); num_o = np.zeros(B + 1); num_s = np.zeros(B + 1); tot = 0
    for d in dlist:
        (c, n), asap, orc = vecs(fc, pol, d, S, mask_fn); w = wgt[d]
        for arr_, tgt in [(c, "c")]: pass
        cw = np.concatenate([[1.0], np.zeros(0)])
        full = np.vstack([np.ones(ND), CNT])                                          # row 0 = point estimate
        C_ = full @ c; A_ = full @ asap; O_ = full @ orc
        num_s += w * (1 - C_ / A_); num_o += w * (1 - O_ / A_); tot += w
    sv = num_s / tot; ov = num_o / tot
    return sv, sv / ov, ov
def ci(v): return float(v[0]), float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))
out = {"addressable_share": addressable, "weights": wgt}
# ---------- 1. weighted results for all fc/policy/S ----------
FCS = sorted(set(S_.fc)); POLS = ["static", "replan", "replan_h2", "replan_h5"]
W = {}
for fc in FCS:
    for pol in POLS:
        if keyc(fc, pol, 8, 24) not in st: continue
        for S in SS:
            sv, fr, ov = wfrac_boot(fc, pol, S); W[f"{fc}|{pol}|S{S}"] = dict(saving=ci(100 * sv), oracle_frac=ci(fr), fleet_saving_pct=100 * addressable * sv[0])
out["weighted"] = W
# ---------- 2. paired contrasts vs NESO-replan (oracle-fraction points, weighted) ----------
def contrast(fa, pa, fb, pb, S, mask_fn=None):
    sa, fa_, _ = wfrac_boot(fa, pa, S, mask_fn); sb, fb_, _ = wfrac_boot(fb, pb, S, mask_fn); d1 = 100 * (fa_ - fb_); d2 = 100 * (sa - sb)
    return dict(d_oracle_frac_pp=ci(d1), d_saving_pp=ci(d2))
C_ = {}
for fc, pol in [("ML-median", "replan"), ("ML-median", "static"), ("ML-gated", "replan"), ("NESO+RollBias7d", "replan"), ("ML-mean", "replan"), ("ML-median", "replan_h2"), ("NESO", "replan_h2"), ("ML-median", "replan_h5"),
                ("ML-median+risk0.5", "replan"), ("ML-median+risk1.0", "replan"), ("NESO", "static")]:
    for S in SS:
        if keyc(fc, pol, 8, S) in st: C_[f"{fc}|{pol}|vs NESO-replan|S{S}"] = contrast(fc, pol, "NESO", "replan", S)
for S in SS: C_[f"NESO|replan|vs NESO|static|S{S}"] = contrast("NESO", "replan", "NESO", "static", S)
out["contrasts"] = C_
# by period for the headline contrast
per_fn = {"2023-24": lambda a: U.TS[sim.CAPS[a]] < pd.Timestamp("2025-01-01"), "2026": lambda a: U.TS[sim.CAPS[a]] >= pd.Timestamp("2025-01-01")}
out["contrast_by_period"] = {p: {f"S{S}": contrast("ML-median", "replan", "NESO", "replan", S, fn) for S in SS} for p, fn in per_fn.items()}
out["weighted_by_period"] = {p: {f"{fc}|{pol}|S{S}": dict(zip(["saving", "oracle_frac"], [ci(100 * wfrac_boot(fc, pol, S, fn)[0]), ci(wfrac_boot(fc, pol, S, fn)[1])])) for fc, pol in [("NESO", "static"), ("NESO", "replan"), ("ML-median", "replan"), ("Oracle", "static")] for S in [24]} for p, fn in per_fn.items()}
# ---------- 3. synthetic vs real (per d,S; oracle fraction, unweighted summary + weighted) ----------
syn = {}
for fc in ["NESO", "syn:fixed-uniform", "syn:iid-gauss", "syn:ar1-gauss", "syn:real-error-bootstrap"]:
    for pol in ["static", "replan"]:
        for S in SS:
            sv, fr, ov = wfrac_boot(fc, pol, S); syn[f"{fc}|{pol}|S{S}"] = ci(fr)
out["synthetic_gap"] = syn
# ---------- 4. reference scenario table (d=8, S=24) with per-job risk metrics, by period ----------
ref = S_[(S_.d == 8) & (S_.S == 24)]
out["reference_d8_S24"] = ref.round(3).to_dict("records")
S_.to_csv("/home/claude/results/decisions_summary.csv", index=False)
json.dump(out, open("/home/claude/results/decision_tables.json", "w"), indent=1, default=float)
print("addressable share", round(addressable, 3))
for S in SS:
    print(f"S={S}: NESO-static {W[f'NESO|static|S{S}']['oracle_frac'][0]:.3f}  NESO-replan {W[f'NESO|replan|S{S}']['oracle_frac'][0]:.3f}  ML-med-replan {W[f'ML-median|replan|S{S}']['oracle_frac'][0]:.3f}",
          "| contrast ML-replan vs NESO-replan (pp of oracle):", [round(x, 2) for x in C_[f"ML-median|replan|vs NESO-replan|S{S}"]["d_oracle_frac_pp"]])
