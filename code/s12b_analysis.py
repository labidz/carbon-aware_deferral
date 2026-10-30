"""Statistics for the revision experiments: baselines, causal tuning, stand-alone model, arrival sensitivity, synthetic fidelity (GB + 13 external grids)."""
import numpy as np, pandas as pd, json, sim, ukciv as U, warnings, os; warnings.filterwarnings("ignore")
rng = np.random.default_rng(77); R = "/home/claude/results/"
st = dict(np.load(R + "decisions_store.npz")); ex = dict(np.load(R + "extras_store.npz")) if os.path.exists(R + "extras_store.npz") else {}; ex2 = dict(np.load(R + "extras2_store.npz")) if os.path.exists(R + "extras2_store.npz") else {}
ALL = {**st, **ex, **ex2}
wl = json.load(open(R + "workload.json")); bins = wl["all_gpu_jobs"]["bins"]; WG = {b["slots"]: b["gpu_hour_share"] for b in bins if b["slots"]}; addr = sum(WG.values())
DS = [1, 2, 4, 8, 16, 24]; SS = [2, 6, 12, 24, 48]
day_all = sim.CAPS // 48; uday, day_id = np.unique(day_all, return_inverse=True); ND = len(uday)
B = 1000; CNT = np.vstack([np.ones(ND), rng.multinomial(ND, np.ones(ND) / ND, size=B)]).astype(float)
hour_share = np.array([wl["submit_hour_share"].get(str(h), 0.0) for h in range(24)]); hour_w = 24 * hour_share / hour_share.sum()
def dv(c, a, w=None): return np.bincount(day_id[a], weights=(c if w is None else c * w), minlength=ND)
def wcap(keys, S, arrival_weights=False):
    """GPU-hour-weighted saving & oracle capture for several items on a COMMON arrival set (paired); returns {key: (saving[B+1], capture[B+1])}."""
    num = {k: np.zeros(B + 1) for k in keys}; numo = np.zeros(B + 1); tot = 0
    for d in DS:
        a = st[f"a|d{d}|S{S}"]; ok = np.ones(len(a), bool)
        for k in keys:
            c = ALL[f"{k}|d{d}|S{S}"]; ok &= np.isfinite(c)
        aa = a[ok]; w = None
        if arrival_weights: w = hour_w[((sim.CAPS[aa] % 48) // 2).astype(int)]
        asap = dv(st[f"asap|d{d}|S{S}"][ok], aa, w); orc = dv(st[f"oracle|d{d}|S{S}"][ok], aa, w); A_ = CNT @ asap; O_ = CNT @ orc; wd = WG[d]
        for k in keys: num[k] += wd * (1 - (CNT @ dv(ALL[f"{k}|d{d}|S{S}"][ok], aa, w)) / A_)
        numo += wd * (1 - O_ / A_); tot += wd
    return {k: (num[k] / tot, num[k] / numo) for k in keys}
ci = lambda v: [float(v[0]), float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))]
out = {}
ITEMS = {"NESO-replan": "NESO|replan", "NESO-static": "NESO|static", "ML-replan": "ML-median|replan", "ML-static": "ML-median|static", "Clim-plan": "Clim|static", "Thr10": "Thr10|rule", "Thr25": "Thr25|rule", "Thr40": "Thr40|rule", "Thr55": "Thr55|rule", "Thr70": "Thr70|rule", "Thr85": "Thr85|rule"}
if "ML-causal|replan|d8|S24" in ALL: ITEMS.update({"MLcausal-replan": "ML-causal|replan", "Standalone-replan": "Standalone|replan"})
out["baselines"] = {}
for S in SS:
    keys = {n: k for n, k in ITEMS.items() if f"{k}|d8|S{S}" in ALL}
    # common arrival set across all items (NaNs from ML-causal block 0 etc. removed jointly)
    res = wcap(list(keys.values()), S); r = {}
    for n, k in keys.items(): r[n] = dict(saving=ci(100 * res[k][0]), capture=ci(100 * res[k][1]), vs_neso_replan_pp=ci(100 * (res[k][1] - res[ITEMS["NESO-replan"]][1])), vs_ml_replan_pp=ci(100 * (res[k][1] - res[ITEMS["ML-replan"]][1])))
    out["baselines"][f"S{S}"] = r
# ---- arrival-time sensitivity (diurnal arrival weights from the Acme submission-hour profile)
out["arrival_sensitivity"] = {}
for S in [6, 24, 48]:
    keys = [ITEMS["NESO-replan"], ITEMS["ML-replan"], ITEMS["Clim-plan"]]; a_ = wcap(keys, S, True); u_ = wcap(keys, S, False)
    out["arrival_sensitivity"][f"S{S}"] = {"diurnal": {k: ci(100 * a_[k][1]) for k in keys} | {"ML_minus_NESO_pp": ci(100 * (a_[keys[1]][1] - a_[keys[0]][1])), "oracle_saving_pct": None},
                                           "uniform": {k: ci(100 * u_[k][1]) for k in keys} | {"ML_minus_NESO_pp": ci(100 * (u_[keys[1]][1] - u_[keys[0]][1]))}}
    out["arrival_sensitivity"][f"S{S}"]["diurnal"]["saving_ML_replan_pct"] = ci(100 * a_[keys[1]][0])
# ---- GB synthetic fidelity (multi-seed)
if os.path.exists(R + "gb_synthetic.json"):
    rec = json.load(open(R + "gb_synthetic.json")); G = {}
    for key, v in rec.items():
        g, sd = key.split("|")
        for pol in v:
            for S, (syn_, real_) in v[pol].items(): G.setdefault(g, {}).setdefault(pol, {}).setdefault(S, []).append((syn_, real_))
    out["gb_synthetic"] = {g: {pol: {S: dict(syn_mean=float(np.mean([x[0] for x in L_])), syn_sd=float(np.std([x[0] for x in L_])), real=float(np.mean([x[1] for x in L_])), gap_pp=float(100 * np.mean([x[0] - x[1] for x in L_])), n_seeds=len(L_)) for S, L_ in G[g][pol].items()} for pol in G[g]} for g in G}
    out["gb_synthetic_mad_pp"] = {g: float(np.mean([abs(x["gap_pp"]) for pol in out["gb_synthetic"][g] for x in out["gb_synthetic"][g][pol].values()])) for g in out["gb_synthetic"]}
# ---- external 13-grid fidelity and revisions
if os.path.exists(R + "external.json"):
    ext = json.load(open(R + "external.json")); regs = list(ext["regions"]); GN = ["fixed_uniform", "iid", "ar1", "bootstrap", "berkson"]; E = {}
    for g in GN:
        E[g] = {}
        for pol in ["static", "replan"]:
            for S in ["3", "6", "12", "24", "48"]:
                diffs = np.array([100 * (np.mean(ext["regions"][r]["syn"][g][pol][S]) - ext["regions"][r]["real"][pol][S]) for r in regs]); bs = [np.mean(rng.choice(diffs, len(diffs))) for _ in range(2000)]
                E[g][f"{pol}|S{S}"] = dict(mean_gap_pp=float(diffs.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], mad_pp=float(np.abs(diffs).mean()), per_region=dict(zip(regs, diffs.tolist())))
    out["external_gap"] = E
    out["external_mad_pp"] = {g: float(np.mean([v["mad_pp"] for v in E[g].values()])) for g in GN}
    out["external_bias_pp"] = {g: float(np.mean([v["mean_gap_pp"] for v in E[g].values()])) for g in GN}
    out["external_revision"] = {"pooled": ext["pooled_revision"], "by_region": {r: ext["regions"][r]["revision_slope"] for r in regs}}
    out["external_regions"] = {r: {k: ext["regions"][r][k] for k in ["mean_ci", "sd_ci", "mae_day1", "mae_day2", "mae_day4", "oracle_saving_pct_S12"]} | {"real_replan_S12": ext["regions"][r]["real"]["replan"]["12"], "real_static_S12": ext["regions"][r]["real"]["static"]["12"]} for r in regs}

# ---- reference scenario (d=8 slots = 4 h, S=24 slots = 12 h) metrics for extra items
ref = {}
a = st["a|d8|S24"]; asap = st["asap|d8|S24"]; orc = st["oracle|d8|S24"]
for nm, k in [("Clim-plan", "Clim|static"), ("Thr40", "Thr40|rule"), ("Thr55", "Thr55|rule"), ("MLcausal-replan", "ML-causal|replan"), ("Standalone-replan", "Standalone|replan"), ("NESO-replan", "NESO|replan"), ("ML-replan", "ML-median|replan")]:
    c = ALL.get(f"{k}|d8|S24")
    if c is None: continue
    ok = np.isfinite(c); sv = 1 - c[ok] / asap[ok]
    ref[nm] = dict(saving_pct=float(100 * (1 - c[ok].sum() / asap[ok].sum())), capture_pct=float(100 * (1 - c[ok].sum() / asap[ok].sum()) / (1 - orc[ok].sum() / asap[ok].sum())), backfire_pct=float(100 * np.mean(c[ok] > asap[ok] + 1e-9)), p5_pct=float(100 * np.percentile(sv, 5)), n=int(ok.sum()))
out["reference_extras"] = ref
json.dump(out, open(R + "new_tables.json", "w"), indent=1, default=float)
for S in ["S6", "S24"]:
    for n, v in out["baselines"].get(S, {}).items(): print(S, f"{n:20s} capture {v['capture'][0]:5.1f} [{v['capture'][1]:5.1f},{v['capture'][2]:5.1f}]  vs NESO-replan {v['vs_neso_replan_pp'][0]:+5.2f} [{v['vs_neso_replan_pp'][1]:+5.2f},{v['vs_neso_replan_pp'][2]:+5.2f}]")
print("GB synthetic MAD pp:", out.get("gb_synthetic_mad_pp")); print("External MAD pp:", out.get("external_mad_pp"), "bias", out.get("external_bias_pp"))
