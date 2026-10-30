"""External validation on 13 grids (CarbonCast data): revision test, accuracy, and synthetic-vs-real decision gap (+ calibrated Berkson-type generator)."""
import numpy as np, pandas as pd, json, time, warnings, statsmodels.api as sm; warnings.filterwarnings("ignore")
from ext_data import load_region, REGIONS
import ext_sim as X
wl = json.load(open("/home/claude/results/workload.json"))["all_gpu_jobs"]["bins"]; sh = [b["gpu_hour_share"] for b in wl if b["slots"]]
WD = {1: sh[0] + sh[1], 2: sh[2], 4: sh[3], 8: sh[4], 12: sh[5]}                    # hourly job lengths <- Acme GPU-hour classes
DS = [1, 2, 4, 8, 12]; SS = [3, 6, 12, 24, 48]; NS = 6; TSPLIT = 92 * 24
BUCK = [(i, i + 12) for i in range(0, 96, 12)]
GENS = ["fixed_uniform", "iid", "ar1", "bootstrap", "berkson"]
def wfrac(env, arr_fn, pathset, pol, S, ref):
    """GPU-hour-weighted oracle capture and saving for one pathset."""
    num = 0; den = 0
    for d in DS:
        a = ref[(d, S)]["arr"]; asap = ref[(d, S)]["asap"]; orc = ref[(d, S)]["orc"]
        off = X.run_policies(env, a, d, S, pathset)[pol]; c = env.realised(a + off, d)
        num += WD[d] * (1 - c.sum() / asap.sum()); den += WD[d] * (1 - orc.sum() / asap.sum())
    return num / den
out = {"regions": {}}; t00 = time.time(); rng_master = np.random.default_rng(2026)
rev_rows = []
import os; os.makedirs("/home/claude/results/ext", exist_ok=True)
for r in REGIONS:
    pj = f"/home/claude/results/ext/{r}.json"; pc = f"/home/claude/results/ext/{r}_rev.csv"
    if os.path.exists(pj) and os.path.exists(pc):
        out["regions"][r] = json.load(open(pj)); rev_rows.append(pd.read_csv(pc)); print(r, "cached", flush=True); continue
    R = load_region(r); env = X.Env(R["A"], R["bstart"]); F = R["F"]; nb = len(R["bstart"]); H = 96
    O = X.oracle_matrix(env); orc_ps = X.PathSet(env, O); real_ps = X.PathSet(env, F)
    E = np.where(np.isfinite(O) & np.isfinite(F), O - F, np.nan)                       # real error paths (A - F)
    mae_l = np.nanmean(np.abs(E), 0)
    train_blocks = R["bstart"] + 96 <= TSPLIT
    coefs, fitted, Xwin = X.fit_berkson(env, F, train_blocks, K=6, buckets=BUCK)
    pool_res = (F - fitted)[train_blocks]; pool_res = pool_res[np.isfinite(pool_res).all(1)]
    pool_err = E[train_blocks]; pool_err = pool_err[np.isfinite(pool_err).all(1)]
    # test arrivals: hours >= TSPLIT
    arrs = np.arange(TSPLIT, env.T)
    ref = {}
    for d in DS:
        for S in SS:
            ok = X.valid_arrivals(env, arrs, d, S, [real_ps, orc_ps]); a = arrs[ok]
            ref[(d, S)] = dict(arr=a, asap=env.realised(a, d), orc=env.realised(a + X.run_policies(env, a, d, S, orc_ps)["static"], d))
    rr = {"mean_ci": float(np.nanmean(R["A"])), "sd_ci": float(np.nanstd(R["A"])), "mae_day1": float(np.nanmean(mae_l[:24])), "mae_day2": float(np.nanmean(mae_l[24:48])), "mae_day4": float(np.nanmean(mae_l[72:])),
          "bias_day2": float(np.nanmean(E[:, 24:48])), "n_blocks": nb, "n_test_arrivals": int(len(ref[(4, 12)]["arr"])), "oracle_saving_pct_S12": float(100 * np.mean([1 - ref[(d, 12)]["orc"].sum() / ref[(d, 12)]["asap"].sum() for d in DS]))}
    rr["real"] = {pol: {S: float(wfrac(env, None, real_ps, pol, S, ref)) for S in SS} for pol in ["static", "replan"]}
    rr["syn"] = {}
    for g in GENS:
        res = {pol: {S: [] for S in SS} for pol in ["static", "replan"]}
        for sd in range(NS):
            rng = np.random.default_rng(1000 * sd + hash(r) % 997)
            if g == "berkson": M = X.gen_berkson(env, fitted, pool_res, rng)
            else: M = X.gen_additive(env, g, mae_l, rng, resid_pool=pool_err)
            ps = X.PathSet(env, M)
            if not all(np.isfinite(ps.costs(*(lambda ci, b, v: (ci, b))(*X.Env.epoch_view(env, np.array([TSPLIT + 24]), 0)), 24, 4)).all() for _ in [0]): pass
            for pol in ["static", "replan"]:
                for S in SS: res[pol][S].append(float(wfrac(env, None, ps, pol, S, ref)))
        rr["syn"][g] = {pol: {S: v for S, v in res[pol].items()} for pol in res}
    # revision over-reaction test: e = A - F_k(l) on r = F_k(l) - F_{k-1}(l+24), l in 0..71 (consecutive daily issues only)
    bs = R["bstart"]; rows = []
    for k in range(1, nb):
        if bs[k] - bs[k - 1] != 24: continue
        l = np.arange(72); w = bs[k] + l; Fk = F[k, l]; Fp = F[k - 1, l + 24]; A = env.A[np.clip(w, 0, env.T - 1)]
        okk = np.isfinite(Fk) & np.isfinite(Fp) & np.isfinite(A) & (w < env.T)
        for ll, f_, p_, a_, w_ in zip(l[okk], Fk[okk], Fp[okk], A[okk], w[okk]): rows.append((r, ll, f_ - p_, a_ - f_, w_ // 24))
    D_ = pd.DataFrame(rows, columns=["region", "lead", "r", "e", "day"]); rev_rows.append(D_)
    m = sm.OLS(D_.e, sm.add_constant(D_.r)).fit(cov_type="cluster", cov_kwds={"groups": D_.day})
    rr["revision_slope"] = dict(slope=float(m.params["r"]), se=float(m.bse["r"]), t=float(m.tvalues["r"]), n=int(len(D_)))
    for nm, (a_, b_) in {"day1": (0, 24), "day2": (24, 48), "day3": (48, 72)}.items():
        dd = D_[(D_.lead >= a_) & (D_.lead < b_)]; mm = sm.OLS(dd.e, sm.add_constant(dd.r)).fit(cov_type="cluster", cov_kwds={"groups": dd.day}); rr[f"revision_slope_{nm}"] = dict(slope=float(mm.params["r"]), se=float(mm.bse["r"]), t=float(mm.tvalues["r"]))
    out["regions"][r] = rr; json.dump(rr, open(pj, "w")); D_.to_csv(pc, index=False)
    print(f"{r:8s} oracle {rr['oracle_saving_pct_S12']:.1f}%  real replan S12 {rr['real']['replan'][12]:.3f} | syn replan S12:", {g: round(float(np.mean(rr['syn'][g]['replan'][12])), 3) for g in GENS}, "| rev slope %.2f (t %.1f)" % (rr["revision_slope"]["slope"], rr["revision_slope"]["t"]), f"{time.time()-t00:.0f}s", flush=True)
ALL = pd.concat(rev_rows); ALL["dm"] = ALL.groupby("region").e.transform(lambda x: x - x.mean()); ALL["rm"] = ALL.groupby("region").r.transform(lambda x: x - x.mean())
mp = sm.OLS(ALL.dm, ALL.rm).fit(cov_type="cluster", cov_kwds={"groups": ALL.region.astype(str) + ALL.day.astype(str)})
out["pooled_revision"] = dict(slope=float(mp.params["rm"]), se=float(mp.bse["rm"]), t=float(mp.tvalues["rm"]), n=int(len(ALL)), regions_negative=int(sum(out["regions"][r]["revision_slope"]["slope"] < 0 for r in REGIONS)),
                              regions_signif_negative=int(sum(out["regions"][r]["revision_slope"]["t"] < -1.96 for r in REGIONS)))
json.dump(out, open("/home/claude/results/external.json", "w"), indent=1); print("DONE", out["pooled_revision"], f"{time.time()-t00:.0f}s")
