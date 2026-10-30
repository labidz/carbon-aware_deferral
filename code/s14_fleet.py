"""Fleet replay: real Acme jobs (0.5-12 h, >=1 GPU), shared GPU capacity, deferral within a slack window driven by re-planned forecasts.
Replay window: 7 weeks from Mon 2023-11-06 (dense GB vintages, walk-forward out-of-sample paths). Trace weeks 8..14 are replayed on the same slot-of-week."""
import numpy as np, pandas as pd, json, time, warnings; warnings.filterwarnings("ignore")
import sim, ukciv as U
t00 = time.time(); H = 96; T = U.T
# ---------------- jobs ----------------
rows = []
for name in ["seren", "kalos"]:
    d = pd.read_csv(f"/home/claude/repos/AcmeTrace/data/job_trace/trace_{name}.csv", usecols=["gpu_num", "duration", "submit_time"]); rows.append(d)
J = pd.concat(rows, ignore_index=True); J = J[(J.gpu_num > 0) & (J.duration >= 1800) & (J.duration <= 12 * 3600)].copy()
J["t"] = pd.to_datetime(J.submit_time, utc=True, format="mixed").dt.tz_localize(None)
mon0 = pd.Timestamp("2023-02-27"); off = (J.t - mon0).dt.total_seconds().values; wk = (off // (7 * 86400)).astype(int); sow = ((off % (7 * 86400)) // 1800).astype(int)
WEEKS = 7; TRACE_W0 = 8
slot0 = int((pd.Timestamp("2023-11-06") - pd.Timestamp(U.t0, unit="s")).total_seconds() // 1800)
jobs = []
for w in range(WEEKS):
    m = wk == TRACE_W0 + w; jobs.append(pd.DataFrame({"a": slot0 + w * 336 + sow[m], "g": J.gpu_num.values[m], "d": np.ceil(J.duration.values[m] / 1800).astype(int), "week": w}))
JB = pd.concat(jobs, ignore_index=True).sort_values("a", kind="stable").reset_index(drop=True)
print("jobs", len(JB), "GPU-h", float((JB.g * JB.d / 2).sum()), f"{time.time()-t00:.0f}s", flush=True)
# ---------------- forecast paths ----------------
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); Wm = np.load("/home/claude/data/wf_main.npz")
s_, L_, f0, blk = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], Wm["blk"]
def P(v): return sim.PathSet(sim.to_matrix(s_, L_, np.where(blk >= 0, v, np.nan).astype(np.float32)))
PS = {"neso": P(f0), "ml": P(f0 + np.nan_to_num(Wm["q50"])), "oracle": sim.PathSet(sim.actual_matrix())}
last_cap = lambda t: int(np.searchsorted(sim.CAPS, t, side="right") - 1)
A = U.A.astype(np.float64)
CA = np.concatenate([[0], np.cumsum(np.nan_to_num(A))]); CN = np.concatenate([[0], np.cumsum(np.isnan(A))])
def run(policy, S, C, jobs_df):
    n = len(jobs_df); a = jobs_df.a.values; g = jobs_df.g.values; d = jobs_df.d.values; dl = a + S
    start = np.full(n, -1); usage = np.zeros(T + 200); waiting = []; nxt = 0; t = int(a.min()); tmax = int(dl.max() + d.max() + 400)
    ps = PS.get(policy)
    while t < min(tmax, T - 100):
        while nxt < n and a[nxt] <= t: waiting.append(nxt); nxt += 1
        if waiting:
            W = np.array(waiting)
            if policy == "asap": order = W
            else:
                ci = last_cap(t); base = t - sim.CAPS[ci] if ci >= 0 else 99
                r = np.maximum(dl[W] - t, 0); dd = d[W]
                if ci < 0 or base > 40: due = np.ones(len(W), bool)
                else:
                    Rm = int(r.max()); o = np.arange(Rm + 1)[None, :]; lo = base + o; hi = lo + dd[:, None]; ok = (hi <= H) & (o <= r[:, None])
                    loc, hic = np.minimum(lo, H), np.minimum(hi, H)
                    cst = (ps.cum[ci][hic] - ps.cum[ci][loc]) / dd[:, None]; bad = (ps.nan[ci][hic] - ps.nan[ci][loc]) > 0
                    cst = np.where(ok & ~bad, cst, np.inf); best = np.argmin(cst, 1); due = (best == 0) | (r == 0) | ~np.isfinite(cst).any(1)
                W = W[due]; order = W[np.lexsort((a[W], dl[W]))]
            started = []
            for j in order:
                if usage[t:t + d[j]].max() + g[j] <= C: usage[t:t + d[j]] += g[j]; start[j] = t; started.append(j)
            if started: waiting = [j for j in waiting if start[j] < 0]
        t += 1
        if nxt >= n and not waiting: break
    ok = start >= 0; hi = start + d; E = np.where(ok, (CA[np.minimum(hi, T)] - CA[np.minimum(start, T)]), np.nan)
    bad = (CN[np.minimum(hi, T)] - CN[np.minimum(start, T)]) > 0; E = np.where(bad, np.nan, E) * g
    return dict(start=start, E=E, viol=float(np.mean(start[ok] > dl[ok])), wait_h=float(np.mean((start[ok] - a[ok]) / 2)), unstarted=int((~ok).sum()), peak=float(usage.max()))
# ---------------- jobs whose whole possible window has known actuals ----------------
S_LIST = [12, 24, 48]; KAPPA = [None, 1.0, 0.6, 0.4, 0.25]; out = {"n_jobs": int(len(JB)), "slot0": slot0}
a_ = JB.a.values; ok_ = np.ones(len(JB), bool)
for S in S_LIST: ok_ &= np.isfinite(CA[np.minimum(a_ + S + JB.d.values, T)] - CA[a_]) & ((CN[np.minimum(a_ + S + JB.d.values, T)] - CN[a_]) == 0)
JB = JB[ok_].reset_index(drop=True); out["n_jobs_valid"] = int(len(JB))
# unconstrained usage under immediate start -> capacity reference
u0 = np.zeros(T + 200)
for a, g, d in zip(JB.a.values, JB.g.values, JB.d.values): u0[a:a + d] += g
win = slice(slot0, slot0 + WEEKS * 336); out["usage_immediate"] = dict(mean=float(u0[win].mean()), p99=float(np.percentile(u0[win], 99)), max=float(u0[win].max()))
print(out["usage_immediate"], f"{time.time()-t00:.0f}s", flush=True)
res = {}
for S in S_LIST:
    for kp in KAPPA:
        C = 1e12 if kp is None else kp * out["usage_immediate"]["max"]
        base = run("asap", S, C, JB); E0 = base["E"]; wkv = JB.week.values
        for pol in ["asap", "neso", "ml", "oracle"]:
            r = base if pol == "asap" else run(pol, S, C, JB)
            fin = np.isfinite(r["E"]) & np.isfinite(E0)
            per_week = [(float(np.nansum(np.where(fin & (wkv == w), r["E"], 0))), float(np.nansum(np.where(fin & (wkv == w), E0, 0)))) for w in range(WEEKS)]
            res[f"{pol}|S{S}|k{kp}"] = dict(saving_pct=float(100 * (1 - np.nansum(np.where(fin, r["E"], 0)) / np.nansum(np.where(fin, E0, 0)))), viol=r["viol"], wait_h=r["wait_h"], unstarted=r["unstarted"], peak_ratio=r["peak"] / out["usage_immediate"]["max"], per_week=per_week)
        print(f"S={S} kappa={kp}:", {p: (round(res[f'{p}|S{S}|k{kp}']['saving_pct'], 2), round(100 * res[f'{p}|S{S}|k{kp}']['viol'], 1)) for p in ["neso", "ml", "oracle"]}, f"{time.time()-t00:.0f}s", flush=True)
out["res"] = res; json.dump(out, open("/home/claude/results/fleet.json", "w"), indent=1); print("DONE")
