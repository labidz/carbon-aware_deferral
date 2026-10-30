"""Full decision-level experiment grid on real forecast vintages (walk-forward, out-of-sample paths)."""
import numpy as np, sim, json, time, pandas as pd, ukciv as U, warnings; warnings.filterwarnings("ignore")
from gate import gate_lambdas
from conformal import rolling_cqr
t00 = time.time(); rng = np.random.default_rng(11)
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); X = np.load("/home/claude/data/g_X.npy", mmap_mode="r")
s, L, f0, y, names = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], M["y"], list(M["names"])
Wm = np.load("/home/claude/data/wf_main.npz"); blk = Wm["blk"]; fi = np.isfinite(y)
col = lambda n: np.asarray(X[:, names.index(n)])
def P(v): return sim.PathSet(sim.to_matrix(s, L, np.where(blk >= 0, v, np.nan).astype(np.float32)))
q50, mean, q10, q90 = (np.nan_to_num(Wm[k]) for k in ["q50", "mean", "q10", "q90"])
lam = gate_lambdas(s, L, y, f0, q50, blk, K=336)
F = {"NESO": P(f0), "ML-median": P(f0 + q50), "ML-mean": P(f0 + mean), "ML-gated": P(f0 + lam * q50),
     "NESO+RollBias7d": P(f0 + np.nan_to_num(col("e_wk"))), "NESO+RollBias7d-tod": P(f0 + np.nan_to_num(col("e_tod_wk")))}
for v in ["no_vint", "no_adapt", "base_only", "base_adapt", "relative", "no_real", "no_mix", "main_w60"]:
    try: F["abl:" + v] = P(f0 + np.nan_to_num(np.load(f"/home/claude/data/wf_{v}.npz")["mean"]))
    except Exception as ex: print("missing variant", v)
ORC = sim.actual_matrix(); F["Oracle"] = sim.PathSet(ORC)
# risk-aware paths: median + k * half-width of rolling-CQR interval
lo, hi = np.minimum(q10, q90), np.maximum(q10, q90); sel = (blk >= 0) & fi
Qr = rolling_cqr(s, L, y - f0, lo, hi, sel); Qr = np.where(np.isfinite(Qr), Qr, 0.0)
half = (hi - lo) / 2 + Qr
for k in [0.5, 1.0]: F[f"ML-median+risk{k}"] = P(f0 + q50 + k * half)
# synthetic noise forecasters at matched per-lead MAE
e = y - f0; m = sel; mae_L = np.array([np.mean(np.abs(e[m & (L == l)])) for l in range(96)]); sig = (mae_L * 1.2533)[None, :]
nc = sim.NC; ORCn = ORC.copy()
idx = np.clip(sim.CAPS[:, None] + np.arange(96)[None, :], 0, U.T - 1)
xa = 2 * mae_L.mean() / np.nanmean(U.A); eps_w = rng.uniform(-xa, xa, U.T)
def syn(noise): return sim.PathSet(np.where(np.isfinite(ORCn), ORCn - noise, np.nan).astype(np.float32))
F["syn:fixed-uniform"] = syn(eps_w[idx] * ORCn)
F["syn:iid-gauss"] = syn(rng.normal(size=(nc, 96)) * sig)
rho = 0.98; z = np.zeros((nc, 96)); z[:, 0] = rng.normal(size=nc)
for l in range(1, 96): z[:, l] = rho * z[:, l-1] + np.sqrt(1 - rho**2) * rng.normal(size=nc)
F["syn:ar1-gauss"] = syn(z * sig)
EM = sim.to_matrix(s, L, np.where(m, e, np.nan)); EMn = np.where(np.isfinite(EM), EM, 0.0)
cap_blk = np.full(nc, -1); ci = np.searchsorted(sim.CAPS, s); cap_blk[ci[blk >= 0]] = blk[blk >= 0]
boot = np.zeros((nc, 96), np.float32)
for b in range(12):
    ids = np.where(cap_blk == b)[0]
    if len(ids): boot[ids] = EMn[rng.choice(ids, len(ids))]
F["syn:real-error-bootstrap"] = syn(boot)
print("forecasters:", list(F), f"{time.time()-t00:.0f}s", flush=True)

arr_all = np.where(cap_blk >= 1)[0]
DS = [1, 2, 4, 8, 16, 24]; SS = [2, 6, 12, 24, 48]
FULL = ["NESO", "ML-median", "ML-mean", "ML-gated", "NESO+RollBias7d", "Oracle"] + [k for k in F if k.startswith("ML-median+risk")]
POL = {"static": 0.0, "replan": 0.0, "replan_h2": 0.02, "replan_h5": 0.05}
summary = []; store = {}
for d in DS:
    for S in SS:
        t = time.time()
        ok = sim.valid_arrivals(arr_all, d, S, list(F.values())); a = arr_all[ok]
        asap = sim.cost_of(a, d, np.zeros(len(a), int)); store[f"a|d{d}|S{S}"] = a; store[f"asap|d{d}|S{S}"] = asap
        per = np.where(sim.CAPS[a] < U.t0*0 + 10**9, 0, 0)  # placeholder
        yr26 = U.TS[sim.CAPS[a]] >= pd.Timestamp("2025-01-01")
        orc_static = sim.cost_of(a, d, sim.run_policies(a, d, S, F["Oracle"])["static"]); store[f"oracle|d{d}|S{S}"] = orc_static
        osav = 1 - orc_static.mean() / asap.mean()
        for fc, ps in F.items():
            pols = POL if fc in FULL else {"static": 0.0, "replan": 0.0}
            for pn, delta in pols.items():
                if fc == "Oracle" and pn != "static": continue
                r = sim.run_policies(a, d, S, ps, delta=delta)
                off = r["static"] if pn == "static" else (r["replan"])
                c = sim.cost_of(a, d, off); store[f"{fc}|{pn}|d{d}|S{S}"] = c.astype(np.float32)
                sv = 1 - c / asap
                for per_name, mk in [("all", np.ones(len(a), bool)), ("2023-24", ~yr26), ("2026", yr26)]:
                    if mk.sum() < 50: continue
                    summary.append(dict(fc=fc, pol=pn, d=d, S=S, period=per_name, n=int(mk.sum()), saving_pct=100 * (1 - c[mk].mean() / asap[mk].mean()),
                                        oracle_frac=(1 - c[mk].mean() / asap[mk].mean()) / (1 - orc_static[mk].mean() / asap[mk].mean()),
                                        backfire_pct=100 * float(np.mean(c[mk] > asap[mk] + 1e-9)), p5_job_saving_pct=100 * float(np.percentile(sv[mk], 5)),
                                        regret_g=float(np.mean(c[mk] - orc_static[mk])), asap_ci=float(asap[mk].mean())))
        print(f"d={d} S={S} n={len(a)} oracle {100*osav:.1f}%  ({time.time()-t:.0f}s, total {time.time()-t00:.0f}s)", flush=True)
pd.DataFrame(summary).to_parquet("/home/claude/results/decisions_summary.parquet")
np.savez_compressed("/home/claude/results/decisions_store.npz", **store)
print("DONE", f"{time.time()-t00:.0f}s")
