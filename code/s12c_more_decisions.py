"""Decisions for additional forecasters: causal-tuned ML median, stand-alone LightGBM."""
import numpy as np, sim, ukciv as U, time, warnings; warnings.filterwarnings("ignore")
t00 = time.time(); st = np.load("/home/claude/results/decisions_store.npz")
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); Wm = np.load("/home/claude/data/wf_main.npz")
s, L, f0, blk = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], Wm["blk"]
def P(res_pred): v = np.where(np.isfinite(res_pred), f0 + res_pred, np.nan).astype(np.float32); return sim.PathSet(sim.to_matrix(s, L, v))
F = {"ML-causal": P(np.load("/home/claude/data/wf_causal.npz")["q50"]), "Standalone": P(np.load("/home/claude/data/wf_standalone.npz")["q50"])}
DS = [1, 2, 4, 8, 16, 24]; SS = [2, 6, 12, 24, 48]; out = {}
for d in DS:
    for S in SS:
        a = st[f"a|d{d}|S{S}"]
        for nm, ps in F.items():
            ok = sim.valid_arrivals(a, d, S, [ps]); r = sim.run_policies(a, d, S, ps)       # NaN-cost arrivals (e.g. block 0 / gaps) are masked via NaN
            for pol in ["static", "replan"]:
                c = sim.cost_of(a, d, r[pol]).astype(np.float32); c[~ok] = np.nan; out[f"{nm}|{pol}|d{d}|S{S}"] = c
    print("d", d, f"{time.time()-t00:.0f}s", flush=True)
np.savez_compressed("/home/claude/results/extras2_store.npz", **out); print("DONE")
