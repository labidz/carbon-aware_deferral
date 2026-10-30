import numpy as np, sim, ukciv as U, pandas as pd, warnings, time; warnings.filterwarnings("ignore")
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); W = np.load("/home/claude/data/wf_main.npz")
s, L, f0 = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"]; blk = W["blk"]
raw = sim.PathSet(sim.to_matrix(s, L, f0))
med = sim.PathSet(sim.to_matrix(s, L, np.where(blk >= 0, f0 + W["q50"], np.nan)))
orc = sim.PathSet(sim.actual_matrix())
# arrivals: captures inside WF blocks (>= Jul 2023), thinned
ci_blk = np.full(sim.NC, -1)
ci = np.searchsorted(sim.CAPS, s); ci_blk[ci[blk >= 0]] = blk[blk >= 0]
arr = np.where(ci_blk >= 1)[0][::2]
t = time.time()
print("arrivals", len(arr))
for d in [2, 8]:
    for S in [12, 24, 48]:
        ok = sim.valid_arrivals(arr, d, S, [raw, med, orc]); a = arr[ok]
        asap = sim.cost_of(a, d, np.zeros(len(a), int))
        out = {}
        for nm, ps in [("NESO", raw), ("ML-med", med), ("Oracle", orc)]:
            r = sim.run_policies(a, d, S, ps)
            for pol, off in r.items(): out[(nm, pol)] = sim.cost_of(a, d, off)
        orc_sav = 1 - out[("Oracle", "static")].mean() / asap.mean()
        line = f"d={d:2d} S={S:2d} n={ok.sum():5d} | " + " | ".join(f"{k[0]}-{k[1][:3]} {100*(1-v.mean()/asap.mean()):5.2f}% ({100*(1-v.mean()/asap.mean())/orc_sav:4.0f}% of orc, backfire {100*np.mean(v>asap):4.1f}%)" for k, v in out.items() if k[0] != "Oracle" or k[1] == "static")
        print(line, flush=True)
print(f"{time.time()-t:.0f}s")
