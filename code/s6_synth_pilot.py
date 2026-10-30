import numpy as np, sim, ukciv as U, warnings, time; warnings.filterwarnings("ignore")
rng = np.random.default_rng(1)
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); W = np.load("/home/claude/data/wf_main.npz")
s, L, f0, y = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], M["y"]; blk = W["blk"]
ORC = sim.actual_matrix()
raw = sim.PathSet(sim.to_matrix(s, L, f0)); orc = sim.PathSet(ORC)
# real error statistics per lead (blocks>=0 and any block incl. training region for stats is fine; use WF blocks)
m = np.isfinite(y) & (blk >= 0); e = (y - f0)
mae_L = np.array([np.mean(np.abs(e[m & (L == l)])) for l in range(96)]); rmse_L = np.array([np.sqrt(np.mean(e[m & (L == l)] ** 2)) for l in range(96)])
bias_L = np.array([np.mean(e[m & (L == l)]) for l in range(96)])
# cross-lead correlation of real error paths (lag-1)
EM = sim.to_matrix(s, L, np.where(m, e, np.nan)); 
rho1 = np.nanmean([np.corrcoef(EM[np.isfinite(EM[:,l])&np.isfinite(EM[:,l+1]), l], EM[np.isfinite(EM[:,l])&np.isfinite(EM[:,l+1]), l+1])[0,1] for l in range(0, 94, 6)])
print("mean MAE_L %.2f  mean lag-1 lead-corr of real error paths %.2f  bias-mean %.2f" % (mae_L.mean(), rho1, bias_L.mean()))
nc = sim.NC
def mk(noise): return sim.PathSet(np.where(np.isfinite(ORC), ORC - noise, np.nan).astype(np.float32))
variants = {}
# (a) fixed-per-target uniform multiplicative noise (Wiesner-style), amplitude matched to overall MAE: E|U(-x,x)|*mean(A)=x/2*mean(A)
xa = 2 * mae_L.mean() / np.nanmean(U.A); eps_w = rng.uniform(-xa, xa, U.T)
idx = np.clip(sim.CAPS[:, None] + np.arange(96)[None, :], 0, U.T - 1); variants["syn_fixed_uniform"] = mk(eps_w[idx] * ORC)
# (b) iid gaussian per (capture, lead), sigma matched per-lead to real MAE
variants["syn_iid_gauss"] = mk(rng.normal(0, 1, (nc, 96)) * (mae_L * 1.2533)[None, :])
# (c) AR(1) across leads, per-lead sd matched
z = np.zeros((nc, 96)); z[:, 0] = rng.normal(size=nc)
for l in range(1, 96): z[:, l] = rho1 * z[:, l-1] + np.sqrt(1 - rho1**2) * rng.normal(size=nc)
variants["syn_ar1_gauss"] = mk(z * (mae_L * 1.2533)[None, :])
# (d) real error paths bootstrapped from another capture of the same block & hour (keeps lead structure + bias)
cap_blk = np.full(nc, -1); ci = np.searchsorted(sim.CAPS, s); cap_blk[ci[blk >= 0]] = blk[blk >= 0]
EMn = np.where(np.isfinite(EM), EM, 0.0); boot = np.zeros((nc, 96), np.float32)
for b in range(12):
    ids = np.where(cap_blk == b)[0]
    if len(ids): boot[ids] = EMn[rng.choice(ids, len(ids))]
variants["syn_real_error_bootstrap"] = sim.PathSet(np.where(np.isfinite(ORC), ORC - boot, np.nan).astype(np.float32))
arr = np.where(cap_blk >= 1)[0][::2]
for d, S in [(2, 24), (8, 24), (8, 48)]:
    ok = sim.valid_arrivals(arr, d, S, [raw, orc] + list(variants.values())); a = arr[ok]
    asap = sim.cost_of(a, d, np.zeros(len(a), int)); o = sim.cost_of(a, d, sim.run_policies(a, d, S, orc)["static"]); osav = 1 - o.mean() / asap.mean()
    print(f"\nd={d} S={S} n={len(a)} oracle saving {100*osav:.2f}%")
    for nm, ps in [("REAL NESO", raw)] + list(variants.items()):
        r = sim.run_policies(a, d, S, ps); out = []
        for pol in ["static", "replan"]:
            c = sim.cost_of(a, d, r[pol]); out.append(f"{pol[:3]} {100*(1-c.mean()/asap.mean()):5.2f}% ({100*(1-c.mean()/asap.mean())/osav:4.0f}% of orc)")
        print(f"  {nm:28s}", " | ".join(out))
