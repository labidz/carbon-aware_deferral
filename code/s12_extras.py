"""GB extras: (B) forecast-free baselines (climatology plan, threshold rule); (A) multi-seed synthetic generators incl. calibrated Berkson-type generator."""
import numpy as np, pandas as pd, json, time, warnings; warnings.filterwarnings("ignore")
import sim, ukciv as U
t00 = time.time(); st = np.load("/home/claude/results/decisions_store.npz")
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); Wm = np.load("/home/claude/data/wf_main.npz")
s, L, f0, y, blk = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], M["y"], Wm["blk"]
cap_blk = np.full(sim.NC, -1); ci = np.searchsorted(sim.CAPS, s); cap_blk[ci[blk >= 0]] = blk[blk >= 0]
DS = [1, 2, 4, 8, 16, 24]; SS = [2, 6, 12, 24, 48]; H = 96; T = U.T
wl = json.load(open("/home/claude/results/workload.json"))["all_gpu_jobs"]["bins"]; WG = {b["slots"]: b["gpu_hour_share"] for b in wl if b["slots"]}
extras = {}
import os
A = U.A.astype(np.float64); idx = sim.CAPS[:, None] + np.arange(H)[None, :]
STAGE1 = os.path.exists("/home/claude/results/extras_store.npz") and os.path.exists("/home/claude/results/clim.npy")
if not STAGE1:
    # ------------------------------------------------------------------ B1: climatology plan (mean of same time of day over the previous 14 settled days)
    A = U.A.astype(np.float64); idx = sim.CAPS[:, None] + np.arange(H)[None, :]            # target windows (NC,H)
    k0 = np.ceil((np.arange(H) + 49) / 48).astype(int)                                       # first day offset whose window is settled at issue
    clim = np.full((sim.NC, H), np.nan, np.float32)
    acc = np.zeros((sim.NC, H)); cnt = np.zeros((sim.NC, H))
    for i in range(14):
        w = idx - 48 * (k0[None, :] + i); ok = (w >= 0) & (w < T); v = np.where(ok, A[np.clip(w, 0, T - 1)], np.nan)
        acc += np.nan_to_num(v); cnt += np.isfinite(v)
    clim = np.where(cnt >= 7, acc / np.maximum(cnt, 1), np.nan).astype(np.float32); clim_ps = sim.PathSet(clim)
    np.save("/home/claude/results/clim.npy", clim)
    print("climatology built", f"{time.time()-t00:.0f}s", flush=True)
    # ------------------------------------------------------------------ B2: threshold rule on latest visible actual vs rolling 7-day percentile
    QS = [10, 25, 40, 55, 70, 85]; pct = {q: np.full(T, np.nan) for q in QS}
    for u in range(400, T):
        win = A[max(0, u - 49 - 335): u - 49 + 1]; win = win[np.isfinite(win)]
        if len(win) >= 100:
            v = np.percentile(win, QS)
            for q, x in zip(QS, v): pct[q][u] = x
    print("percentiles built", f"{time.time()-t00:.0f}s", flush=True)
    def run_threshold(a, d, S, q):
        s_ = sim.CAPS[a]; n = len(a); started = np.zeros(n, bool); start = np.full(n, S)
        for j in range(S + 1):
            u = s_ + j; cur = U.vis_actual(u - 2, u); thr = pct[q][np.clip(u, 0, T - 1)]
            go = (np.isfinite(cur) & np.isfinite(thr) & (cur <= thr)) | (j == S); new = go & ~started; start[new] = j; started |= new
        return start
    for d in DS:
        for S_ in SS:
            a = st[f"a|d{d}|S{S_}"]
            r = sim.run_policies(a, d, S_, clim_ps); extras[f"Clim|static|d{d}|S{S_}"] = sim.cost_of(a, d, r["static"]).astype(np.float32)
            for q in QS: extras[f"Thr{q}|rule|d{d}|S{S_}"] = sim.cost_of(a, d, run_threshold(a, d, S_, q)).astype(np.float32)
        print("baselines d", d, f"{time.time()-t00:.0f}s", flush=True)
    np.savez_compressed("/home/claude/results/extras_store.npz", **extras)

# ------------------------------------------------------------------ A: synthetic generators, 6 seeds, evaluated on blocks >= 5 (Nov 2023 onward); Berkson fitted on Jun-Oct 2023
NS = 6; TRAIN_END = int((pd.Timestamp("2023-10-29") - pd.Timestamp(U.t0, unit="s")).total_seconds() // 1800)
Fl = U.F[:, 0 - U.LMIN: 0 - U.LMIN + H].astype(np.float64); K = 8
Xw = np.stack([np.where((np.arange(T) + k >= 0) & (np.arange(T) + k < T), A[np.clip(np.arange(T) + k, 0, T - 1)], np.nan) for k in range(-K, K + 1)], 1)   # (T,17)
BK = [(0, 4), (4, 12), (12, 24), (24, 36), (36, 48), (48, 60), (60, 72), (72, 96)]; fitted = np.full((T, H), np.nan)
for lo, hi in BK:
    rows = []; ys = []
    for l in range(lo, hi):
        w = np.arange(T); sl = w - l; ok = (sl >= 0) & (sl < TRAIN_END) & np.isfinite(Fl[:, l]) & np.isfinite(Xw).all(1)
        rows.append(Xw[ok]); ys.append(Fl[ok, l])
    Xb = np.concatenate(rows); yb = np.concatenate(ys); mu = Xb.mean(0); Xc = Xb - mu; ym = yb.mean()
    beta = np.linalg.solve(Xc.T @ Xc + 1e-3 * len(yb) * np.eye(Xc.shape[1]), Xc.T @ (yb - ym))
    okx = np.isfinite(Xw).all(1)
    for l in range(lo, hi): fitted[okx, l] = (Xw[okx] - mu) @ beta + ym
print("berkson fitted", f"{time.time()-t00:.0f}s", flush=True)
Rf = Fl - fitted
caps_tr = np.where(sim.CAPS < TRAIN_END)[0]; RP = np.where(np.isfinite(Rf[np.clip(sim.CAPS[caps_tr][:, None] + np.arange(H), 0, T - 1), np.arange(H)[None, :]]), Rf[np.clip(sim.CAPS[caps_tr][:, None] + np.arange(H), 0, T - 1), np.arange(H)[None, :]], np.nan)
good = np.isfinite(RP).sum(1) >= 90; caps_tr = caps_tr[good]; RP = np.nan_to_num(RP[good])
tod_tr = sim.CAPS[caps_tr] % 48; groups = {t: np.where(tod_tr == t)[0] for t in range(48)}
FIT = np.where(np.isfinite(fitted[np.clip(idx, 0, T - 1), np.arange(H)[None, :]]) & (idx < T), fitted[np.clip(idx, 0, T - 1), np.arange(H)[None, :]], np.nan)
def gen_berk(rng):
    j = np.array([rng.choice(groups[t]) if len(groups[t]) else rng.integers(0, len(RP)) for t in (sim.CAPS % 48)]); return (FIT + RP[j]).astype(np.float32)
m_ = np.isfinite(y) & (blk >= 0); e = y - f0; mae_L = np.array([np.mean(np.abs(e[m_ & (L == l)])) for l in range(H)]); sig = (mae_L * 1.2533)[None, :]
ORC = sim.actual_matrix(); EM = sim.to_matrix(s, L, np.where(m_, e, np.nan)); EMn = np.where(np.isfinite(EM), EM, 0.0)
def gen_additive(kind, rng):
    nc = sim.NC
    if kind == "fixed_uniform": x = 2 * mae_L.mean() / np.nanmean(U.A); eps = rng.uniform(-x, x, T); noise = eps[np.clip(idx, 0, T - 1)] * ORC
    elif kind == "iid": noise = rng.normal(size=(nc, H)) * sig
    elif kind == "ar1":
        z = np.zeros((nc, H)); z[:, 0] = rng.normal(size=nc)
        for l in range(1, H): z[:, l] = 0.98 * z[:, l - 1] + np.sqrt(1 - 0.98 ** 2) * rng.normal(size=nc)
        noise = z * sig
    elif kind == "bootstrap":
        noise = np.zeros((nc, H), np.float32)
        for b in range(12):
            ids = np.where(cap_blk == b)[0]
            if len(ids): noise[ids] = EMn[rng.choice(ids, len(ids))]
    return np.where(np.isfinite(ORC), ORC - noise, np.nan).astype(np.float32)
GENS = ["fixed_uniform", "iid", "ar1", "bootstrap", "berkson"]
PJ = "/home/claude/results/gb_synthetic_partial.json"
rec = json.load(open(PJ)) if os.path.exists(PJ) else {}   # key "g|seed" -> {pol: {S: [syn, real]}}
idx = np.clip(sim.CAPS[:, None] + np.arange(H)[None, :], 0, T - 1)
for sd in range(NS):
    for g in GENS:
        key = f"{g}|{sd}"
        if key in rec: continue
        rng = np.random.default_rng(100 * sd + GENS.index(g)); ps = sim.PathSet(gen_berk(rng) if g == "berkson" else gen_additive(g, rng)); rec[key] = {pol: {} for pol in ["static", "replan"]}
        for S_ in SS:
            num = {pol: 0 for pol in ["static", "replan"]}; numr = {pol: 0 for pol in ["static", "replan"]}; den = 0
            for d in DS:
                a_all = st[f"a|d{d}|S{S_}"]; mk = (cap_blk[a_all] >= 5); a = a_all[mk]; ok = sim.valid_arrivals(a, d, S_, [ps]); a = a[ok]
                asap = st[f"asap|d{d}|S{S_}"][mk][ok]; orc = st[f"oracle|d{d}|S{S_}"][mk][ok]; w = WG[d]; den += w * (1 - orc.sum() / asap.sum())
                r = sim.run_policies(a, d, S_, ps)
                for pol in ["static", "replan"]:
                    num[pol] += w * (1 - sim.cost_of(a, d, r[pol]).sum() / asap.sum()); cr_ = st[f"NESO|{pol}|d{d}|S{S_}"][mk][ok]; numr[pol] += w * (1 - cr_.sum() / asap.sum())
            for pol in ["static", "replan"]: rec[key][pol][str(S_)] = [float(num[pol] / den), float(numr[pol] / den)]
        json.dump(rec, open(PJ, "w"))
    print("seed", sd, f"{time.time()-t00:.0f}s", flush=True)
json.dump(rec, open("/home/claude/results/gb_synthetic.json", "w"), indent=1)
print("DONE", f"{time.time()-t00:.0f}s")
