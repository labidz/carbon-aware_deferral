"""Learned start-now classifier (LSN) trained on hindsight-optimal labels from out-of-sample ML paths."""
import numpy as np, sim, json, time, lightgbm as lgb, pandas as pd, ukciv as U, warnings; warnings.filterwarnings("ignore")
t00 = time.time()
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True)
s, L, f0, y = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], M["y"]
Wm = np.load("/home/claude/data/wf_main.npz"); blk = Wm["blk"]
q50, q10, q90 = (np.nan_to_num(Wm[k]) for k in ["q50", "q10", "q90"])
def P(v): return sim.PathSet(sim.to_matrix(s, L, np.where(blk >= 0, v, np.nan).astype(np.float32)))
med, raw, HW = P(f0 + q50), P(f0), P(np.abs(q90 - q10) / 2)
cap_blk = np.full(sim.NC, -1); ci = np.searchsorted(sim.CAPS, s); cap_blk[ci[blk >= 0]] = blk[blk >= 0]
st = np.load("/home/claude/results/decisions_store.npz")

def feats(a_idx, d, S, j):
    n = len(a_idx); s_ = sim.CAPS[a_idx]; ci_, base = sim.epoch_view(s_, j); u = s_ + j
    c = med.costs(ci_, base, S - j, d); cr = raw.costs(ci_, base, S - j, d); hw = HW.costs(ci_, base, S - j, d)
    c0 = c[:, 0]; alt = c[:, 1:]; calt = np.nanmin(alt, 1); ostar = np.nanargmin(alt, 1) + 1
    c0r = cr[:, 0]; calt_r = np.nanmin(cr[:, 1:], 1)
    f = {"d": np.full(n, d), "S": np.full(n, S), "j": np.full(n, j), "r": np.full(n, S - j), "stale": base,
         "c0": c0, "gap": c0 - calt, "relgap": (c0 - calt) / np.maximum(c0, 1), "ostar": ostar, "gap_raw": c0r - calt_r, "relgap_raw": (c0r - calt_r) / np.maximum(c0r, 1),
         "med_minus_raw_now": c0 - c0r, "hw0": hw[:, 0], "hw_star": hw[np.arange(n), ostar], "hw_ratio": hw[:, 0] / np.maximum(hw[np.arange(n), ostar], 1e-3),
         "tod_sin": np.sin(2 * np.pi * (u % 48) / 48), "tod_cos": np.cos(2 * np.pi * (u % 48) / 48)}
    if j >= 2:
        cip, bp = sim.epoch_view(s_, j - 2); cp = med.costs(cip, bp, S - j + 2, d)
        pg = cp[:, 2] - np.nanmin(cp[:, 3:], 1) if cp.shape[1] > 3 else np.full(n, np.nan)
        f["dgap2"] = f["gap"] - pg; f["dc0_2"] = c0 - cp[:, 2]
    else: f["dgap2"] = np.full(n, np.nan); f["dc0_2"] = np.full(n, np.nan)
    m48 = med.costs(ci_, base, 0, 48)[:, 0]; f["lvl48"] = c0 - m48
    names = list(f); Xf = np.stack([np.asarray(f[k], dtype=np.float32) for k in names], 1)
    act = sim.realised(s_[:, None] + j + np.arange(S - j + 1)[None, :], d)
    best_later = np.nanmin(act[:, 1:], 1); lab = (act[:, 0] <= best_later + 1e-9).astype(int); wt = np.abs(act[:, 0] - best_later) + 0.5
    return Xf, names, lab, wt

# ---- training rows (blocks 1-4, ex last 3 days before Nov-2023) ----
tr_cut = (np.datetime64("2023-10-29") - np.datetime64("1970-01-01")).astype("timedelta64[s]").astype(int)
Xs, ys, ws = [], [], []
for d in [2, 4, 8, 16]:
    for S in [6, 12, 24, 48]:
        a = st[f"a|d{d}|S{S}"]; keep = np.isin(cap_blk[a], [1, 2, 3, 4]) & (U.t0 + 1800 * sim.CAPS[a] < tr_cut); a = a[keep][::3]
        for j in range(S):
            Xf, names, lab, wt = feats(a, d, S, j); ok = np.isfinite(Xf[:, names.index("c0")]) & np.isfinite(Xf[:, names.index("gap")]); Xs.append(Xf[ok]); ys.append(lab[ok]); ws.append(wt[ok])
Xtr, ytr, wtr = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
print("LSN training rows", Xtr.shape, "pos rate %.3f" % ytr.mean(), f"{time.time()-t00:.0f}s", flush=True)
clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=500, colsample_bytree=0.8, subsample=0.7, subsample_freq=1, reg_lambda=30, n_jobs=1, verbose=-1)
clf.fit(Xtr, ytr, sample_weight=wtr)
imp = dict(zip(names, clf.booster_.feature_importance("gain").round(0).tolist()))

def run_lsn(a, d, S, theta):
    n = len(a); started = np.zeros(n, bool); start = np.full(n, S)
    for j in range(S + 1):
        if j == S: new = ~started
        else:
            Xf, nm, _, _ = feats(a, d, S, j); p = clf.predict_proba(np.nan_to_num(Xf, nan=np.nan))[:, 1]; new = (p >= theta) & ~started
        start[new] = j; started |= new
        if started.all(): break
    return start

thetas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]; DS = [1, 2, 4, 8, 16, 24]; SS = [2, 6, 12, 24, 48]
# ---- tune theta on validation blocks 5-6 (grid d in {2,4,8,16}, S in {6,12,24,48}), average relative saving ----
val_score = {th: [] for th in thetas}
for d in [2, 4, 8, 16]:
    for S in [6, 12, 24, 48]:
        a_all = st[f"a|d{d}|S{S}"]; asap_all = st[f"asap|d{d}|S{S}"]; mk = np.isin(cap_blk[a_all], [5, 6]); a = a_all[mk]; asap = asap_all[mk]
        for th in thetas: val_score[th].append(1 - sim.cost_of(a, d, run_lsn(a, d, S, th)).mean() / asap.mean())
vs = {th: float(np.mean(v)) for th, v in val_score.items()}; theta = max(vs, key=vs.get)
print("validation mean saving by theta", {k: round(100 * v, 2) for k, v in vs.items()}, "-> theta", theta, f"{time.time()-t00:.0f}s", flush=True)
# ---- evaluate on blocks 5..11 for the full (d,S) grid, store costs aligned with main-grid arrival arrays ----
out = {}
for d in DS:
    for S in SS:
        a = st[f"a|d{d}|S{S}"]; mk = cap_blk[a] >= 5; c = np.full(len(a), np.nan, np.float32)
        c[mk] = sim.cost_of(a[mk], d, run_lsn(a[mk], d, S, theta)); out[f"LSN|d{d}|S{S}"] = c
    print("d", d, f"{time.time()-t00:.0f}s", flush=True)
np.savez_compressed("/home/claude/results/lsn_store.npz", **out)
json.dump({"theta": theta, "val_saving": vs, "feature_gain": imp, "n_train": int(len(ytr))}, open("/home/claude/results/lsn_meta.json", "w"), indent=1)
print("DONE", f"{time.time()-t00:.0f}s")
