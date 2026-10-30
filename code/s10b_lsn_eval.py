import numpy as np, pandas as pd, json, sim, ukciv as U, warnings; warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)
st = np.load("/home/claude/results/decisions_store.npz"); ls = np.load("/home/claude/results/lsn_store.npz")
wl = json.load(open("/home/claude/results/workload.json"))["all_gpu_jobs"]["bins"]; wgt = {b["slots"]: b["gpu_hour_share"] for b in wl if b["slots"]}
M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); Wm = np.load("/home/claude/data/wf_main.npz"); s, L, blk = M["s"].astype(np.int64), M["L"].astype(np.int64), Wm["blk"]
cap_blk = np.full(sim.NC, -1); ci = np.searchsorted(sim.CAPS, s); cap_blk[ci[blk >= 0]] = blk[blk >= 0]
day_all = sim.CAPS // 48; uday, day_id = np.unique(day_all, return_inverse=True); ND = len(uday)
B = 1000; CNT = np.vstack([np.ones(ND), rng.multinomial(ND, np.ones(ND) / ND, size=B)]).astype(float)
DS = [1, 2, 4, 8, 16, 24]; SS = [2, 6, 12, 24, 48]
def dv(c, a): return np.bincount(day_id[a], weights=np.nan_to_num(c.astype(float)), minlength=ND)
def wfrac(key_fc, S, blocks):
    ns = 0; no = 0; tot = 0
    for d in DS:
        a = st[f"a|d{d}|S{S}"]; m = np.isin(cap_blk[a], blocks); a = a[m]
        c = (ls[f"LSN|d{d}|S{S}"] if key_fc == "LSN" else st[f"{key_fc}|d{d}|S{S}"])[m]
        A_ = CNT @ dv(st[f"asap|d{d}|S{S}"][m], a); C_ = CNT @ dv(c, a); O_ = CNT @ dv(st[f"oracle|d{d}|S{S}"][m], a); w = wgt[d]
        ns = ns + w * (1 - C_ / A_); no = no + w * (1 - O_ / A_); tot += w
    return ns / tot, (ns / tot) / (no / tot)
ci_ = lambda v: [float(v[0]), float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))]
out = {}
for name, blocks in [("val(Nov-Dec 2023)", [5, 6]), ("test(2024+2026)", [7, 8, 9, 10, 11])]:
    out[name] = {}
    for S in SS:
        r = {}
        for lab, key in [("NESO-replan", "NESO|replan"), ("ML-median-replan", "ML-median|replan"), ("ML-median-static", "ML-median|static"), ("ML-median-replan-h2", "ML-median|replan_h2"), ("LSN", "LSN")]:
            sv, fr = wfrac(key, S, blocks); r[lab] = dict(saving=ci_(100 * sv), oracle_frac=ci_(fr))
        _, fl = wfrac("LSN", S, blocks); _, fm = wfrac("ML-median|replan", S, blocks); _, fn = wfrac("NESO|replan", S, blocks)
        r["LSN_minus_ML-replan_pp"] = ci_(100 * (fl - fm)); r["LSN_minus_NESO-replan_pp"] = ci_(100 * (fl - fn)); out[name][f"S{S}"] = r
    # tail-risk at d=8,S=24 on same arrivals
    a = st["a|d8|S24"]; m = np.isin(cap_blk[a], blocks); asap = st["asap|d8|S24"][m]; tr = {}
    for lab, c in [("NESO-replan", st["NESO|replan|d8|S24"][m]), ("ML-median-replan", st["ML-median|replan|d8|S24"][m]), ("ML-median+risk0.5-replan", st["ML-median+risk0.5|replan|d8|S24"][m]), ("LSN", ls["LSN|d8|S24"][m]), ("Oracle", st["oracle|d8|S24"][m])]:
        tr[lab] = dict(saving_pct=float(100 * (1 - c.mean() / asap.mean())), backfire_pct=float(100 * np.mean(c > asap + 1e-9)), p5_job_saving_pct=float(100 * np.percentile(1 - c / asap, 5)), n=int(m.sum()))
    out[name]["tail_d8_S24"] = tr
json.dump(out, open("/home/claude/results/lsn_eval.json", "w"), indent=1)
for name in out:
    print("\n", name)
    for S in [6, 24, 48]:
        r = out[name][f"S{S}"]; print(f"  S={S}", {k: round(v["oracle_frac"][0], 3) for k, v in r.items() if isinstance(v, dict) and "oracle_frac" in v}, " LSN-ML(pp):", [round(x, 2) for x in r["LSN_minus_ML-replan_pp"]])
    print("  tail d8 S24:", {k: (round(v["saving_pct"], 2), round(v["backfire_pct"], 1)) for k, v in out[name]["tail_d8_S24"].items()})
