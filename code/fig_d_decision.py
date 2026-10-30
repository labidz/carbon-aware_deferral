import numpy as np, pandas as pd, json, sys, matplotlib.pyplot as plt, sim, ukciv as U
from figstyle import *
which = sys.argv[1:] or ["7", "8", "9", "10"]
DT = json.load(open("/home/claude/results/decision_tables.json")); ch = json.load(open("/home/claude/results/characterize.json"))
S_ = pd.read_parquet("/home/claude/results/decisions_summary.parquet"); SS = [2, 6, 12, 24, 48]; DS = [1, 2, 4, 8, 16, 24]
if "7" in which:
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.8), gridspec_kw={"width_ratios": [1.25, 1.25, 0.9]}); a, b, c = axs
    hrs = [s_ / 2 for s_ in SS]; syn = DT["synthetic_gap"]
    gens = [("syn:fixed-uniform", C["gold"], "o", "uniform, fixed per target"), ("syn:iid-gauss", C["pink"], "s", "i.i.d. Gaussian"), ("syn:ar1-gauss", C["orc"], "^", "AR(1) Gaussian"), ("syn:real-error-bootstrap", C["ml2"], "D", "resampled real error paths")]
    allv = [v[0] for k, v in syn.items()]; ymin = np.floor((min(allv) - 0.02) * 20) / 20
    for ax, pol, ttl, lab in [(a, "static", "Commit-once policy", "a"), (b, "replan", "Re-planning policy", "b")]:
        ax.plot(hrs, [syn[f"NESO|{pol}|S{S}"][0] for S in SS], "o-", color="k", lw=2, ms=4, label="real NESO forecast")
        ax.fill_between(hrs, [syn[f"NESO|{pol}|S{S}"][1] for S in SS], [syn[f"NESO|{pol}|S{S}"][2] for S in SS], color="k", alpha=0.12, lw=0)
        for k, col, m, l in gens: ax.plot(hrs, [syn[f"{k}|{pol}|S{S}"][0] for S in SS], m + "--", color=col, ms=3.5, lw=1.1, label=l)
        ax.set_xscale("log"); ax.set_xticks(hrs); ax.set_xticklabels([f"{h:g}" for h in hrs]); ax.set_xlabel("Slack (h)"); ax.set_ylim(ymin, 0.96); ax.set_title(ttl); panel(ax, lab)
    a.set_ylabel("Share of oracle saving captured"); h_, l_ = b.get_legend_handles_labels()
    sh = ch["shrinkage"]; ls_ = ["L=2.0h", "L=6.0h", "L=12.0h", "L=24.0h"]; x = np.arange(4)
    c.bar(x - 0.18, [sh[k]["corr_A_e"] for k in ls_], 0.36, color="k", label="real NESO forecast"); pass
    c.set_xticks(x); c.set_xticklabels(["2", "6", "12", "24"]); c.set_xlabel("Lead (h)"); c.set_ylabel("corr(actual, forecast error)"); c.set_title("Error is coupled to truth"); c.axhline(0, color=C["gold"], lw=2.5, label="additive synthetic noise (0 by construction)"); h2, l2 = c.get_legend_handles_labels(); c.set_ylim(-0.03, 0.55); panel(c, "c")
    fig.legend(h_ + [plt.Line2D([0], [0], color=C["gold"], lw=2.5)], l_ + ["additive synthetic noise\n(zero by construction)"], loc="lower center", ncol=3, fontsize=6.2, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout(); save(fig, "fig7_synthetic_gap")
if "8" in which:
    P = S_[(S_.period == "all") & (S_.pol == "replan")]
    def mat(fc): return P[P.fc == fc].pivot(index="d", columns="S", values="oracle_frac").loc[DS, SS].values
    A_, B_ = mat("NESO"), mat("ML-median"); D_ = 100 * (B_ - A_)
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5))
    for ax, M, ttl, cm, vmin, vmax, lab, fmt in [(axs[0], A_, "NESO, re-planning", "viridis", 0.55, 0.95, "a", "{:.2f}"), (axs[1], B_, "ML median, re-planning", "viridis", 0.55, 0.95, "b", "{:.2f}"), (axs[2], D_, "Difference (pp of oracle)", "PuOr_r", -8, 8, "c", "{:+.1f}")]:
        im = ax.imshow(M, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto"); ax.grid(False)
        ax.set_xticks(range(5)); ax.set_xticklabels([f"{s_/2:g}" for s_ in SS]); ax.set_yticks(range(6)); ax.set_yticklabels([f"{d/2:g}" for d in DS]); ax.set_xlabel("Slack (h)")
        for i in range(6):
            for j in range(5): ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6, color="white" if (cm == "viridis" and M[i, j] < 0.78) else "black")
        ax.set_title(ttl); panel(ax, lab); plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    axs[0].set_ylabel("Job duration (h)"); fig.tight_layout(); save(fig, "fig8_heatmaps")
if "9" in which:
    st = np.load("/home/claude/results/decisions_store.npz"); ls = np.load("/home/claude/results/lsn_store.npz"); lm = json.load(open("/home/claude/results/lsn_meta.json"))
    M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); Wm = np.load("/home/claude/data/wf_main.npz"); s, blk = M["s"].astype(np.int64), Wm["blk"]
    cap_blk = np.full(sim.NC, -1); ci = np.searchsorted(sim.CAPS, s); cap_blk[ci[blk >= 0]] = blk[blk >= 0]
    a = st["a|d8|S24"]; m = np.isin(cap_blk[a], [7, 8, 9, 10, 11]); asap = st["asap|d8|S24"][m]; day = (sim.CAPS[a][m] // 48); ud, inv = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(3); cnt = rng.multinomial(len(ud), np.ones(len(ud)) / len(ud), size=1000).astype(float)
    pts = [("NESO", "static", "NESO commit-once", C["neso"], "o"), ("NESO", "replan", "NESO re-plan", C["neso"], "s"), ("NESO", "replan_h2", "NESO re-plan + 2% hysteresis", C["neso"], "^"),
           ("ML-median", "static", "ML commit-once", C["ml"], "o"), ("ML-median", "replan", "ML re-plan", C["ml"], "s"), ("ML-median", "replan_h2", "ML re-plan + 2% hysteresis", C["ml"], "^"),
           ("ML-median+risk0.5", "replan", "ML risk-aware ($\\lambda$=0.5)", C["orc"], "D"), ("ML-median+risk1.0", "replan", "ML risk-aware ($\\lambda$=1)", C["orc"], "P"), ("NESO+RollBias7d", "replan", "NESO + rolling bias", C["gold"], "X"), ("LSN", "", "Learned start-now (LSN)", C["pink"], "*")]
    def boot_sv(c):
        dc = np.bincount(inv, weights=c, minlength=len(ud)); da = np.bincount(inv, weights=asap, minlength=len(ud)); return 100 * (1 - (cnt @ dc) / (cnt @ da)), 100 * (1 - c.mean() / asap.mean())
    ref_bs, ref_sv = boot_sv(st["NESO|replan|d8|S24"][m])
    fig, ax = plt.subplots(figsize=(W1 * 1.55, 3.3))
    for fc, pol, lab, col, mk in pts:
        c = (ls["LSN|d8|S24"] if fc == "LSN" else st[f"{fc}|{pol}|d8|S24"])[m]
        bs, sv = boot_sv(c); bf = 100 * np.mean(c > asap + 1e-9); dd = bs - ref_bs; dsv = sv - ref_sv
        ax.errorbar(bf, dsv, yerr=[[dsv - np.percentile(dd, 2.5)], [np.percentile(dd, 97.5) - dsv]], fmt=mk, color=col, ms=6 if mk != "*" else 9, capsize=2, label=lab)
    obs, osv = boot_sv(st["oracle|d8|S24"][m]); ax.axhline(osv - ref_sv, color="k", ls="--", lw=0.8); ax.text(13.9, osv - ref_sv + 0.06, "oracle (perfect foresight)", fontsize=6.3, ha="left")
    ax.axhline(0, color=C["neso"], ls=":", lw=0.8); ax.invert_xaxis()
    ax.set_xlabel("Backfire rate (%): jobs worse than starting immediately  [better $\\rightarrow$]"); ax.set_ylabel("Saving relative to NESO re-plan (pp)\n(paired day-block 95% CI)")
    ax.set_title("4-h job, 12-h slack; held-out Jan-Feb 2024 and Jul-Oct 2026", fontsize=7.5)
    ax.legend(fontsize=5.6, loc="lower left", bbox_to_anchor=(1.01, 0.0)); fig.tight_layout(); save(fig, "fig9_policy_frontier")
if "10" in which:
    fe = json.load(open("/home/claude/results/forecast_eval.json")); W = DT["weighted"]; wl = json.load(open("/home/claude/results/workload.json"))["all_gpu_jobs"]["bins"]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw={"width_ratios": [1.2, 1]}); a, b = axs
    lab = {"NESO": "NESO", "ML-median": "ML median", "ML-mean": "ML mean", "ML-gated": "ML gated", "NESO+RollBias7d": "NESO+rolling bias", "abl:no_vint": "no vintage", "abl:no_adapt": "no adaptive mem.", "abl:base_only": "calendar only", "abl:base_adapt": "calendar+adaptive", "abl:no_real": "no realised info", "abl:no_mix": "no gen. mix", "abl:relative": "relative feats", "abl:main_w60": "recency-weighted"}
    xs, ys, ks = [], [], []
    for k, l in lab.items():
        key = f"{k}|replan|S24"
        if k in fe["overall"] and key in W: xs.append(fe["overall"][k]["mae"]); ys.append(W[key]["oracle_frac"][0]); ks.append(l)
    a.scatter(xs, ys, s=22, color=C["ml"], zorder=3)
    show = {"NESO": (3, -7), "ML median": (-4, 5), "calendar only": (3, -7), "NESO+rolling bias": (-62, 4), "no realised info": (3, -8), "recency-weighted": (3, 3), "no vintage": (3, -8), "calendar+adaptive": (3, 3)}
    for x_, y_, k in zip(xs, ys, ks):
        if k in show: a.annotate(k, (x_, y_), fontsize=5.8, xytext=show[k], textcoords="offset points")
    a.text(0.02, 0.04, "unlabelled: other full-feature variants", transform=a.transAxes, fontsize=5.6, style="italic")
    rho = pd.Series(xs).corr(pd.Series(ys), method="spearman"); z = np.polyfit(xs, ys, 1); xx = np.linspace(min(xs), max(xs), 10); a.plot(xx, np.polyval(z, xx), color="k", lw=0.8, ls="--")
    a.set_xlabel("Forecast MAE (gCO$_2$/kWh, Jul 2023-Oct 2026)"); a.set_ylabel("Oracle share captured (re-plan, 12-h slack)"); a.set_title(f"Forecast skill vs decision value (Spearman $\\rho$={rho:.2f})", fontsize=7.5); panel(a, "a")
    labs = ["<=0.5", "0.5-1", "1-2", "2-4", "4-8", "8-12", ">12"]; sh = [x_["gpu_hour_share"] * 100 for x_ in wl]
    cols = [C["ml"]] * 6 + ["#BBBBBB"]; b.bar(range(7), sh, color=cols); b.set_xticks(range(7)); b.set_xticklabels(labs, fontsize=6.5); b.set_xlabel("Job duration (h)"); b.set_ylabel("Share of GPU-hours (%)")
    b.set_title("Only deferrable-length jobs are addressable", fontsize=7.5); b.text(2.3, 60, f"jobs $\\leq$12 h: {sum(sh[:6]):.0f}% of GPU-h\nfleet-level saving, 12 h slack:\nML re-plan {W['ML-median|replan|S24']['fleet_saving_pct']:.1f}% (oracle {W['Oracle|static|S24']['fleet_saving_pct']:.1f}%)", fontsize=6.3, va="top"); panel(b, "b")
    fig.tight_layout(); save(fig, "fig10_value_and_reach"); print("spearman", rho)
print("done", which)
