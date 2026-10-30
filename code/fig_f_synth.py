import numpy as np, json, matplotlib.pyplot as plt
from figstyle import *
o = json.load(open("/home/claude/results/new_tables.json")); ch = json.load(open("/home/claude/results/characterize.json")); G = o["gb_synthetic"]
GN = [("fixed_uniform", C["gold"], "o", "uniform, fixed per target"), ("iid", C["pink"], "s", "i.i.d. Gaussian"), ("ar1", C["orc"], "^", "AR(1) Gaussian"), ("bootstrap", C["ml2"], "D", "resampled real errors"), ("berkson", C["warn"], "P", "calibrated (Berkson-type)")]
SS = ["2", "6", "12", "24", "48"]; hrs = [int(s) / 2 for s in SS]
fig, axs = plt.subplots(2, 2, figsize=(W2, 5.4)); (a, b), (c, d) = axs
for ax, pol, ttl, lab in [(a, "static", "Commit-once policy (GB)", "a"), (b, "replan", "Re-planning policy (GB)", "b")]:
    real = [np.mean([G[g][pol][S]["real"] for g, *_ in GN]) for S in SS]; ax.plot(hrs, real, "o-", color="k", lw=2.2, ms=4, label="real NESO forecast", zorder=5)
    for g, col, mk, l in GN:
        m = np.array([G[g][pol][S]["syn_mean"] for S in SS]); sd = np.array([G[g][pol][S]["syn_sd"] for S in SS])
        ax.plot(hrs, m, mk + "--", color=col, ms=3.5, lw=1.1, label=l); ax.fill_between(hrs, m - sd, m + sd, color=col, alpha=0.18, lw=0)
    ax.set_xscale("log"); ax.set_xticks(hrs); ax.set_xticklabels([f"{h:g}" for h in hrs]); ax.set_xlabel("Slack (h)"); ax.set_ylim(0.5, 0.97); ax.set_title(ttl, fontsize=8); panel(ax, lab)
a.set_ylabel("Share of oracle saving captured"); h_, l_ = b.get_legend_handles_labels()
sh = ch["shrinkage"]; ls_ = ["L=2.0h", "L=6.0h", "L=12.0h", "L=24.0h"]; x = np.arange(4)
c.bar(x, [sh[k]["corr_A_e"] for k in ls_], 0.5, color="k"); c.axhline(0, color=C["gold"], lw=2.5); c.set_xticks(x); c.set_xticklabels(["2", "6", "12", "24"]); c.set_xlabel("Lead (h)"); c.set_ylabel("corr(actual, forecast error)"); c.set_title("Real error is coupled to the truth;\nadditive noise is not (gold line)", fontsize=8); c.set_ylim(-0.03, 0.55); panel(c, "c")
xx = np.arange(5); w = 0.38; gb = [o["gb_synthetic_mad_pp"][g] for g, *_ in GN]; ex = [o["external_mad_pp"][g] for g, *_ in GN]
d.bar(xx - w / 2, gb, w, color=[col for _, col, *_ in GN], edgecolor="k", lw=0.5, label="Great Britain (held-out blocks, 6 seeds)"); d.bar(xx + w / 2, ex, w, color=[col for _, col, *_ in GN], hatch="///", edgecolor="k", lw=0.5, label="13 external grids (6 seeds)")
d.set_xticks(xx); d.set_xticklabels(["uniform", "i.i.d.", "AR(1)", "resampled", "calibrated"], fontsize=7); d.set_ylabel("Mean absolute deviation from real\ncapture (pp, all slacks)"); d.set_title("Fidelity of synthetic forecasts", fontsize=8); d.set_ylim(0, 34); d.legend(fontsize=6, loc="upper left"); panel(d, "d")
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.5, -0.005)); save(fig, "fig7_synthetic_gap"); print("ok")
