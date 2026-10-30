import numpy as np, pandas as pd, json, matplotlib.pyplot as plt, ukciv as U
from figstyle import *
ch = json.load(open("/home/claude/results/characterize.json"))
# ---------------- Fig 3: error structure ----------------
fig, axs = plt.subplots(2, 2, figsize=(W2, 5.0)); (a, b), (c, d) = axs
tab = pd.DataFrame(ch["error_by_lead"]); col = {"2023-24": C["ml"], "2026": C["warn"]}
for p in ["2023-24", "2026"]:
    t = tab[tab.period == p]; a.plot(t.lead_h, t.mae, "o-", color=col[p], ms=3, label=p); a.fill_between(t.lead_h, t.mae_lo, t.mae_hi, color=col[p], alpha=0.18, lw=0)
    b.plot(t.lead_h, t.bias, "s-", color=col[p], ms=3, label=p)
a.set_xlabel("Forecast lead (h)"); a.set_ylabel("MAE (gCO$_2$/kWh)"); a.set_title("Error grows with lead"); a.legend(); panel(a, "a")
b.axhline(0, color="k", lw=0.6); b.set_xlabel("Forecast lead (h)"); b.set_ylabel("Mean error $A-f$ (g/kWh)"); b.set_title("Systematic under-forecast at long leads"); panel(b, "b")
mb = pd.DataFrame(ch["monthly_bias_L12plus"]); mb = mb[mb.n > 5000]; colors = [col["2026"] if m.startswith("2026") else C["ml"] for m in mb.m]
c.bar(range(len(mb)), mb.bias, color=colors, width=0.7); c.set_xticks(range(len(mb))); c.set_xticklabels([m[2:] for m in mb.m], rotation=60, ha="right", fontsize=6.5); c.axhline(0, color="k", lw=0.6)
c.set_ylabel("Mean error, leads $\\geq$ 6 h (g/kWh)"); c.set_title("Bias is non-stationary across months"); panel(c, "c")
for p in ["2023-24", "2026"]:
    d.plot(np.arange(48) / 2, ch["diurnal_bias_L24"][p], color=col[p], label=p)
d.axhline(0, color="k", lw=0.6); d.set_xlabel("Target time of day (UTC, h)"); d.set_ylabel("Mean error at 12 h lead (g/kWh)"); d.set_title("Diurnal error profile"); d.legend(); panel(d, "d")
fig.tight_layout(); save(fig, "fig3_error_structure")
# ---------------- Fig 4: efficiency tests ----------------
X = np.load("/home/claude/data/g_X.npy", mmap_mode="r"); M = np.load("/home/claude/data/g_meta.npz", allow_pickle=True); names = list(M["names"])
s, L, f0, y = M["s"].astype(np.int64), M["L"].astype(np.int64), M["f0"], M["y"]; e = y - f0; fi = np.isfinite(y)
per = np.where(U.TS[np.clip(s + L, 0, U.T - 1)] < pd.Timestamp("2025-01-01"), "2023-24", "2026")
def binned(x, yv, nb=12):
    ok = np.isfinite(x) & np.isfinite(yv); q = np.quantile(x[ok], np.linspace(0, 1, nb + 1)); bi = np.clip(np.digitize(x[ok], q[1:-1]), 0, nb - 1)
    mx = np.array([x[ok][bi == i].mean() for i in range(nb)]); my = np.array([yv[ok][bi == i].mean() for i in range(nb)]); se = np.array([yv[ok][bi == i].std() / np.sqrt((bi == i).sum()) for i in range(nb)]); return mx, my, se
fig, axs = plt.subplots(2, 2, figsize=(W2, 5.0)); (a, b), (c, d) = axs
rev4 = np.asarray(X[:, names.index("rev4")]); es24 = np.asarray(X[:, names.index("e_same24")])
for lead, mk in [(12, "o"), (24, "s")]:
    m = fi & (L == lead) & (per == "2023-24"); mx, my, se = binned(rev4[m], e[m]); a.errorbar(mx, my, 1.96 * se, fmt=mk + "-", color=C["ml"] if lead == 12 else C["gold"], ms=3, capsize=1.5, label=f"lead {lead/2:g} h")
a.axhline(0, color="k", lw=0.6); a.axvline(0, color="k", lw=0.6); a.set_xlabel("Revision over last 2 h, $f_t-f_{t-4}$ (g/kWh)"); a.set_ylabel("Subsequent error $A-f_t$ (g/kWh)"); a.set_title("Revisions over-react (2023-24)"); a.legend(); panel(a, "a")
def forest(ax, d, keys, labels, title, yl):
    for i, (p, c_) in enumerate([("2023-24", C["ml"]), ("2026", C["warn"])]):
        xs, ys, es = [], [], []
        for j, k in enumerate(keys):
            r = d[k.format(p=p)]; xs.append(j + (-0.12 if i == 0 else 0.12)); ys.append(r["slope"]); es.append(1.96 * r["se"])
        ax.errorbar(xs, ys, es, fmt="o", color=c_, ms=3.5, capsize=2, label=p)
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels); ax.set_xlabel("Forecast lead (h)"); ax.set_ylabel(yl); ax.set_title(title); ax.legend()
rv = ch["revision_efficiency_rev4"]; ks = ["{p}|L=%.1fh" % h for h in [1.0, 3.0, 6.0, 12.0, 20.0, 30.0]]
forest(b, rv, ks, ["1", "3", "6", "12", "20", "30"], "Slope of error on 2-h revision", "OLS slope (day-clustered 95% CI)"); panel(b, "b")
for lead, mk, cc in [(12, "o", C["ml"]), (24, "s", C["gold"])]:
    m = fi & (L == lead) & (per == "2023-24"); mx, my, se = binned(es24[m], e[m]); c.errorbar(mx, my, 1.96 * se, fmt=mk + "-", color=cc, ms=3, capsize=1.5, label=f"lead {lead/2:g} h")
c.axhline(0, color="k", lw=0.6); c.axvline(0, color="k", lw=0.6); c.set_xlabel("Mean same-lead error of last 12 h (g/kWh)"); c.set_ylabel("Current error $A-f$ (g/kWh)"); c.set_title("Errors persist (2023-24)"); c.legend(); panel(c, "c")
pr = ch["error_persistence"]; ks2 = ["e_same24|{p}|L=%.1fh" % h for h in [1.0, 6.0, 12.0, 24.0, 40.0]]
forest(d, pr, ks2, ["1", "6", "12", "24", "40"], "Slope of error on recent same-lead error", "OLS slope (day-clustered 95% CI)"); panel(d, "d")
fig.tight_layout(); save(fig, "fig4_efficiency"); print("ok")
