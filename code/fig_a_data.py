import numpy as np, pandas as pd, json, matplotlib.pyplot as plt, matplotlib.patches as mp, ukciv as U
from figstyle import *
# ---------- Fig 1: pipeline schematic ----------
fig, ax = plt.subplots(figsize=(W2, 2.9)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(-2, 38)
def box(x, y, w, h, t, col, fs=5.7):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0", fc=col, ec="#333", lw=0.7)); ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs, linespacing=1.25)
def arr(x0, y0, x1, y1): ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333", shrinkA=0, shrinkB=0))
box(0.5, 25.5, 22, 10, "Git-scraped archive:\nNESO forecast every 30 min\n(2023-24 and 2026)\n+ generation mix", "#E8F1FA")
box(0.5, 13.5, 22, 9, "13 external grids\n(CarbonCast, 2021):\ndaily vintages", "#E8F1FA")
box(0.5, 1.5, 22, 9, "Acme LLM-cluster job\ntraces (Seren, Kalos)\nCC BY 4.0", "#E8F1FA")
box(27, 24, 20, 11.5, "UK-CIV dataset\nvintage tensor\n(window x lead)\nQC, final actuals,\nsplits", "#FFF3D6"); arr(22.7, 30.5, 26.8, 30.5)
box(52, 27, 22, 9, "Leakage-free features:\nvintage lags, revisions,\nerror memory, gen. mix", "#E6F4EA"); arr(47.2, 31, 51.8, 31.5)
box(52, 15.5, 22, 9.5, "Walk-forward models:\nLightGBM (mean, q10/50/90)\nVintageNet (deep quantile)", "#E6F4EA"); arr(63, 27, 63, 25.2); arr(22.7, 18, 51.8, 19)
box(52, 4, 22, 9.5, "Rolling conformal\ncalibration (14 d),\n80% intervals", "#E6F4EA"); arr(63, 15.5, 63, 13.7)
box(78, 19, 21.5, 16, "Deferral simulator:\nASAP, commit-once,\nre-plan, hysteresis,\nrisk-aware, learned\nstart-now, oracle;\nfleet replay", "#FDE8E0"); arr(74.2, 20.5, 77.8, 25); arr(74.2, 9, 77.8, 21)
box(78, 3, 21.5, 11, "Outcomes: carbon saving,\n% of oracle, backfire rate,\n13-grid validation,\nsynthetic-noise gap,\noverhead", "#FDE8E0"); arr(88.7, 18.8, 88.7, 14.2)
ax.plot([11.5, 11.5, 88.7], [1.4, -0.8, -0.8], color="#333", lw=0.9); arr(88.7, -0.8, 88.7, 2.8)
ax.text(30, 0.1, "job durations, GPU counts, deadlines", fontsize=6, style="italic", color="#333")
save(fig, "fig1_pipeline")
# ---------- Fig 2: forecast evolution on a representative day ----------
d = np.load("/home/claude/data/ukciv/tensor.npz", allow_pickle=True); F, A, t0, LMIN = d["F"], d["A"], int(d["t0"]), int(d["LMIN"]); T = len(A)
ts = pd.to_datetime(t0 + 1800 * np.arange(T), unit="s"); dates = ts.normalize()
ok = np.isfinite(A) & np.isfinite(F[:, 24 - LMIN])
dm = pd.DataFrame({"date": dates[ok], "e": np.abs(A[ok] - F[ok, 24 - LMIN])}).groupby("date").e.agg(["mean", "size"]); dm = dm[dm["size"] >= 44]
cand = dm[(dm.index.year == 2023) & (dm.index.month.isin([9, 10, 11]))]
med = cand["mean"].median(); dd = (cand["mean"] - med).abs().idxmin()          # representative (median-error) autumn-2023 day
w = np.where(dates == dd)[0]; assert len(w) == 48
x = ts[w].hour + ts[w].minute / 60
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]; leads = [47.5, 24, 12, 6, 2, 0]; cols = plt.cm.Blues(np.linspace(0.35, 0.95, len(leads)))
for l, c in zip(leads, cols): ax.plot(x, F[w, int(l * 2) - LMIN], color=c, lw=1.0, label=(f"issued {l:g} h ahead" if l else "issued at window start"))
ax.plot(x, A[w], color="k", lw=1.8, label="final actual"); ax.set_xlabel("Hour of day (UTC)"); ax.set_ylabel("Carbon intensity (gCO$_2$/kWh)")
ax.set_title(f"Forecast vintages for {str(ts[w[0]].date())} (median-error day)"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.3); panel(ax, "a")
ax = axs[1]; LS = [0, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 95]; M = np.full((48, len(LS)), np.nan)
for j, l in enumerate(LS): M[:, j] = F[w, l - LMIN] - A[w]
im = ax.imshow(M.T, aspect="auto", cmap="RdBu_r", vmin=-60, vmax=60, extent=[0, 24, 49.5, -1], origin="upper"); ax.set_xlabel("Target hour (UTC)"); ax.set_ylabel("Lead (h)")
ax.set_title("Forecast error $f-A$ by lead"); cb = plt.colorbar(im, ax=ax, pad=0.02); cb.set_label("g/kWh", fontsize=7); ax.grid(False); panel(ax, "b")
save(fig, "fig2_vintages"); print("example day", ts[w[0]])
