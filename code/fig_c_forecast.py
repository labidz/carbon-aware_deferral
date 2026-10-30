import numpy as np, pandas as pd, json, matplotlib.pyplot as plt
from figstyle import *
fe = json.load(open("/home/claude/results/forecast_eval.json")); pb = json.load(open("/home/claude/results/prob_main.json"))
blocks = ["2023-06","2023-07","2023-08","2023-09","2023-10","2023-11","2023-12","2024-01/02","2026-07","2026-08","2026-09","2026-10"]
# ---------------- Fig 5: forecast accuracy ----------------
fig = plt.figure(figsize=(W2, 5.2)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
a = fig.add_subplot(gs[0, :]); x = np.arange(len(blocks)); wd = 0.27
for i, (k, c, lab) in enumerate([("NESO", C["neso"], "NESO (operational)"), ("NESO+RollBias7d", C["gold"], "NESO + rolling 7-day bias"), ("ML-median", C["ml"], "LightGBM median (walk-forward)")]):
    a.bar(x + (i - 1) * wd, [fe["by_block"][k][b] for b in blocks], wd, color=c, label=lab)
a.set_xticks(x); a.set_xticklabels(blocks, fontsize=6.8, rotation=30, ha="right"); a.set_ylabel("MAE (gCO$_2$/kWh)"); a.set_title("Out-of-sample MAE by month (all leads; monthly retraining)"); a.legend(ncol=3, loc="upper right", fontsize=6.5); a.set_ylim(0, 34); panel(a, "a")
b = fig.add_subplot(gs[1, 0]); lb = ["0-2h", "2-6h", "6-12h", "12-24h", "24-48h"]; x2 = np.arange(5)
for i, (k, c, lab) in enumerate([("NESO", C["neso"], "NESO"), ("ML-median", C["ml"], "ML median"), ("ML-gated", C["ml2"], "ML gated")]):
    b.bar(x2 + (i - 1) * 0.27, [fe["by_lead"][k][l] for l in lb], 0.27, color=c, label=lab)
b.set_xticks(x2); b.set_xticklabels(lb); b.set_xlabel("Forecast lead"); b.set_ylabel("MAE (gCO$_2$/kWh)"); b.set_title("By lead (Jul 2023-Oct 2026)"); b.legend(fontsize=6.5); panel(b, "b")
c_ = fig.add_subplot(gs[1, 1])
order = [("abl:base_only", "calendar + f only"), ("abl:base_adapt", "+ adaptive error memory"), ("abl:base_vint", "+ vintage lags/revisions"), ("abl:base_real", "+ recent realised info"), ("abl:no_real", "full w/o realised info"), ("abl:no_vint", "full w/o vintage"), ("abl:no_adapt", "full w/o adaptive mem."), ("abl:no_mix", "full w/o generation mix"), ("ML-mean", "full (L2 mean)"), ("ML-median", "full (median)")]
order = [(k, l) for k, l in order if k in fe["overall"]]
ys = np.arange(len(order))[::-1]
for y_, (k, l) in zip(ys, order):
    r = fe["overall"][k]; col_ = C["ml"] if k.startswith("ML") else C["ml2"]
    c_.errorbar(r["rel_change_vs_NESO_pct"], y_, xerr=[[r["rel_change_vs_NESO_pct"] - r["rel_lo"]], [r["rel_hi"] - r["rel_change_vs_NESO_pct"]]], fmt="o", color=col_, ms=3.5, capsize=2)
c_.axvline(0, color="k", lw=0.7); c_.set_yticks(ys); c_.set_yticklabels([l for _, l in order], fontsize=6.5); c_.set_xlabel("MAE change vs NESO (%), 95% day-block CI"); c_.set_title("Feature-group ablation"); panel(c_, "c")
fig.tight_layout(); save(fig, "fig5_forecast_accuracy")
# ---------------- Fig 6: probabilistic ----------------
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.7)); a, b, c_ = axs
meth = [("NESO+empirical residual quantiles (static)", C["neso"], "NESO + static residual quantiles"), ("Raw quantile model (q10,q90)", C["gold"], "Raw quantile model"), ("Static CQR", C["warn"], "Static CQR"), ("Rolling CQR (14 d)", C["ml"], "Rolling CQR (14 d)")]
bl = [n for n in blocks[1:]]
for k, c, lab in meth:
    cov = [pb["by_block"][k].get(n, {}).get("coverage", np.nan) for n in bl]; a.plot(range(len(bl)), cov, "o-", ms=3, color=c, label=lab)
a.axhline(0.8, color="k", ls="--", lw=0.8); a.set_xticks(range(len(bl))); a.set_xticklabels([n[2:] for n in bl], rotation=60, ha="right", fontsize=6.3); a.set_ylabel("Empirical coverage"); a.set_title("Coverage by month (nominal 80%)"); a.set_ylim(0.4, 1.0); panel(a, "a"); a.legend(fontsize=5.8, loc="lower left")
x2 = np.arange(5)
for i, (k, c, lab) in enumerate(meth): b.bar(x2 + (i - 1.5) * 0.2, [pb["by_lead"][k][l]["coverage"] for l in lb], 0.2, color=c)
b.axhline(0.8, color="k", ls="--", lw=0.8); b.set_xticks(x2); b.set_xticklabels(lb, fontsize=6.5); b.set_ylim(0.4, 1.0); b.set_xlabel("Forecast lead"); b.set_title("Coverage by lead"); panel(b, "b")
for i, (k, c, lab) in enumerate(meth): c_.bar(x2 + (i - 1.5) * 0.2, [pb["by_lead"][k][l]["interval_score"] for l in lb], 0.2, color=c)
c_.set_xticks(x2); c_.set_xticklabels(lb, fontsize=6.5); c_.set_xlabel("Forecast lead"); c_.set_ylabel("Interval score (lower is better)"); c_.set_title("Sharpness + calibration"); panel(c_, "c")
fig.tight_layout(); save(fig, "fig6_probabilistic"); print("ok")
