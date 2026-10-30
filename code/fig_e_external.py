import numpy as np, json, matplotlib.pyplot as plt
from figstyle import *
E = json.load(open("/home/claude/results/external.json")); ML = json.load(open("/home/claude/results/external_ml.json")); regs = list(E["regions"])
GN = [("fixed_uniform", C["gold"], "o", "uniform, fixed per target"), ("iid", C["pink"], "s", "i.i.d. Gaussian"), ("ar1", C["orc"], "^", "AR(1) Gaussian"), ("bootstrap", C["ml2"], "D", "resampled real errors"), ("berkson", C["warn"], "P", "calibrated (Berkson-type)")]
real = {r: E["regions"][r]["real"]["replan"]["12"] for r in regs}; order = sorted(regs, key=lambda r: real[r])
fig, axs = plt.subplots(1, 3, figsize=(W2, 3.9), gridspec_kw={"width_ratios": [1.5, 1.05, 1.05]}); a, b, c = axs
y = np.arange(len(order))
for g, col, mk, lab in GN:
    a.scatter([np.mean(E["regions"][r]["syn"][g]["replan"]["12"]) for r in order], y, s=16, color=col, marker=mk, label=lab, zorder=3, alpha=0.9)
a.scatter([real[r] for r in order], y, s=34, color="k", zorder=4, label="real forecast")
for yy, r in zip(y, order): a.plot([real[r], max(np.mean(E["regions"][r]["syn"][g]["replan"]["12"]) for g, *_ in GN)], [yy, yy], color="#bbbbbb", lw=0.6, zorder=1)
a.set_yticks(y); a.set_yticklabels([f"{r} ({E['regions'][r]['oracle_saving_pct_S12']:.0f}%)" for r in order], fontsize=6.5); a.set_xlabel("Oracle saving captured\n(re-planning, 12 h slack)"); a.set_title("Real vs synthetic forecasts, 13 grids\n(label: oracle saving)", fontsize=7.5); a.set_xlim(0.2, 1.02); panel(a, "a")
h_, l_ = a.get_legend_handles_labels()
rs = [(r, E["regions"][r]["revision_slope"]) for r in regs]; rs.sort(key=lambda x: x[1]["slope"]); yy = np.arange(len(rs))
for i, (r, v) in enumerate(rs):
    col = C["warn"] if (v["t"] < -1.96) else (C["ml"] if v["t"] > 1.96 else "#888888")
    b.errorbar(v["slope"], i, xerr=1.96 * v["se"], fmt="o", color=col, ms=3.5, capsize=2)
p = E["pooled_revision"]; b.errorbar(p["slope"], len(rs) + 0.6, xerr=1.96 * p["se"], fmt="D", color="k", ms=4, capsize=2); b.axvline(0, color="k", lw=0.7)
b.set_yticks(list(yy) + [len(rs) + 0.6]); b.set_yticklabels([r for r, _ in rs] + ["pooled"], fontsize=6.5); b.set_xlabel("Slope of error on 24-h revision"); b.set_title("Revision efficiency by grid\n(orange: over-react; blue: under-react)", fontsize=7.5); panel(b, "b")
d = [100 * (ML["decisions"][r]["full"]["replan"]["12"] - ML["decisions"][r]["raw"]["replan"]["12"]) for r in regs]; o2 = np.argsort(d)
c.barh(np.arange(13), [d[i] for i in o2], color=[C["ml"] if d[i] > 0 else C["warn"] for i in o2]); c.set_yticks(np.arange(13)); c.set_yticklabels([regs[i] for i in o2], fontsize=6.5); c.axvline(0, color="k", lw=0.7)
mae = [100 * (ML["accuracy"]["full"][r]["corrected"] / ML["accuracy"]["full"][r]["raw"] - 1) for r in regs]
c.set_xlabel("Change in oracle capture (pp)\n(post-processed vs raw, 12 h slack)"); c.set_title(f"Post-processing transfer: mean {np.mean(d):+.1f} pp;\nMAE improved in {sum(m < 0 for m in mae)}/13 grids", fontsize=7.5); panel(c, "c")
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.legend(h_, l_, loc="lower center", ncol=6, fontsize=6.3, bbox_to_anchor=(0.5, -0.005)); save(fig, "fig11_external"); print("ok")
