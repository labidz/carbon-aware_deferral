import matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({
 "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
 "legend.fontsize": 7, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
 "xtick.major.width": 0.7, "ytick.major.width": 0.7, "lines.linewidth": 1.3, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
 "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "pdf.fonttype": 42})
# colour-blind-safe palette (Okabe-Ito)
C = {"neso": "#444444", "ml": "#0072B2", "ml2": "#56B4E9", "orc": "#009E73", "warn": "#D55E00", "gold": "#E69F00", "pink": "#CC79A7", "yellow": "#F0E442", "dl": "#CC79A7"}
W1, W2 = 3.5, 7.2      # single / double column widths in inches (Elsevier: 90 mm / 190 mm)
def panel(ax, s): ax.text(-0.13, 1.13, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
def save(fig, name): fig.savefig(f"/home/claude/figs/{name}.png"); fig.savefig(f"/home/claude/figs/{name}.pdf"); plt.close(fig)
