"""Rebuild Figure 1 from results/primary_bootstrap.csv.

Usage:  python make_figure.py [results_dir] [output_png]
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = sys.argv[1] if len(sys.argv) > 1 else "results"
out = sys.argv[2] if len(sys.argv) > 2 else "figures/figure1.png"

b = pd.read_csv(f"{res}/primary_bootstrap.csv")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
styles = {"L2": ("o", "-", "white"), "L3": ("s", "--", "grey"), "L4": ("^", "-.", "black")}
for ax, split in zip(axes, ["ID", "OOD"]):
    for lv, (mk, ls, fc) in styles.items():
        d = b[(b.split == split) & (b.level == lv)].sort_values("w_conf")
        yerr = [d.observed_delta_pp - d.ci95_low_pp, d.ci95_high_pp - d.observed_delta_pp]
        ax.errorbar(d.w_conf, d.observed_delta_pp, yerr=yerr, marker=mk, linestyle=ls, capsize=3,
                    label=lv, color="k", markerfacecolor=fc, linewidth=1)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_title(split, fontsize=10)
    ax.set_xlabel("w_conf (confidence-coupling parameter)", fontsize=8)
    ax.set_xticks([0.0, 0.5, 1.1, 2.0])
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Intuition - confidence (pp)", fontsize=8)
axes[0].legend(frameon=False, fontsize=8, title="Cue level", title_fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=300)
print(f"wrote {out}")
