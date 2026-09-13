#!/usr/bin/env python3
"""
plot_degradation.py  --  the headline IEEE figure with no color, single-column.
Reads results/degradation_results.csv -> three vertically-stacked macro-F1-vs-degradation
panels (a: mask/absent, b: random/wrong, c: zero/normalized), four models each.
Vertical stack so the figure fits one column. Saves results/fig_degradation.pdf and .png.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows   = list(csv.DictReader(open("results/degradation_results.csv")))
LEVELS = [0, 25, 50, 75, 100]
# (csv key, legend label, line style, marker) -- all black, distinguished by style + marker
MODELS = [("baseline",      "Baseline",                 "-",  "o"),
          ("ours_p50",      "Masking ($p$=0.5)",        "--", "s"),
          ("ours_p70",      "Masking ($p$=0.7)",        ":",  "^"),
          ("normalize_p70", "Normalization ($p$=0.7)",  "-.", "D")]
PANELS = [("mask",   "(a) Header masked (absent)"),
          ("random", "(b) Header randomized (wrong)"),
          ("zero",   "(c) Header zeroed (normalized)")]

def series(model, mode):
    d = {int(r["degradation_pct"]): float(r["macro_f1"])
         for r in rows if r["model"] == model and r["mode"] == mode}
    return [d[l] for l in LEVELS]

# tight-but-safe shared y-axis from the actual data
allvals = [v for k, _, _, _ in MODELS for m, _ in PANELS for v in series(k, m)]
ymin = max(0.0, min(allvals) - 0.05)

plt.rcParams.update({"font.size": 8, "font.family": "serif",
                     "axes.grid": True, "grid.linestyle": ":",
                     "grid.linewidth": 0.5, "grid.alpha": 0.5})

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15), sharex=True, sharey=True)  # wide: spans both columns
for ax, (mode, title) in zip(axes, PANELS):
    for key, label, ls, mk in MODELS:
        ax.plot(LEVELS, series(key, mode), color="black", linestyle=ls, marker=mk,
                markersize=4, linewidth=1.0, markerfacecolor="white", label=label)
    ax.set_title(title, fontsize=8)
    ax.set_xticks(LEVELS)
    ax.tick_params(labelsize=7)
    ax.set_ylim(ymin, 1.0)
axes[0].set_ylabel("Macro-F1", fontsize=8)
for ax in axes:
    ax.set_xlabel("Header degradation (%)", fontsize=8)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
           ncol=4, fontsize=7, frameon=False)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig("results/fig_degradation.pdf", bbox_inches="tight")
fig.savefig("results/fig_degradation.png", dpi=300, bbox_inches="tight")
print("saved -> results/fig_degradation.pdf and results/fig_degradation.png")
