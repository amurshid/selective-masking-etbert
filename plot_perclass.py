#!/usr/bin/env python3
"""
plot_perclass.py  --  Per-class robustness: all 20 classes, ordered by baseline vulnerability.
Reads degradation_perclass.csv -> 20 subplots (5x4 grid), each showing F1 vs degradation
for Baseline vs Masking (p=0.7) vs Normalization (p=0.7), mask mode. Classes ordered by how
much baseline drops (0% -> 100%), so the most header-dependent classes come first.
Saves results/fig_perclass.pdf and .png.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results/degradation_perclass.csv")))
LEVELS = [0, 25, 50, 75, 100]
CLASS_NAMES = ["Shifu","Geodo","BitTorrent","Gmail","Neris","SMB","Nsis-ay","Cridex",
               "Zeus","Miuref","MySQL","Virut","Weibo","WorldOfWarcraft","Outlook",
               "Facetime","Tinba","FTP","Htbot","Skype"]
# (csv key, legend label, line style, marker)
SERIES = [("baseline",      "Baseline",        "-",  "o"),
          ("ours_p70",      "Masking",         ":",  "^"),
          ("normalize_p70", "Normalization",   "-.", "D")]

def f1series(model, cid):
    d = {int(r["degradation_pct"]): float(r["f1"])
         for r in rows if r["model"] == model and r["mode"] == "mask" and int(r["label"]) == cid}
    return [d.get(l, 0.0) for l in LEVELS]

# order classes by baseline's F1 drop (0% -> 100%), descending
drops = []
for cid in range(20):
    b = f1series("baseline", cid)
    drops.append((b[0] - b[4], cid, CLASS_NAMES[cid]))
drops.sort(reverse=True)

plt.rcParams.update({"font.size": 7, "font.family": "serif",
                     "axes.grid": True, "grid.linestyle": ":",
                     "grid.linewidth": 0.4, "grid.alpha": 0.4})

fig, axes = plt.subplots(4, 5, figsize=(9.0, 7.0))
axes = axes.flatten()
for idx, (drop, cid, name) in enumerate(drops):
    ax = axes[idx]
    for key, label, ls, mk in SERIES:
        ax.plot(LEVELS, f1series(key, cid), color="black", linestyle=ls, marker=mk,
                markersize=2.5, linewidth=0.9, markerfacecolor="white", label=label)
    ax.set_title(f"{name} ($\\Delta$={drop:.2f})", fontsize=7, fontweight="bold")
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([str(l) for l in LEVELS], fontsize=5)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=5)
    ax.set_xlabel("Degradation (%)", fontsize=5)
    ax.set_ylabel("F1", fontsize=5)
    if idx == 0:
        ax.legend(loc="lower left", fontsize=5, framealpha=0.9)

fig.suptitle("Per-class robustness under header masking: all 20 classes (ordered by baseline vulnerability)",
             fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig("results/fig_perclass.pdf", bbox_inches="tight")
fig.savefig("results/fig_perclass.png", dpi=300, bbox_inches="tight")
print("saved -> results/fig_perclass.pdf and results/fig_perclass.png")
