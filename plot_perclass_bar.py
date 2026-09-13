#!/usr/bin/env python3
"""
plot_perclass_bar.py  --  per-class bar chart with no color, DOUBLE-COLUMN
F1 at 100% header omission for Baseline, Masking (p=0.7), and Normalization (p=0.7),
all 20 classes, sorted by baseline vulnerability (largest collapse at top). Vertical bars across a full-width figure. Saves results/fig_perclass_bar.pdf and .png.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results/degradation_perclass.csv")))
CLASS_NAMES = ["Shifu","Geodo","BitTorrent","Gmail","Neris","SMB","Nsis-ay","Cridex",
               "Zeus","Miuref","MySQL","Virut","Weibo","WorldOfWarcraft","Outlook",
               "Facetime","Tinba","FTP","Htbot","Skype"]

def f1(model, cid, lvl):
    for r in rows:
        if (r["model"] == model and r["mode"] == "mask"
                and int(r["label"]) == cid and int(r["degradation_pct"]) == lvl):
            return float(r["f1"])
    return 0.0

# order classes by baseline F1 collapse under full omission (0% -> 100%), largest first
order = sorted(range(20), key=lambda c: f1("baseline", c, 0) - f1("baseline", c, 100), reverse=True)
names = [CLASS_NAMES[c] for c in order]
base  = [f1("baseline", c, 100)      for c in order]
mask  = [f1("ours_p70", c, 100)      for c in order]
norm  = [f1("normalize_p70", c, 100) for c in order]

plt.rcParams.update({"font.size": 8, "font.family": "serif"})
y = np.arange(20)
h = 0.27
fig, ax = plt.subplots(figsize=(7.16, 2.35))  # wide: spans both columns
ax.bar(y - h, base, width=h, color="white", edgecolor="black", hatch="////", linewidth=0.6, label="Baseline")
ax.bar(y,     mask, width=h, color="0.80",  edgecolor="black", linewidth=0.6, label="Masking ($p$=0.7)")
ax.bar(y + h, norm, width=h, color="0.45",  edgecolor="black", linewidth=0.6, label="Normalization ($p$=0.7)")
ax.set_xticks(y)
ax.set_xticklabels(names, fontsize=6.5, rotation=45, ha="right")   # most vulnerable class at the left
ax.set_ylabel("Macro-F1 at 100%\nheader omission", fontsize=7.5)
ax.set_ylim(0, 1.05)
ax.tick_params(axis="y", labelsize=7)
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig("results/fig_perclass_bar.pdf", bbox_inches="tight")
fig.savefig("results/fig_perclass_bar.png", dpi=300, bbox_inches="tight")
print("saved -> results/fig_perclass_bar.pdf and results/fig_perclass_bar.png")
