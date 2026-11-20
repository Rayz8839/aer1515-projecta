#!/usr/bin/env python3
"""Column-width results figure for the paper (Fig. 2): AP vs IoU threshold, and mAP by object size.

    .venv/bin/python ca1m_eval/make_paper_figs.py   ->  results/report/paper/fig_results.pdf
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "ca1m")
OUT = os.path.join(ROOT, "results", "report", "paper")
os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
})
SETUPS = [  # key, label, colour, linestyle
    ("gt2d_rgbd", "true 2D + LiDAR", "#1baf7a", "--"),
    ("owl_rgbd", "OWLv2 + LiDAR", "#2a78d6", "-"),
    ("gt2d_rgb", "true 2D, no depth", "#eda100", "--"),
    ("owl_da3metric", "OWLv2 + DA3", "#6250d6", "-"),
    ("owl_rgb", "OWLv2, no depth", "#eb6834", "-"),
]
M = {k: json.load(open(os.path.join(RES, k, "metrics.json")))["per_frame"] for k, *_ in SETUPS}

fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.75), gridspec_kw={"width_ratios": [1.15, 1]})
for k, lab, col, ls in SETUPS:
    ap = M[k]["all"]["AP"]
    t = sorted(float(x) for x in ap)
    a.plot(t, [ap[str(x)] for x in t], color=col, ls=ls, lw=1.3, marker="o", ms=2, label=lab)
a.set_xlabel("3D IoU threshold")
a.set_ylabel("AP (per frame)")
a.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
a.set_ylim(0, 0.8)
a.set_title("(a)", fontsize=7, loc="left")
a.legend(fontsize=5.3, loc="upper right", handlelength=1.8, labelspacing=0.25)

sizes = ["small", "medium", "large"]
bars = [("owl_rgb", "no depth", "#eb6834"), ("owl_da3metric", "DA3", "#6250d6"), ("owl_rgbd", "LiDAR", "#2a78d6")]
w = 0.27
for j, (k, lab, col) in enumerate(bars):
    v = [M[k][s]["mAP"] for s in sizes]
    b.bar([i + (j - 1) * (w + 0.015) for i in range(3)], v, w, color=col, label=lab)
b.set_xticks(range(3), ["small\n<0.01 m³", "medium", "large\n>0.1 m³"], fontsize=6)
b.set_ylabel("mAP (per frame), OWLv2")
b.grid(axis="x", visible=False)
b.set_title("(b)", fontsize=7, loc="left")
b.legend(fontsize=5.5, loc="upper left", handlelength=1.0)
fig.tight_layout(pad=0.3, w_pad=0.8)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_results.{ext}"), dpi=300)
print("wrote", os.path.join(OUT, "fig_results.pdf"))
