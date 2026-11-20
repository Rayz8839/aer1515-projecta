#!/usr/bin/env python3
"""Build report figures + tables from results/ca1m/*/metrics.json and the run logs.

    python ca1m_eval/make_report_figures.py      ->  results/report/
"""

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.image import imread  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "ca1m")
OUT = os.path.join(ROOT, "results", "report")
os.makedirs(OUT, exist_ok=True)

CONFIGS = ["owl_rgbd", "owl_rgb", "gt2d_rgbd", "gt2d_rgb"]
LABEL = {
    "owl_rgbd": "OWLv2 boxes\n+ depth",
    "owl_rgb": "OWLv2 boxes\nRGB only",
    "gt2d_rgbd": "True 2D boxes\n+ depth",
    "gt2d_rgb": "True 2D boxes\nRGB only",
}
PAPER = {  # Boxer paper, Table 2 (per-frame) and Table 3 (per-scene), CA-1M column
    "owl_rgbd": (0.204, 0.204),
    "owl_rgb": (0.081, 0.081),
    "gt2d_rgbd": (0.412, 0.434),
    "gt2d_rgb": (0.126, 0.089),
}
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 300,
    "legend.frameon": False,
})

M = {c: json.load(open(os.path.join(RES, c, "metrics.json"))) for c in CONFIGS}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def bar_labels(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=7, color=INK2)


# ---------------------------------------------------------------- Fig 1: pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(-2, 102)
    ax.set_ylim(-4, 48)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=color + "22", ec=color, lw=1.2))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.5,
                weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.2, sub, ha="center", va="top", fontsize=7, color=INK2,
                linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                     color=INK2, lw=1.0))

    # inputs
    box(0, 33, 20, 13, "Colour image", "one video frame", BLUE)
    box(0, 17, 20, 13, "Camera pose", "where the camera is,\nwhich way is down", BLUE)
    box(0, 1, 20, 13, "Depth (optional)", "iPad LiDAR distance\nper pixel", AQUA)
    # stage 1
    box(26, 33, 22, 13, "1. 2D detector", "OWLv2 draws 2D boxes\n+ a label and score", ORANGE)
    # stage 2
    box(26, 9, 22, 21, "2. Image features", "DINOv3 splits the image\ninto 16x16-pixel patches;\n"
        "each patch gets: what it\nlooks like, which direction\nit points, how far it is", ORANGE)
    # stage 3
    box(54, 15, 22, 24, "3. BoxerNet", "for each 2D box, looks at\nthe relevant patches and\n"
        "outputs one 3D box:\ncentre, size, rotation\nabout the vertical axis,\n+ a confidence", YELLOW)
    # stage 4
    box(81, 15, 19, 24, "4. Fusion", "same object seen in\nmany frames: merge\noverlapping 3D boxes\n"
        "into one per object", YELLOW)
    ax.text(65, 11,  "Output A:\n3D boxes in every frame", ha="center", va="top", fontsize=7.2, color=INK)
    ax.text(90.5, 11,  "Output B:\none box per object\nfor the whole room", ha="center", va="top",
            fontsize=7.2, color=INK)

    arrow(20, 39.5, 26, 39.5)          # image -> detector
    arrow(20, 37, 26, 26)              # image -> features
    arrow(20, 23.5, 26, 20)            # pose -> features
    arrow(20, 7.5, 26, 13)             # depth -> features
    arrow(48, 39.5, 57, 39.6)          # detector -> boxernet
    arrow(48, 20, 54, 24)              # features -> boxernet
    arrow(76, 27, 81, 27)          # boxernet -> fusion
    arrow(65, 15, 65, 11.5)
    arrow(90.5, 15, 90.5, 11.5)
    ax.text(50, -3.5, "Our project replaces the depth input (green) with depth estimated from the "
            "image alone, and later replaces the pose and the fusion step.",
            ha="center", fontsize=7, color=INK2, style="italic")
    save(fig, "fig1_pipeline")


# ---------------------------------------------------------------- Fig 2: ours vs paper
def fig_reproduction():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    x = range(len(CONFIGS))
    w = 0.38
    for k, (ax, key, title) in enumerate(zip(axes, ("per_frame", "per_scene"),
                                             ("Each frame scored separately",
                                              "Whole room (after fusion)"))):
        ours = [M[c][key]["all"]["mAP"] for c in CONFIGS]
        paper = [PAPER[c][k] for c in CONFIGS]
        b1 = ax.bar([i - w / 2 - 0.01 for i in x], paper, w, color="#86b6ef", label="Boxer paper")
        b2 = ax.bar([i + w / 2 + 0.01 for i in x], ours, w, color=BLUE, label="Our reproduction")
        bar_labels(ax, b1)
        bar_labels(ax, b2)
        ax.set_xticks(list(x), [LABEL[c] for c in CONFIGS], fontsize=7.5)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("mAP (higher is better)")
    axes[0].set_ylim(0, 0.52)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, "fig2_reproduction")


# ---------------------------------------------------------------- Fig 3: AP vs IoU threshold
def fig_ap_vs_iou():
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    colors = {"gt2d_rgbd": AQUA, "owl_rgbd": BLUE, "gt2d_rgb": YELLOW, "owl_rgb": ORANGE}
    for c in ["gt2d_rgbd", "owl_rgbd", "gt2d_rgb", "owl_rgb"]:
        ap = M[c]["per_frame"]["all"]["AP"]
        t = sorted(float(k) for k in ap)
        y = [ap[str(k)] if str(k) in ap else ap[k] for k in t]
        ax.plot(t, y, color=colors[c], lw=2, marker="o", ms=3.5, label=LABEL[c].replace("\n", " "))
    ax.set_xlabel("3D overlap (IoU) needed to count as correct")
    ax.set_ylabel("AP, per frame")
    ax.set_xticks([0.05, 0.15, 0.25, 0.35, 0.45])
    ax.set_ylim(0, None)
    ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    save(fig, "fig3_ap_vs_iou")


# ---------------------------------------------------------------- Fig 4: size breakdown
def fig_sizes():
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    sizes = ["small", "medium", "large"]
    names = {"small": "small\n< 0.01 m³", "medium": "medium\n0.01–0.1 m³", "large": "large\n> 0.1 m³"}
    w = 0.38
    for j, (c, col) in enumerate([("owl_rgb", ORANGE), ("owl_rgbd", BLUE)]):
        v = [M[c]["per_frame"][s]["mAP"] for s in sizes]
        b = ax.bar([i + (j - 0.5) * (w + 0.02) for i in range(3)], v, w, color=col,
                   label=LABEL[c].replace("\n", " "))
        bar_labels(ax, b)
    n = [M["owl_rgbd"]["per_frame"][s]["n_gt"] for s in sizes]
    ax.set_xticks(range(3), [f"{names[s]}\n(n={k:,})" for s, k in zip(sizes, n)], fontsize=7)
    ax.set_ylabel("mAP, per frame")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig4_object_size")


# ---------------------------------------------------------------- Fig 5: example frames
def fig_examples(seq="ca1m-val-42897552", frame=125):
    vdir = os.path.join(OUT, "viz")
    rows = [("owl_rgbd", "OWLv2 boxes + depth"), ("owl_rgb", "OWLv2 boxes, RGB only")]
    paths = [os.path.join(vdir, c, seq, "boxer_viz", f"boxer_viz_{frame:05d}.jpg") for c, _ in rows]
    if not all(os.path.exists(p) for p in paths):
        print("skip fig5 (render frames first, see README)")
        return
    imgs = [imread(p) for p in paths]
    H, W = imgs[0].shape[:2]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    panels = [(imgs[0][:, : W // 2], "Input: OWLv2 2D boxes"),
              (imgs[0][:, W // 2:], "3D boxes, with depth"),
              (imgs[1][:, W // 2:], "3D boxes, RGB only")]
    for ax, (im, t) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(t, fontsize=8.5, color=INK, loc="left")
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    save(fig, "fig5_example_frame")


# ---------------------------------------------------------------- tables
def tables():
    L = []
    L.append("TABLE A. Class-agnostic 3D mAP on CA-1M (10 validation sequences)\n")
    L.append(f"{'Configuration':26s}{'per-frame':>11s}{'paper':>8s}{'per-scene':>11s}{'paper':>8s}")
    for c in CONFIGS:
        L.append(f"{LABEL[c].replace(chr(10), ' '):26s}{M[c]['per_frame']['all']['mAP']:11.3f}"
                 f"{PAPER[c][0]:8.3f}{M[c]['per_scene']['all']['mAP']:11.3f}{PAPER[c][1]:8.3f}")
    L.append("\nTABLE B. Per-frame mAP by object size (GT volume)\n")
    L.append(f"{'Configuration':26s}{'all':>8s}{'small':>8s}{'medium':>8s}{'large':>8s}{'AP@0.25':>9s}")
    for c in CONFIGS:
        r = M[c]["per_frame"]
        L.append(f"{LABEL[c].replace(chr(10), ' '):26s}" + "".join(
            f"{r[s]['mAP']:8.3f}" for s in ("all", "small", "medium", "large"))
            + f"{r['all']['AP']['0.25']:9.3f}")
    n = M["owl_rgbd"]["per_frame"]
    L.append(f"\nGT objects counted per frame (summed over frames): small {n['small']['n_gt']:,}, "
             f"medium {n['medium']['n_gt']:,}, large {n['large']['n_gt']:,}")

    # runtimes from tqdm lines in the logs
    L.append("\nTABLE C. Runtime on one RTX 5080 (16 GB), all frames, every sequence\n")
    L.append(f"{'Sequence':20s}{'frames':>7s}" + "".join(f"{c:>11s}" for c in CONFIGS))
    tot = {c: 0 for c in CONFIGS}
    nf_tot = 0
    seqs = sorted({os.path.basename(p)[:-4] for p in glob.glob(os.path.join(RES, "owl_rgbd", "*.log"))})
    for s in seqs:
        row, nf = [], 0
        for c in CONFIGS:
            txt = open(os.path.join(RES, c, s + ".log")).read().replace("\r", "\n")
            m = re.findall(r"BoxerNet: 100%\|.*?\| (\d+)/\d+ \[(\d+):(\d+)<", txt)
            nf, mm, ss = map(int, m[-1])
            sec = mm * 60 + ss
            tot[c] += sec
            row.append(f"{sec:10d}s")
        nf_tot += nf
        L.append(f"{s:20s}{nf:7d}" + " ".join(row))
    L.append(f"{'TOTAL':20s}{nf_tot:7d}" + " ".join(f"{tot[c]:10d}s" for c in CONFIGS))
    L.append(f"{'frames / second':27s}" + " ".join(f"{nf_tot / tot[c]:10.1f} " for c in CONFIGS))
    L.append("(owl_rgb reuses the OWLv2 detections from owl_rgbd, so it skips the detector.)")
    txt = "\n".join(L)
    open(os.path.join(OUT, "tables.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    # fig_pipeline()
    # fig_reproduction()
    # fig_ap_vs_iou()
    # fig_sizes()
    fig_examples()
    # tables()
    print(f"\nwrote figures to {OUT}")
