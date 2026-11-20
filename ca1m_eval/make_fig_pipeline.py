#!/usr/bin/env python3
"""Illustrated pipeline figure (paper Fig. 1), built from one real CA-1M frame.

    .venv/bin/python ca1m_eval/make_fig_pipeline.py   ->  results/report/paper/fig_pipeline.pdf
"""

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "boxer"))
from loaders.ca_loader import CALoader  # noqa: E402
from utils.file_io import load_bb2d_csv, read_obb_csv  # noqa: E402
from utils.image import draw_bb3s  # noqa: E402

SEQ, FRAME = "ca1m-val-42897552", 125
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "report", "paper")
os.makedirs(OUT, exist_ok=True)
INK, INK2 = "#0b0b0b", "#52514e"
BLUE, ORANGE, GREEN, YELLOW, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5})


def footprints(obbs):
    b, T, out = obbs.bb3_object, obbs.T_world_object, []
    for i in range(len(obbs)):
        pts = torch.tensor([[x, y, 0.5 * (b[i, 4] + b[i, 5])] for x, y in
                            [(b[i, 0], b[i, 2]), (b[i, 1], b[i, 2]), (b[i, 1], b[i, 3]), (b[i, 0], b[i, 3])]])
        out.append((T[i] * pts)[:, :2].numpy())
    return out


def load_assets():
    loader = CALoader(SEQ, start_frame=1, skip_frames=1, max_frames=99999, resize=None)
    d = loader.load(FRAME)
    tag = loader.image_tags[FRAME]
    img = (d["img0"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    H, W = img.shape[:2]
    data_dir = loader.data_dir
    lidar = np.array(Image.open(os.path.join(data_dir, tag + ".wide", "depth.png"))) / 1000.0
    da3 = np.array(Image.open(os.path.join(RES, "depth", "da3metric", SEQ, tag + ".png"))) / 1000.0

    # 2D boxes from OWLv2 (stored at the 960x960 detector resolution)
    det = load_bb2d_csv(os.path.join(RES, "ca1m", "owl_rgbd", SEQ, "owl_2dbbs.csv"))[int(tag)]
    img2d = img.copy()
    for (x1, y1, x2, y2) in det["bb2d"]:
        p1 = (int(x1 * W / det["img_width"]), int(y1 * H / det["img_height"]))
        p2 = (int(x2 * W / det["img_width"]), int(y2 * H / det["img_height"]))
        cv2.rectangle(img2d, p1, p2, (235, 104, 52), 3)

    # 3D boxes for this frame, drawn back onto the image
    pred = read_obb_csv(os.path.join(RES, "ca1m", "owl_rgbd", SEQ, "boxer_3dbbs.csv"))[int(tag)]
    img3d = np.ascontiguousarray(img.copy())
    draw_bb3s(img3d, d["T_world_rig0"], d["cam0"], pred, colors=[(237, 161, 0)] * len(pred), thickness=3)

    # whole-room map: fused boxes vs true boxes, from above
    fused = torch.cat(list(read_obb_csv(
        os.path.join(RES, "ca1m", "owl_rgbd", SEQ, "boxer_3dbbs_fused_nms.csv")).values()))
    from eval_map import load_gt  # noqa: E402  (ca1m_eval on path below)
    gt, frames = load_gt(os.path.join(ROOT, "boxer", "sample_data"), SEQ, 1, 1, 99999)
    seen = torch.unique(torch.cat(list(frames.values())))
    return img, lidar, da3, img2d, img3d, footprints(fused), footprints(gt[seen])


def draw_pose_icon(ax):
    """Camera with x/y/z axes (each labelled with its rotation) and gravity."""
    ax.set_xlim(-1.45, 1.55)
    ax.set_ylim(-1.05, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((-0.28, -0.17), 0.42, 0.3, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc="#e8e7e3", ec=INK2, lw=0.7, zorder=3))
    ax.add_patch(Polygon([[0.14, -0.1], [0.34, -0.2], [0.34, 0.16], [0.14, 0.06]], fc="#e8e7e3", ec=INK2,
                         lw=0.7, zorder=3))
    o = np.array([-0.07, -0.02])
    for v, c, lab, off in [(np.array([0.95, -0.42]), "#d03b3b", "x (roll)", (0.05, -0.17)),
                           (np.array([-0.8, -0.55]), "#0ca30c", "y (pitch)", (-0.05, -0.17)),
                           (np.array([0.0, 0.92]), BLUE, "z (yaw)", (0.0, 0.14))]:
        ax.add_patch(FancyArrowPatch(o, o + v, arrowstyle="-|>", mutation_scale=6, color=c, lw=1.1, zorder=2))
        ax.add_patch(Arc(o + 0.62 * v, 0.34, 0.15, angle=np.degrees(np.arctan2(v[1], v[0])) + 90,
                         theta1=25, theta2=315, color=c, lw=0.8))
        ax.text(*(o + v + np.array(off)), lab, color=c, fontsize=5.6, ha="center", va="center")
    ax.add_patch(FancyArrowPatch((1.35, 1.05), (1.35, 0.15), arrowstyle="-|>", mutation_scale=6, color=INK2, lw=1))
    ax.text(1.35, -0.02, "gravity", fontsize=5.4, color=INK2, ha="center", va="center")


def main():
    img, lidar, da3, img2d, img3d, fused_fp, gt_fp = load_assets()
    FW, FH = 7.16, 2.7
    fig = plt.figure(figsize=(FW, FH))
    R = FW / FH

    def add_img(x, y, w, im, cmap=None, vmax=None, edge=None, lw=1.4):
        h = w * im.shape[0] / im.shape[1] * R
        ax = fig.add_axes([x, y, w, h])
        ax.imshow(im, cmap=cmap, vmin=0 if vmax else None, vmax=vmax, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(edge or "#bdbcb7")
            sp.set_linewidth(lw if edge else 0.6)
        return h

    ov = fig.add_axes([0, 0, 1, 1], zorder=10)
    ov.set_xlim(0, 1)
    ov.set_ylim(0, 1)
    ov.axis("off")

    def txt(x, y, s, **kw):
        kw.setdefault("fontsize", 5.8)
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        kw.setdefault("color", INK)
        ov.text(x, y, s, **kw)

    def head(x, y, s, color=INK2):
        txt(x, y, s, fontsize=6.4, weight="bold", color=color)

    def arrow(p0, p1, color=INK2, lw=0.9):
        ov.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw,
                                     shrinkA=0, shrinkB=0))

    def badge(x, y, letter, color):
        ov.add_patch(matplotlib.patches.Ellipse((x, y), 0.05 / R, 0.05, fc=color, ec="white",
                                                lw=0.8, zorder=20))
        ov.text(x, y - 0.002, letter, fontsize=5.6, color="white", weight="bold", ha="center", va="center",
                zorder=21)

    def panel(x, y, w, h, color):
        ov.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.012",
                                    fc=color + "1f", ec=color, lw=1.0, zorder=0))

    TOP = 0.56  # bottom of the main image row
    # ---------- column 1: inputs
    head(0.085, 0.955, "Inputs (each frame)")
    h = add_img(0.005, TOP, 0.16, img)
    txt(0.085, TOP - 0.03, "RGB image")
    draw_pose_icon(fig.add_axes([0.0, 0.255, 0.17, 0.235]))
    txt(0.085, 0.245, "camera pose + gravity")
    add_img(0.005, 0.045, 0.077, lidar, cmap="turbo", vmax=3.5)
    add_img(0.088, 0.045, 0.077, da3, cmap="turbo", vmax=3.5, edge=GREEN)
    txt(0.0435, 0.02, "LiDAR depth")
    txt(0.1265, 0.02, "DA3 depth", color=GREEN, weight="bold")
    badge(0.163, 0.205, "A", GREEN)
    badge(0.163, 0.47, "C", BLUE)

    # ---------- column 2: 2D detector
    head(0.272, 0.955, "1. 2D detector", ORANGE)
    add_img(0.197, TOP, 0.15, img2d, edge=ORANGE, lw=1.0)
    txt(0.272, TOP - 0.03, "OWLv2: 2D boxes + names")
    arrow((0.166, TOP + h / 2), (0.197, TOP + h / 2))

    # ---------- column 3: BoxerNet
    panel(0.38, 0.045, 0.2, 0.87, YELLOW)
    head(0.48, 0.955, "2. BoxerNet: 2D to 3D", "#a87300")
    # BoxerNet resizes to 960x960 and uses 16x16-px patches -> 60x60 grid.
    # Image grid drawn every 4th patch line (64 px at 960) so it stays visible.
    sq = cv2.resize(img, (960, 960), interpolation=cv2.INTER_AREA)
    sq[::64, :] = 255
    sq[:, ::64] = 255
    gimg = cv2.resize(sq, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    bw = 0.088
    add_img(0.392, 0.63, bw, gimg)
    # exact per-patch mean depth: LiDAR resized like the loader (nearest) to 960x960, 16x16 blocks
    dsq = cv2.resize(lidar.astype(np.float32), (960, 960), interpolation=cv2.INTER_NEAREST)
    blocks = dsq.reshape(60, 16, 60, 16)
    cnt = (blocks > 0).sum(axis=(1, 3))
    pd = np.where(cnt > 0, blocks.sum(axis=(1, 3)) / np.maximum(cnt, 1), np.nan)
    pd = cv2.resize(pd, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_NEAREST)
    add_img(0.488, 0.63, bw, pd, cmap="turbo", vmax=3.5)
    txt(0.436, 0.6, "image patches")
    txt(0.532, 0.6, "mean depth")
    txt(0.48, 0.355,
        "each patch is described by:\n"
        "- its appearance (DINOv3)\n"
        "- its viewing direction (from pose)\n"
        "- its mean depth (optional)\n\n"
        "each 2D box looks at these patches\n"
        "and predicts one 3D box\n"
        "(centre, size, yaw) + confidence", linespacing=1.4)
    arrow((0.347, TOP + h / 2), (0.38, TOP + h / 2))          # 2D boxes -> BoxerNet
    arrow((0.166, 0.37), (0.38, 0.37), color=BLUE, lw=0.8)    # pose
    arrow((0.166, 0.125), (0.38, 0.125), color=GREEN, lw=1.1)  # depth

    # ---------- column 4: output A + fusion
    head(0.68, 0.955, "Output A: each frame")
    add_img(0.605, TOP, 0.15, img3d, edge=YELLOW, lw=1.0)
    txt(0.68, TOP - 0.03, "3D boxes")
    arrow((0.58, TOP + h / 2), (0.605, TOP + h / 2))
    panel(0.605, 0.32, 0.15, 0.135, PURPLE)
    txt(0.68, 0.387, "3. Fusion: merge boxes of\none object across frames", fontsize=5.9)
    badge(0.752, 0.45, "B", PURPLE)
    arrow((0.68, TOP - 0.06), (0.68, 0.455))

    # legend of our changes
    lx, ly = 0.605, 0.045
    panel(lx, ly, 0.15, 0.235, "#9a9994")
    txt(lx + 0.075, 0.25, "Our changes", fontsize=6.1, weight="bold")
    for k, (letter, col, line) in enumerate([("A", GREEN, "depth from RGB (done)"),
                                             ("B", PURPLE, "multi-view refinement"),
                                             ("C", BLUE, "estimated pose, gravity")]):
        yy = 0.195 - k * 0.055
        badge(lx + 0.016, yy, letter, col)
        txt(lx + 0.031, yy, line, ha="left", fontsize=5.2)

    # ---------- column 5: output B
    head(0.885, 0.955, "Output B: whole room")
    mx = fig.add_axes([0.785, 0.12, 0.2, 0.78])
    for fp in gt_fp:
        mx.add_patch(Polygon(fp, closed=True, fill=False, ec="#9a9994", lw=0.6, ls="--"))
    for fp in fused_fp:
        mx.add_patch(Polygon(fp, closed=True, fill=False, ec=PURPLE, lw=0.9))
    allp = np.concatenate(gt_fp + fused_fp)
    lo, hi = np.percentile(allp, 1, 0) - 0.2, np.percentile(allp, 99, 0) + 0.2
    mx.set_xlim(lo[0], hi[0])
    mx.set_ylim(lo[1], hi[1])
    mx.set_aspect("equal")
    mx.set_xticks([])
    mx.set_yticks([])
    for sp in mx.spines.values():
        sp.set_edgecolor(PURPLE)
        sp.set_linewidth(1.0)
    txt(0.885, 0.06, "room seen from above\nsolid: ours, dashed: true", fontsize=5.6)
    fig.canvas.draw()
    arrow((0.755, 0.387), (mx.get_position().x0, 0.387))

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_pipeline.{ext}"), dpi=300)
    print(f"wrote {OUT}/fig_pipeline.pdf")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
