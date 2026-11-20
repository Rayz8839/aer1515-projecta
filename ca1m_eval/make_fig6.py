#!/usr/bin/env python3
"""Fig. 6: Boxer's 3D boxes vs the true 3D boxes for one frame.

Top row: boxes drawn on the camera image. Bottom row: the same boxes seen from above
(true boxes in grey in every panel), where errors in distance become visible.

    python ca1m_eval/make_fig6.py [--seq ca1m-val-42897552] [--frame 125]
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "boxer"))
from loaders.ca_loader import CALoader  # noqa: E402
from utils.file_io import read_obb_csv  # noqa: E402
from utils.image import draw_bb3s  # noqa: E402

RES = os.path.join(ROOT, "results", "ca1m")
OUT = os.path.join(ROOT, "results", "report")
PANELS = [  # (source, title, colour)
    ("owl_rgbd", "LiDAR depth", "#2a78d6"),
    ("owl_da3metric", "DA3 depth (ours)", "#6250d6"),
    ("owl_rgb", "No depth", "#eb6834"),
    ("gt", "True 3D boxes", "#1baf7a"),
]


def rgb255(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def footprints(obbs):
    """Corners (N,4,2) of each box seen from above (boxes only rotate about vertical z)."""
    b = obbs.bb3_object
    T = obbs.T_world_object
    out = []
    for i in range(len(obbs)):
        xs = [b[i, 0], b[i, 1], b[i, 1], b[i, 0]]
        ys = [b[i, 2], b[i, 2], b[i, 3], b[i, 3]]
        pts = torch.tensor([[x, y, (b[i, 4] + b[i, 5]) / 2] for x, y in zip(xs, ys)])
        out.append((T[i] * pts)[:, :2].numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="ca1m-val-42897552")
    ap.add_argument("--frame", type=int, default=125, help="frame index, same as boxer_viz_<frame>.jpg")
    args = ap.parse_args()

    loader = CALoader(args.seq, start_frame=1, skip_frames=1, max_frames=99999, resize=None)
    d = loader.load(args.frame)
    img = (d["img0"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    cam, T_wr, t = d["cam0"], d["T_world_rig0"], int(d["time_ns0"])
    sets = {"gt": d["obbs"]}
    for c in ("owl_rgbd", "owl_da3metric", "owl_rgb"):
        sets[c] = read_obb_csv(os.path.join(RES, c, args.seq, "boxer_3dbbs.csv")).get(t)
        if sets[c] is None:
            sys.exit(f"no {c} boxes at frame {args.frame}")

    # camera position + viewing direction, seen from above
    T_wc = T_wr @ cam.T_camera_rig.inverse()
    cpos = T_wc.t.numpy().reshape(3)
    fwd = T_wc.R.numpy().reshape(3, 3)[:, 2]

    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.9), gridspec_kw={"height_ratios": [0.62, 1.0]})
    gt_fp = footprints(sets["gt"])
    allpts = np.concatenate(gt_fp + [fp for c in ("owl_rgbd", "owl_da3metric", "owl_rgb") for fp in footprints(sets[c])]
                            + [cpos[None, :2]])
    lo = np.minimum(np.percentile(allpts, 2, 0), cpos[:2]) - 0.3
    hi = np.maximum(np.percentile(allpts, 98, 0), cpos[:2]) + 0.3
    for k, (src, title, col) in enumerate(PANELS):
        o = sets[src]
        viz = np.ascontiguousarray(img.copy())
        draw_bb3s(viz, T_wr, cam, o, draw_label=True, colors=[rgb255(col)] * len(o),
                  thickness=2, text_sz=0.9)
        ax = axes[0, k]
        ax.imshow(viz)
        ax.set_title(f"{title} ({len(o)} boxes)", fontsize=6.8, loc="left")
        ax.axis("off")

        ax = axes[1, k]
        for fp in gt_fp:
            ax.add_patch(Polygon(fp, closed=True, fill=False, ec="#9a9994", lw=0.8, ls="--"))
        if src != "gt":
            for fp in footprints(o):
                ax.add_patch(Polygon(fp, closed=True, fill=False, ec=col, lw=1.4))
        else:
            for fp in gt_fp:
                ax.add_patch(Polygon(fp, closed=True, fill=False, ec=col, lw=1.4))
        ax.plot(*cpos[:2], marker="o", color="#0b0b0b", ms=4)
        ax.annotate("", xy=cpos[:2] + 0.6 * fwd[:2] / (np.linalg.norm(fwd[:2]) + 1e-9), xytext=cpos[:2],
                    arrowprops=dict(arrowstyle="-|>", color="#0b0b0b", lw=1))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.grid(color="#e4e3df", lw=0.5)
        if k == 0:
            ax.set_ylabel("metres, seen from above", fontsize=6.5)
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.6)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "paper", f"fig_qualitative.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}/fig6_pred_vs_true.png")


if __name__ == "__main__":
    main()
