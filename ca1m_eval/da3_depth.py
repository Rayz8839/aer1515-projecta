#!/usr/bin/env python3
"""Predict metric depth for every CA-1M frame with Depth Anything 3 (DA3METRIC-LARGE).

Runs in its own env (.venv-da3). For each frame it:
  1. predicts depth from the single colour image (no other frames, no pose),
  2. converts to metres: depth_m = focal_px * output / 300, with the focal length
     taken at DA3's processing resolution (same as DA3's own apply_metric_scaling),
  3. saves it like CA-1M's LiDAR depth: uint16 PNG in millimetres,
     results/depth/da3metric/<seq>/<frame>.png,
  4. compares it with the iPad LiDAR depth (on pixels where LiDAR is valid).

    .venv-da3/bin/python ca1m_eval/da3_depth.py                 # all 10 sequences
    .venv-da3/bin/python ca1m_eval/da3_depth.py --seqs ca1m-val-42897552 --max_frames 5
"""

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "boxer", "sample_data")
SEQS = [
    "ca1m-val-45662921", "ca1m-val-45261179", "ca1m-val-47115543", "ca1m-val-45261143",
    "ca1m-val-45261615", "ca1m-val-42897545", "ca1m-val-45261133", "ca1m-val-42897552",
    "ca1m-val-45663113", "ca1m-val-42897521",
]


def depth_errors(pred, gt):
    """pred, gt in metres, same shape. Only pixels with valid LiDAR (gt > 0)."""
    m = gt > 0
    p, g = pred[m], gt[m]
    ratio = p / g
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),        # mean |error| / true depth
        "abs_err_m": float(np.mean(np.abs(p - g))),          # mean |error| in metres
        "scale": float(np.median(ratio)),                    # >1: predicts too far
        "delta1": float(np.mean(np.maximum(ratio, 1 / ratio) < 1.25)),  # share within 25%
        "n": int(m.sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", nargs="*", default=SEQS)
    ap.add_argument("--model", default="depth-anything/DA3METRIC-LARGE")
    ap.add_argument("--name", default="da3metric", help="output folder under results/depth/")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--process_res", type=int, default=504)
    args = ap.parse_args()

    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(args.model).to("cuda").eval()
    out_root = os.path.join(ROOT, "results", "depth", args.name)
    summary = {}
    for seq in args.seqs:
        data_dir = os.path.dirname(os.path.dirname(
            glob.glob(os.path.join(DATA, seq, "**/world.gt/instances.json"), recursive=True)[0]))
        tags = sorted({os.path.basename(d).split(".")[0] for d in glob.glob(os.path.join(data_dir, "*.wide"))})
        tags = tags[: args.max_frames] if args.max_frames else tags
        os.makedirs(os.path.join(out_root, seq), exist_ok=True)
        errs, t0 = [], time.time()
        for tag in tags:
            wide = os.path.join(data_dir, tag + ".wide")
            img = np.array(Image.open(os.path.join(wide, "image.png")).convert("RGB"))
            H, W = img.shape[:2]
            K = np.array(json.load(open(os.path.join(wide, "image", "K.json"))))
            with torch.no_grad():
                pred = model.inference([img], process_res=args.process_res)
            raw = pred.depth[0]                                   # (h, w) at processing res
            h, w = raw.shape
            # focal length rescaled to DA3's processing resolution
            f_proc = 0.5 * (K[0, 0] * w / W + K[1, 1] * h / H)
            depth_m = raw * f_proc / 300.0
            cv2.imwrite(os.path.join(out_root, seq, tag + ".png"),
                        np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16))

            lidar = np.array(Image.open(os.path.join(wide, "depth.png"))).astype(np.float32) / 1000.0
            d_at_lidar = cv2.resize(depth_m, (lidar.shape[1], lidar.shape[0]), interpolation=cv2.INTER_LINEAR)
            errs.append(depth_errors(d_at_lidar, lidar))
        dt = time.time() - t0
        agg = {k: float(np.mean([e[k] for e in errs])) for k in ("abs_rel", "abs_err_m", "scale", "delta1")}
        agg.update(frames=len(tags), seconds=round(dt, 1), proc_hw=[int(h), int(w)])
        summary[seq] = agg
        print(f"{seq}: {len(tags)} frames in {dt:.0f}s | AbsRel {agg['abs_rel']:.3f} | "
              f"mean err {agg['abs_err_m']:.3f} m | scale {agg['scale']:.3f} | within 25%: {agg['delta1']:.1%}",
              flush=True)

    allf = sum(v["frames"] for v in summary.values())
    tot = {k: sum(v[k] * v["frames"] for v in summary.values()) / allf
           for k in ("abs_rel", "abs_err_m", "scale", "delta1")}
    summary["ALL"] = {**tot, "frames": allf, "seconds": sum(v["seconds"] for v in summary.values())}
    print(f"ALL: {allf} frames | AbsRel {tot['abs_rel']:.3f} | mean err {tot['abs_err_m']:.3f} m | "
          f"scale {tot['scale']:.3f} | within 25%: {tot['delta1']:.1%}")
    if not args.max_frames:
        json.dump(summary, open(os.path.join(out_root, "depth_errors.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
