#!/usr/bin/env python3
"""Class-agnostic 3D mAP on CA-1M, reimplementing Boxer's protocol (paper Sec. 5, Supp. B/C).

Protocol (from the paper; [?] marks choices the paper doesn't specify and we picked):
  * Class-agnostic: labels are discarded, all boxes are one "anything" class.
  * mAP = mean of AP over 3D IoU thresholds 0.05, 0.10, ..., 0.50 (Omni3D style).
  * AP: COCO 101-point interpolated precision; greedy matching in score order,
    each GT matched at most once, detections pooled over all frames/scenes  [?]
  * GT: CA-1M world boxes, floors/walls removed, boxes with any side > 3 m removed,
    sides < 5 cm expanded to 5 cm (preds expanded too).
  * per-frame GT = instances listed in that frame's <tag>.wide/instances.json   [?]
  * per-scene GT = union of per-frame GT over the evaluated frames               [?]
  * Size buckets on GT volume: small < 0.01 m^3 <= medium <= 0.1 m^3 < large.
    COCO-style: out-of-bucket GTs are "ignore", and unmatched preds whose
    volume is outside the bucket are dropped.
  * No cap on detections.

Usage:
    python ca1m_eval/eval_map.py --pred_dir output/owl_rgbd
    python ca1m_eval/eval_map.py --pred_dir output/owl_rgbd --skip_n 1 --start_n 1 --json out.json

`pred_dir` holds <seq>/boxer_3dbbs.csv (per frame) and, if present,
<seq>/boxer_3dbbs_fused.csv (per scene), i.e. what run_boxer.py writes. Any method that
writes CSVs in that format (ObbCsvWriter2) can be scored the same way.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOXER_ROOT = os.path.join(PROJECT_ROOT, "boxer")
sys.path.insert(0, BOXER_ROOT)

from loaders.base_loader import BaseLoader  # noqa: E402
from loaders.ca_loader import _load_world_obbs  # noqa: E402
from utils.file_io import read_obb_csv  # noqa: E402
from utils.tw.obb import iou_exact7, iou_mc9  # noqa: E402

IOU_THRESHOLDS = np.round(np.arange(0.05, 0.501, 0.05), 2)
REC_POINTS = np.linspace(0.0, 1.0, 101)
SIZE_BUCKETS = {"all": (0.0, np.inf), "small": (0.0, 0.01), "medium": (0.01, 0.1), "large": (0.1, np.inf)}
BOXER_SEQS = [
    "ca1m-val-45662921", "ca1m-val-45261179", "ca1m-val-47115543", "ca1m-val-45261143",
    "ca1m-val-45261615", "ca1m-val-42897545", "ca1m-val-45261133", "ca1m-val-42897552",
    "ca1m-val-45663113", "ca1m-val-42897521",
]


# --------------------------------------------------------------------------- GT
def find_data_dir(data_root, seq):
    hits = glob.glob(os.path.join(data_root, seq, "**/world.gt/instances.json"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"no CA-1M data for {seq} under {data_root}")
    return os.path.dirname(os.path.dirname(hits[0]))


def clean_obbs(obbs, structure_sem_ids=()):
    """Apply Boxer's CA-1M filtering (same calls as CALoader.load)."""
    if len(structure_sem_ids):
        obbs = BaseLoader.filter_obbs_by_sem_id(obbs, list(structure_sem_ids))
    obbs = BaseLoader.filter_obbs_large(obbs, 3.0)
    return BaseLoader.expand_obbs_min_dim(obbs, 0.05)


def load_gt(data_root, seq, start_n, skip_n, max_n):
    """Returns (world_obbs_clean, {time_ns: tensor of row indices into world_obbs_clean})."""
    data_dir = find_data_dir(data_root, seq)
    all_obbs, id2inst, sem_id_to_name = _load_world_obbs(data_dir)
    structure = BaseLoader.find_structure_sem_ids(sem_id_to_name)
    keep = ~torch.isin(all_obbs.sem_id.squeeze(-1), torch.tensor(structure, dtype=torch.long))
    sizes = all_obbs.bb3_object
    dims = torch.stack([sizes[:, 1] - sizes[:, 0], sizes[:, 3] - sizes[:, 2], sizes[:, 5] - sizes[:, 4]], -1)
    keep &= dims.max(-1).values <= 3.0
    old2new = -torch.ones(len(all_obbs), dtype=torch.long)
    old2new[keep] = torch.arange(int(keep.sum()))
    gt = BaseLoader.expand_obbs_min_dim(all_obbs[keep].clone(), 0.05)

    # Same frame selection as CALoader(start_frame, skip_frames, max_frames).
    tags = sorted({os.path.basename(d).split(".")[0] for d in glob.glob(os.path.join(data_dir, "*.wide"))})
    tags = tags[start_n:][::skip_n][:max_n]
    frames = {}
    for tag in tags:
        with open(os.path.join(data_dir, tag + ".wide", "instances.json")) as f:
            ids = [b["id"] for b in json.load(f) if b["id"] in id2inst]
        idx = old2new[torch.tensor([id2inst[i] for i in ids], dtype=torch.long)] if ids else torch.zeros(0, dtype=torch.long)
        frames[int(tag)] = idx[idx >= 0]
    return gt, frames


# ------------------------------------------------------------------- IoU / AP
IOU_MODE = "exact7"  # set from --iou


def iou_matrix(pred, gt):
    """exact7: analytic 7-DoF IoU (yaw only, like the paper's iou_mc7, but exact).
    mc9: sampled IoU with full rotation (a few CA-1M GT boxes are tilted up to ~13 deg)."""
    if len(pred) == 0 or len(gt) == 0:
        return torch.zeros(len(pred), len(gt))
    if IOU_MODE == "exact7":
        return iou_exact7(pred.float(), gt.float()).float().cpu()
    return iou_mc9(pred.float(), gt.float(), samp_per_dim=32).float().cpu()


def volumes(obbs):
    b = obbs.bb3_object
    return ((b[:, 1] - b[:, 0]) * (b[:, 3] - b[:, 2]) * (b[:, 5] - b[:, 4])).numpy()


def match_image(ious, scores, gt_vol, pred_vol, lo, hi, thr):
    """COCO-style greedy matching for one image. Returns (scores, tp, n_gt) after ignore rules."""
    gt_ignore = (gt_vol < lo) | (gt_vol > hi)
    # COCO puts non-ignored GT first so they're preferred when matching.
    gorder = np.argsort(gt_ignore, kind="stable")
    ious = ious[:, gorder] if ious.size else ious
    gt_ignore = gt_ignore[gorder]
    order = np.argsort(-scores, kind="stable")
    gt_used = np.zeros(len(gt_ignore), bool)
    tp = np.zeros(len(order), bool)
    det_ignore = np.zeros(len(order), bool)
    for k, d in enumerate(order):
        best, best_iou = -1, min(thr, 1 - 1e-10)
        for g in range(len(gt_ignore)):
            if gt_used[g]:
                continue
            if best > -1 and not gt_ignore[best] and gt_ignore[g]:
                break  # already matched a real GT; don't switch to ignored ones
            if ious[d, g] < best_iou:
                continue
            best, best_iou = g, ious[d, g]
        if best >= 0:
            gt_used[best] = True
            tp[k] = True
            det_ignore[k] = gt_ignore[best]
        else:
            v = pred_vol[d]
            det_ignore[k] = (v < lo) or (v > hi)
    keep = ~det_ignore
    return scores[order][keep], tp[keep], int((~gt_ignore).sum())


def average_precision(scores, tp, n_gt):
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    tp = tp[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, np.finfo(float).eps)
    for i in range(len(prec) - 1, 0, -1):  # monotone envelope
        prec[i - 1] = max(prec[i - 1], prec[i])
    idx = np.searchsorted(rec, REC_POINTS, side="left")
    q = np.array([prec[i] if i < len(prec) else 0.0 for i in idx])
    return float(q.mean())


def evaluate(images):
    """images: list of (ious[N,M] np, scores[N], gt_vol[M], pred_vol[N]). Returns {bucket: {thr: AP, 'mAP': x}}."""
    out = {}
    for bucket, (lo, hi) in SIZE_BUCKETS.items():
        aps, rec = {}, {}
        for thr in IOU_THRESHOLDS:
            S, T, N = [], [], 0
            for ious, sc, gv, pv in images:
                s, t, n = match_image(ious, sc, gv, pv, lo, hi, thr)
                S.append(s)
                T.append(t)
                N += n
            aps[float(thr)] = average_precision(np.concatenate(S) if S else np.zeros(0),
                                                np.concatenate(T) if T else np.zeros(0, bool), N)
            # recall = share of true objects matched by some prediction (any score)
            rec[float(thr)] = float(sum(int(t.sum()) for t in T) / N) if N else float("nan")
        n_pred = sum(len(s) for s in S)
        out[bucket] = {"mAP": float(np.nanmean(list(aps.values()))), "AP": aps, "recall": rec,
                       "n_gt": N, "n_pred": n_pred}
    return out


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True, help="dir with <seq>/boxer_3dbbs.csv")
    ap.add_argument("--data_root", default=os.path.join(BOXER_ROOT, "sample_data"))
    ap.add_argument("--seqs", nargs="*", default=BOXER_SEQS)
    ap.add_argument("--start_n", type=int, default=1, help="must match run_boxer.py --start_n (default 1)")
    ap.add_argument("--skip_n", type=int, default=1, help="must match run_boxer.py --skip_n")
    ap.add_argument("--max_n", type=int, default=99999, help="must match run_boxer.py --max_n")
    ap.add_argument("--name", default="boxer", help="CSV prefix (run_boxer --write_name)")
    ap.add_argument("--fused", default="fused_nms", help="per-scene file: <name>_3dbbs_<fused>.csv (fused_nms = paper setting, fused = Boxer code default)")
    ap.add_argument("--json", default=None, help="write full results here")
    ap.add_argument("--iou", choices=["exact7", "mc9"], default="exact7",
                    help="exact7 = yaw-only analytic (paper-consistent); mc9 = full-rotation sampled")
    args = ap.parse_args()
    global IOU_MODE
    IOU_MODE = args.iou

    frame_imgs, scene_imgs, missing = [], [], []
    n_frames_pred = 0
    for seq in args.seqs:
        gt, frames = load_gt(args.data_root, seq, args.start_n, args.skip_n, args.max_n)
        csv = os.path.join(args.pred_dir, seq, f"{args.name}_3dbbs.csv")
        if not os.path.exists(csv):
            missing.append(seq)
            continue
        preds = read_obb_csv(csv)
        n_frames_pred += len(preds)
        # Per frame. Frames with no predictions still contribute their GT (as misses).
        for t, gidx in frames.items():
            g = gt[gidx]
            p = preds.get(t)
            if p is None or len(p) == 0:
                if len(g):
                    frame_imgs.append((np.zeros((0, len(g))), np.zeros(0), volumes(g), np.zeros(0)))
                continue
            p = BaseLoader.expand_obbs_min_dim(p.clone(), 0.05)
            ious = iou_matrix(p, g).numpy()
            frame_imgs.append((ious, p.prob.squeeze(-1).numpy(), volumes(g), volumes(p)))
        # Per scene: fused boxes vs union of GT seen in the evaluated frames.
        fcsv = os.path.join(args.pred_dir, seq, f"{args.name}_3dbbs_{args.fused}.csv")
        if os.path.exists(fcsv):
            fused = read_obb_csv(fcsv)
            p = torch.cat(list(fused.values())) if fused else None
            seen = torch.unique(torch.cat(list(frames.values()))) if frames else torch.zeros(0, dtype=torch.long)
            g = gt[seen]
            if p is None or len(p) == 0:
                scene_imgs.append((np.zeros((0, len(g))), np.zeros(0), volumes(g), np.zeros(0)))
            else:
                p = BaseLoader.expand_obbs_min_dim(p.clone(), 0.05)
                ious = iou_matrix(p, g).numpy()
                scene_imgs.append((ious, p.prob.squeeze(-1).numpy(), volumes(g), volumes(p)))

    if missing:
        print(f"[warn] no predictions for {len(missing)} seqs: {missing}")
    res = {"pred_dir": args.pred_dir, "iou": args.iou, "seqs": [s for s in args.seqs if s not in missing],
           "per_frame": evaluate(frame_imgs) if frame_imgs else None,
           "per_scene": evaluate(scene_imgs) if scene_imgs else None}

    print(f"\n{args.pred_dir}  ({len(res['seqs'])} seqs, {len(frame_imgs)} GT/pred frames, {n_frames_pred} frames with preds)")
    print(f"{'':10s}{'all':>8s}{'small':>8s}{'medium':>8s}{'large':>8s}   AP@0.25")
    for k in ("per_frame", "per_scene"):
        r = res[k]
        if r is None:
            print(f"{k:10s}  (no {'fused ' if k == 'per_scene' else ''}CSVs)")
            continue
        row = "".join(f"{r[b]['mAP']:8.3f}" for b in SIZE_BUCKETS)
        print(f"{k:10s}{row}   {r['all']['AP'][0.25]:.3f}")
    for k in ("per_frame", "per_scene"):
        r = res[k]
        if r is not None:
            a = r["all"]
            print(f"{k}: {a['n_gt']:,} true objects, {a['n_pred']:,} predicted boxes; "
                  f"found (recall) at IoU 0.05: {a['recall'][0.05]:.1%}, at 0.25: {a['recall'][0.25]:.1%}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
