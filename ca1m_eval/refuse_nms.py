#!/usr/bin/env python3
"""Re-run Boxer's fusion with its 3D NMS step switched ON (paper: tau_nms = 0.6).

Boxer's released code has NMS but leaves it off (enable_nms=False) and run_boxer.py
has no flag for it. This reads the per-frame boxes Boxer already wrote and writes
<seq>/boxer_3dbbs_fused_nms.csv next to the original boxer_3dbbs_fused.csv.
All other fusion settings are Boxer's defaults.

    python ca1m_eval/refuse_nms.py results/ca1m/owl_rgbd [results/ca1m/owl_rgb ...]
"""

import glob
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "boxer"))

from utils.file_io import ObbCsvWriter2, read_obb_csv  # noqa: E402
from utils.fuse_3d_boxes import BoundingBox3DFuser  # noqa: E402
from utils.tw.tensor_utils import tensor2string, unpad_string  # noqa: E402


def refuse(csv_in, csv_out, nms_iou=0.6):
    timed = read_obb_csv(csv_in)
    if not timed:
        open(csv_out, "w").close()
        return 0
    obbs = torch.cat(list(timed.values()), dim=0)
    fuser = BoundingBox3DFuser(enable_nms=True, nms_iou_threshold=nms_iou)  # rest = defaults
    inst = fuser.fuse(obbs)
    writer = ObbCsvWriter2(csv_out)
    if inst:
        fused = torch.stack([i.obb for i in inst], dim=0)
        fused.set_prob((torch.round(fused.prob * 100) / 100).squeeze(-1), use_mask=False)
        names = {int(o.sem_id.item()): unpad_string(tensor2string(o.text.int())) for o in fused}
        writer.write(fused, timestamps_ns=0, sem_id_to_name=names)
    writer.close()
    return len(inst)


if __name__ == "__main__":
    import contextlib
    import io

    for d in sys.argv[1:]:
        for csv in sorted(glob.glob(os.path.join(d, "*", "boxer_3dbbs.csv"))):
            seq = os.path.basename(os.path.dirname(csv))
            out = csv.replace("boxer_3dbbs.csv", "boxer_3dbbs_fused_nms.csv")
            with contextlib.redirect_stdout(io.StringIO()):  # fuser is very chatty
                n = refuse(csv, out)
            print(f"{d.rstrip('/').split('/')[-1]:10s} {seq}: {n} objects after fusion + NMS")
