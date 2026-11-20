# Interim findings (as of 2026-11-13)

All numbers: 10 CA-1M validation videos (the paper's), 6,148 scored frames, class-agnostic mAP over IoU 0.05–0.5.
"Gap closed" = (mAP − RGB-only) / (LiDAR − RGB-only), using our own RGB-only and LiDAR numbers.

## 1. Reproduction of Boxer (validates our scorer)
| Setup | per-frame ours / paper | per-scene ours / paper | found (recall @0.05) |
|---|---|---|---|
| OWLv2 + LiDAR depth   | 0.216 / 0.204 | 0.288 / 0.204 | 43.6% |
| OWLv2, RGB only       | 0.100 / 0.081 | 0.095 / 0.081 | 33.4% |
| gt2d + LiDAR depth    | 0.450 / 0.412 | 0.404 / 0.434 | 81.5% |
| gt2d, RGB only        | 0.157 / 0.126 | 0.116 / 0.089 | 54.0% |
- Per-frame: within 0.01–0.04 of the paper, same ranking. Validated.
- Per-scene: does not match the paper. Turning on the paper's 3D NMS (0.6) changed nothing (fusion already merges at IoU 0.3). Cause still unknown.

## 2. New: DA3METRIC-LARGE depth instead of LiDAR (single image, no pose)
| | per-frame | per-scene | found @0.05 | gap closed (per-frame / per-scene) |
|---|---|---|---|---|
| OWLv2 + DA3 metric depth | 0.101 | 0.108 | 34.6% | ~1% / ~7% |
Depth quality vs iPad LiDAR (all 6,167 frames): mean relative error 8.7% (16 cm), median scale 0.97 (≈3% too close), 91% of pixels within 25%. Runs at ~20 frames/s.
**Takeaway:** depth that looks good by usual depth metrics adds almost nothing. A few % depth error is a 10+ cm shift at 2–3 m, which is larger than many objects, so 3D IoU fails (see §4).

## 3. What limits accuracy
- **Missed objects (2D detector):** OWLv2 never boxes ~38% of objects. Even with perfect depth only 43.6% of objects are found. Small objects worst.
- **Duplicates:** OWLv2 removes duplicates only within the same label, so "drawer"+"dresser" on one object both survive -> false positives in class-agnostic scoring.
- **No depth -> fusion fails to merge:** gt2d RGB-only gives 1,225 scene boxes for 887 objects.
- **Size:** small objects (<0.01 m³, the largest group) score 0.027 RGB-only, 0.096 with LiDAR.

## 4. Why 3D IoU is harsh (for the report)
IoU(d) = (1−ε)^d / (2−(1−ε)^d) for a shift ε per axis. At ε = 10%: 1D 0.818, 2D 0.681, 3D 0.574.
Largest shift that still passes: IoU≥0.25 -> 26% (3D) vs 60% (1D); IoU≥0.5 -> 12.6% (3D) vs 33% (1D).
A 10 cm object must be placed within ~1.3 cm per axis to pass at IoU 0.5. Size error compounds too: 10% too big everywhere -> 3D IoU 0.75 (1D 0.91).

## 5. Things we changed vs released Boxer (must be stated)
- Scorer is our reimplementation (paper's is unreleased); unstated choices listed in ca1m_eval/README.md.
- Boxer code changes: one optional depth-source switch in loaders/ca_loader.py (BOXER_DEPTH_DIR; default unchanged).
- Per-scene uses NMS 0.6 as in the paper (no effect on numbers).

## 6. Proposed updates to the proposal
- Depth methods cut to 2–3 (prof): DA3METRIC (done), DA3NESTED-GIANT-LARGE-1.1 with ARKit poses (multi-view + pose-conditioned), + one non-DA3 single-image model (MoGe-2 or UniDepthV2). VGGT dropped / optional.
- Report depth metrics next to mAP; test whether *scale* is the limit (oracle-scale experiment, below).
- Add 2D-detector recall as a stated limitation / possible variable (newer open-vocab detectors, lower OWLv2 threshold, class-agnostic NMS).
- Keep camera-only setting (MASt3R-SLAM/VGGT poses, GeoCalib gravity) for week 8.
- Answer the prof's "why so low": harsh 3D IoU (§4), small objects, detector misses — not class confusion (scoring ignores labels).
- Per-scene numbers: report with caveat, or drop to per-frame as the main metric.

## Suggested next diagnostic (not run)
Oracle scale: multiply DA3 depth per frame by its median LiDAR/DA3 ratio, re-run (~10 min). If mAP jumps toward 0.216, per-frame scale error is the bottleneck (supports multi-view / pose-conditioned depth); if not, local shape errors are.
