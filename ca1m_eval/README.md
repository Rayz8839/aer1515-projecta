# CA-1M evaluation (Boxer baseline reproduction)

Boxer doesn't release its evaluation code. This folder reimplements its CA-1M protocol
(paper Sec. 5 and Supp. B/C) so we can reproduce its numbers and then score our own variants.

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate          # from project root
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install numpy opencv-python tqdm dill pillow scipy
(cd boxer && bash scripts/download_ckpts.sh)

python ca1m_eval/download_ca1m_eval.py --accept-license   # 10 val seqs, ~5 GB -> boxer/sample_data/
bash   ca1m_eval/run_baselines.sh                         # all 4 configs -> results/ca1m/<cfg>/metrics.txt
python ca1m_eval/eval_map.py --pred_dir results/ca1m/owl_rgbd   # re-score any run
```

| config      | 2D boxes | depth        | paper per-frame | paper per-scene |
|-------------|----------|--------------|-----------------|-----------------|
| `owl_rgbd`  | OWLv2    | iPad LiDAR   | 0.204           | 0.204           |
| `owl_rgb`   | OWLv2    | none         | 0.081           | 0.081           |
| `gt2d_rgbd` | GT       | iPad LiDAR   | 0.412           | 0.434           |
| `gt2d_rgb`  | GT       | none         | 0.126           | 0.089           |

## Protocol
Class-agnostic mAP averaged over 3D IoU 0.05:0.05:0.5. Score = (2D score + 3D confidence) / 2.
No detection cap. Floors and walls are removed, as are objects with any side over 3 m, and sides
under 5 cm are expanded to 5 cm. Size buckets are by GT volume: small < 0.01 m³, medium 0.01–0.1 m³,
large > 0.1 m³.

The paper doesn't specify the following, so these are our choices:
- COCO 101-point AP, with detections pooled across all frames and scenes.
- Per-frame GT is the instances listed in that frame's `instances.json`. Per-scene GT is the union
  of those over the evaluated frames.
- Every frame is evaluated, from `start_n=1` (run_boxer's default).
- IoU is the analytic yaw-only `iou_exact7`. A few GT boxes are tilted; `--iou mc9` handles full
  rotation and changes mAP by about 0.01 or less.

## Scoring a new method
Write `results/<method>/<seq>/boxer_3dbbs.csv` (per-frame, `utils.file_io.ObbCsvWriter2`,
`time_ns` = CA-1M frame tag) and optionally `boxer_3dbbs_fused.csv` (per-scene), then run
`eval_map.py --pred_dir results/<method>`. For a new depth source, the easiest hook is to replace
`depth_m` in `boxer/loaders/ca_loader.py::_load_frame`.
