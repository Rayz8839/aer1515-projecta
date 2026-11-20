#!/usr/bin/env bash
# Reproduce Boxer's CA-1M numbers (paper Tables 2 & 3) on the 10 val sequences.
#
#   config      2D boxes   depth (iPad LiDAR)   paper per-frame / per-scene mAP
#   owl_rgbd    OWLv2      yes                  0.204 / 0.204
#   owl_rgb     OWLv2      no  (--no_sdp)       0.081 / 0.081
#   gt2d_rgbd   GT         yes                  0.412 / 0.434
#   gt2d_rgb    GT         no                   0.126 / 0.089
#   owl_da3metric OWLv2    DA3METRIC-LARGE (from RGB)   (ours, no paper number)
#
# Usage:  bash ca1m_eval/run_baselines.sh [config ...]      (default: all four)
#   SKIP_N=5 bash ca1m_eval/run_baselines.sh owl_rgbd       (every 5th frame, quicker)
# owl_rgb reuses owl_rgbd's cached OWLv2 detections, so run owl_rgbd first.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
OUT="${OUT:-$ROOT/results/ca1m}"
SKIP_N="${SKIP_N:-1}"
START_N="${START_N:-1}"
SEQS=(45662921 45261179 47115543 45261143 45261615 42897545 45261133 42897552 45663113 42897521)
[[ $# -eq 0 ]] && set -- owl_rgbd owl_rgb gt2d_rgbd gt2d_rgb
CONFIGS=("$@")

cd "$ROOT/boxer"
for cfg in "${CONFIGS[@]}"; do
  case $cfg in
    owl_rgbd)  flags=() ;;
    owl_rgb)   flags=(--no_sdp --cache2d) ;;
    gt2d_rgbd) flags=(--gt2d) ;;
    gt2d_rgb)  flags=(--gt2d --no_sdp) ;;
    owl_da3metric) flags=(--cache2d) ;;   # depth = DA3METRIC-LARGE (ca1m_eval/da3_depth.py)
    *) echo "unknown config $cfg"; exit 1 ;;
  esac
  dir="$OUT/$cfg"
  mkdir -p "$dir"
  for id in "${SEQS[@]}"; do
    seq="ca1m-val-$id"
    if [[ -f "$dir/$seq/boxer_3dbbs_fused.csv" ]]; then echo "[$cfg] $seq done, skipping"; continue; fi
    if [[ $cfg == owl_rgb || $cfg == owl_da3metric ]]; then  # same OWLv2 boxes as owl_rgbd, only depth differs
      mkdir -p "$dir/$seq"
      cp "$OUT/owl_rgbd/$seq/owl_2dbbs.csv" "$dir/$seq/owl_2dbbs.csv"
    fi
    echo "[$cfg] $seq"
    depth_dir=""
    [[ $cfg == owl_da3metric ]] && depth_dir="$ROOT/results/depth/da3metric"
    BOXER_DEPTH_DIR="$depth_dir" "$PY" run_boxer.py --input "$seq" --skip_viz --fuse --skip_n "$SKIP_N" --start_n "$START_N" \
      --output_dir "$dir" "${flags[@]}" > "$dir/$seq.log" 2>&1 \
      || { echo "  failed, see $dir/$seq.log"; tail -5 "$dir/$seq.log"; exit 1; }
  done
  # Boxer's code leaves the paper's 3D NMS (0.6) off; re-fuse with it on -> boxer_3dbbs_fused_nms.csv
  "$PY" "$ROOT/ca1m_eval/refuse_nms.py" "$dir"
  "$PY" "$ROOT/ca1m_eval/eval_map.py" --pred_dir "$dir" --skip_n "$SKIP_N" --start_n "$START_N" \
    --json "$dir/metrics.json" | tee "$dir/metrics.txt"
done
