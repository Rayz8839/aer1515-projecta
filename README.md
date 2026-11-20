# Camera-Only Metric 3D Object Boxes (AER1515)

We reproduce [Boxer](https://github.com/facebookresearch/boxer)'s CA-1M results (its evaluation code
is unreleased, so we reimplemented it), then replace the iPad LiDAR depth with monocular metric depth
from [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) to see how much of the
RGB vs RGB-D gap it closes.

## Layout

| path | contents |
|------|----------|
| `ca1m_eval/` | our code: data download, evaluation (mAP), DA3 depth, figure scripts. See [`ca1m_eval/README.md`](ca1m_eval/README.md) for the protocol. |
| `boxer/` | Boxer, git submodule (unmodified upstream + `patches/boxer_ca_loader.patch`) |
| `third_party/depth-anything-3/` | Depth Anything 3, git submodule |
| `patches/` | our changes to upstream code (Boxer: optional `BOXER_DEPTH_DIR` depth override) |
| `results/` | **sample only**: metrics for every config, one sequence (`ca1m-val-42897552`) of raw outputs, report figures, paper source |

## Setup

```bash
git clone --recurse-submodules <this repo> && cd <repo>
git -C boxer apply ../patches/boxer_ca_loader.patch

# Boxer env
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install numpy opencv-python tqdm dill pillow scipy matplotlib
(cd boxer && bash scripts/download_ckpts.sh)
deactivate

# Depth Anything 3 env (separate, conflicting deps)
uv venv .venv-da3 --python 3.12 && source .venv-da3/bin/activate
uv pip install -e third_party/depth-anything-3
deactivate
```

## Reproduce

```bash
.venv/bin/python ca1m_eval/download_ca1m_eval.py --accept-license   # 10 CA-1M val seqs, ~5 GB -> boxer/sample_data/
bash ca1m_eval/run_baselines.sh                                     # owl_rgbd owl_rgb gt2d_rgbd gt2d_rgb
.venv-da3/bin/python ca1m_eval/da3_depth.py                         # DA3 depth -> results/depth/da3metric/
bash ca1m_eval/run_baselines.sh owl_da3metric                       # Boxer with DA3 depth
.venv/bin/python ca1m_eval/make_report_figures.py                   # figures/tables -> results/report/
```

Each config writes `results/ca1m/<config>/metrics.txt`. Full runs (~870 MB of outputs) are git-ignored.

## Results (CA-1M val, 10 sequences, mAP over 3D IoU 0.05–0.5)

| config | 2D boxes | depth | per-frame | per-scene | paper (frame / scene) |
|--------|----------|-------|-----------|-----------|-----------------------|
| `owl_rgbd`      | OWLv2 | iPad LiDAR    | 0.216 | 0.288 | 0.204 / 0.204 |
| `owl_rgb`       | OWLv2 | none          | 0.100 | 0.095 | 0.081 / 0.081 |
| `owl_da3metric` | OWLv2 | DA3 (ours)    | 0.101 | 0.108 | – |
| `gt2d_rgbd`     | GT    | iPad LiDAR    | 0.450 | 0.404 | 0.412 / 0.434 |
| `gt2d_rgb`      | GT    | none          | 0.157 | 0.116 | 0.126 / 0.089 |

DA3 depth is within ~9% of LiDAR on average (`results/depth/da3metric/depth_errors.json`) but
recovers almost none of the lost accuracy; see `results/report/` for figures and analysis.
