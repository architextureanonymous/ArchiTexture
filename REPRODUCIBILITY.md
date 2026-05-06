# Reproducibility

This is the terminal mirror of [notebooks/reproducibility.ipynb](notebooks/reproducibility.ipynb).

## Hosted datasets

| Dataset | Kaggle URL | n | Croissant |
|---|---|---|---|
| ControlNet-stitched PTD | https://www.kaggle.com/datasets/architexanonymous/architexture-controlnet-ptd-1742 | 1,742 | `croissant/controlnet_ptd_1742_croissant.json` |
| DeTexture ADE20K (validation-only) | https://www.kaggle.com/datasets/architexanonymous/architexture-detexture-ade20k-56 | 56 | `croissant/detexture_ade20k_56_croissant.json` |

For NeurIPS reviewer access, enable private link sharing on each dataset in the Kaggle UI. See [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) for full details.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install -e scripts/feature_clustering
jupyter notebook notebooks/reproducibility.ipynb
```

## Results

All four benchmarks are evaluated under both routes on the **exact same dataset instances** committed in the bundle. The numbers below are reproduced by running the commands in the sections that follow.

| Dataset | n | Proposal-space mIoU | Proposal-space ARI | Feature-clustering mIoU | Feature-clustering ARI |
| --- | ---: | ---: | ---: | ---: | ---: |
| RWTD | 256 | `0.4611` | `0.6966` | `0.8395` | `0.7261` |
| STLD | 182 (covered) | `0.7195` | `0.7791` | `0.7522` | `0.6176` |
| ControlNet bridge | 1742 | `0.6803` | `0.6039` | `0.8424` | `0.7314` |
| DeTexture ADE20K | 56 | `0.5008` | `0.3675` | `0.7435` | `0.5532` |

## Proposal-space route

Runs four official bundle evaluators on committed prediction masks. No training, no inference — masks are already committed.

```bash
python proposal_repro/verify_results.py
```

Writes `proposal_repro/verified_results.json` and prints a match/fail table. All four rows must show `OK`.

| Dataset | Evaluator | Prediction masks |
| --- | --- | --- |
| RWTD | `eval_upstream_texture_metrics.py` — per-GT-instance IoU/ARI | `proposal-space-route/reports/release_swinb_full256_audit/official_export/` |
| STLD | `eval_stld_direct.py` — direct-foreground IoU/ARI, ArchiTexture **covered-182** row | `proposal-space-route/experiments/khan_synthetic_gallery_20260312/eval/strict_ptd_learned/masks/` |
| ControlNet bridge | `eval_binary_partition_maskbank.py` — partition-invariant IoU/ARI | `proposal-space-route/experiments/perlin_controlnet_eval_20260312/full_0p3/stageA_0p3/strict_ptd_learned/masks/` |
| DeTexture ADE20K | `eval_two_mask_partition_maskbank.py` — two-mask partition-invariant IoU/ARI | `proposal-space-route/experiments/detexture_ade20k_eval_20260317/full_validation/stageA_0p3/strict_ptd_learned/masks/` |

All evaluator scripts are from `ArchiTexture_NeurIPS_ED_submission_20260502/proposal-space-route/scripts/`.

## Feature-clustering route

### RWTD and STLD

```bash
python scripts/feature_clustering/repro_table_1.py \
  --output-root outputs/repro_notebook/<run_id>/feature_clustering/table_1 \
  --rwtd-root datasets/RWTD \
  --stld-root datasets/STLD \
  --cstd-root datasets/CSTD
```

### ControlNet bridge (bundle data)

Uses the 1742-image benchmark committed in the bundle — no separate download.

```bash
python -m scripts.feature_clustering.main eval-cstd-binary \
  --dataset-root ArchiTexture_NeurIPS_ED_submission_20260502/proposal-space-route/data/synthetic_texture_perlin_stitched_recovered/synthetic_texture_perlin_stitched \
  --variant feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2 \
  --device cuda \
  --failure-policy skip \
  --output-dir outputs/bundle_controlnet_fc_eval
```

### DeTexture ADE20K (bundle data)

Uses the 56-image curated validation set committed in the bundle — no separate download.

```bash
python scripts/eval_bundle_detexture_fc.py \
  --benchmark-root ArchiTexture_NeurIPS_ED_submission_20260502/proposal-space-route/experiments/detexture_ade20k_eval_20260317/benchmarks/detexture_validation_refined \
  --variant feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2 \
  --device cuda \
  --failure-policy skip \
  --output-dir outputs/bundle_detexture_fc_eval
```

## Live proposal-space reproduction (optional)

Trains and evaluates a fresh proposal-space selector without using RWTD or STLD labels. Produces numbers close to but not identical to the committed readout (different random seed, image-derived proposal banks).

```bash
python proposal_repro/run_proposal_repro.py \
  --output-root outputs/repro_notebook/<run_id>/proposal_repro \
  --rwtd-root datasets/RWTD \
  --stld-root datasets/STLD
```

Output: `proposal_repro_table.csv` (rows `RWTD`, `STLD`; columns `MIOU`, `ARI`).

## Case gallery visualization

Computed at runtime from committed masks and local dataset images. Runs automatically after the verification cell in the notebook. No standalone terminal command — copy the gallery cell and run with `REPO_ROOT` set to the checkout root.

| Column | Source |
| --- | --- |
| Input | `datasets/RWTD/image/val/{id}.jpg` or `datasets/STLD/images/{id}.png` |
| GT | RWTD: `TextureSAM_upstream_20260303/Kaust256/labeles/{id}.png` · STLD: `datasets/STLD/labels/{id}.png` |
| Selector | `reports/final_round_learned_single_selector_{rwtd,stld}/…/mask…` |
| ArchiTexture | RWTD: `reports/release_swinb_full256_audit/official_export/mask_0_{id}.png` · STLD: `eval/strict_ptd_learned/masks/{id}.png` |

## Standalone t-sweep

Replays the ControlNet `t` sweep with the bundled checkpoint and texture pool.

```bash
python scripts/standalone_t_sweep_bundle/run_t_sweep.py \
  --output_path outputs/repro_notebook/<run_id>/standalone_t_sweep_smoke \
  --row_seeds 1,3,6,14 \
  --t_values 100,189,278,367,456,544,633,722,811,900 \
  --controlnet_path scripts/standalone_t_sweep_bundle/controlnet_rwtd_checkpoints/controlnet-500 \
  --texture_pool_dir scripts/standalone_t_sweep_bundle/dtd_selected \
  --device cuda \
  --dtype fp16 \
  --use_perlin
```

## Troubleshooting

- Missing `datasets/RWTD` or `datasets/STLD` — mount local drops before running feature-clustering.
- Missing `runwayml/stable-diffusion-v1-5` — let the notebook fetch it once with network access, or prepopulate the HuggingFace cache.
- Missing feature-clustering imports (`einops` etc.) — re-run the setup cell; it installs both the top-level repo and `scripts/feature_clustering`.
- Wrong kernel — restart Jupyter from the `.venv` environment.
