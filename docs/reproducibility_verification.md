# Reproducibility Verification

Verified 2026-05-05. All numbers are mIoU / ARI. Both routes run on the **exact same committed bundle dataset instances**.

## Kaggle artifacts

| Dataset | Kaggle URL | Samples | Croissant |
|---|---|---|---|
| ControlNet-stitched PTD | https://www.kaggle.com/datasets/architexanonymous/architexture-controlnet-ptd-1742 | 1,742 | `croissant/controlnet_ptd_1742_croissant.json` (PASS) |
| DeTexture ADE20K validation | https://www.kaggle.com/datasets/architexanonymous/architexture-detexture-ade20k-56 | 56 | `croissant/detexture_ade20k_56_croissant.json` (PASS) |

Note: Enable private link sharing in the Kaggle UI for reviewer access during the NeurIPS review period.


| Dataset | n | Proposal-space mIoU | Proposal-space ARI | Feature-clustering mIoU | Feature-clustering ARI |
|---------|--:|----:|----:|----:|----:|
| RWTD | 256 | 0.4611 | 0.6966 | 0.8395 | 0.7261 |
| STLD | 182 (covered) | 0.7195 | 0.7791 | 0.7522 | 0.6176 |
| ControlNet bridge | 1742 | 0.6803 | 0.6039 | 0.8424 | 0.7314 |
| DeTexture ADE20K | 56 | 0.5008 | 0.3675 | 0.7435 | 0.5532 |

## Source artifacts

| Dataset | Route | Artifact |
|---------|-------|---------|
| RWTD | Proposal-space | `proposal_repro/verified_results.json` |
| STLD | Proposal-space | `proposal_repro/verified_results.json` |
| ControlNet bridge | Proposal-space | `proposal_repro/verified_results.json` |
| DeTexture ADE20K | Proposal-space | `proposal_repro/verified_results.json` |
| RWTD | Feature-clustering | `outputs/repro_notebook/20260503_130943/feature_clustering/table_1/rwtd_eval/summary.json` |
| STLD | Feature-clustering | `outputs/repro_notebook/20260503_130943/feature_clustering/table_1/stld_eval/summary.json` |
| ControlNet bridge | Feature-clustering | `outputs/bundle_controlnet_fc_eval/summary.json` |
| DeTexture ADE20K | Feature-clustering | `outputs/bundle_detexture_fc_eval/summary.json` |

## Reproduction commands

### Proposal-space (all four datasets)

```bash
python proposal_repro/verify_results.py
```

### Feature-clustering — RWTD and STLD

```bash
python scripts/feature_clustering/repro_table_1.py \
  --output-root outputs/repro_notebook/<run_id>/feature_clustering/table_1 \
  --rwtd-root datasets/RWTD \
  --stld-root datasets/STLD \
  --cstd-root datasets/CSTD
```

### Feature-clustering — ControlNet bridge (bundle, no download)

```bash
python -m scripts.feature_clustering.main eval-cstd-binary \
  --dataset-root ArchiTexture_NeurIPS_ED_submission_20260502/proposal-space-route/data/synthetic_texture_perlin_stitched_recovered/synthetic_texture_perlin_stitched \
  --variant feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2 \
  --device cuda \
  --failure-policy skip \
  --output-dir outputs/bundle_controlnet_fc_eval
```

### Feature-clustering — DeTexture ADE20K (bundle, no download)

```bash
python scripts/eval_bundle_detexture_fc.py \
  --benchmark-root ArchiTexture_NeurIPS_ED_submission_20260502/proposal-space-route/experiments/detexture_ade20k_eval_20260317/benchmarks/detexture_validation_refined \
  --variant feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2 \
  --device cuda \
  --failure-policy skip \
  --output-dir outputs/bundle_detexture_fc_eval
```
