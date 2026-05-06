# Data Access

## Hosted evaluation datasets

### ControlNet-stitched PTD Benchmark v1 (1,742 samples, ~780MB)

- **Kaggle:** https://www.kaggle.com/datasets/architexanonymous/architexture-controlnet-ptd-1742
- **Paper:** Appendix Table 15, Supporting Route E
- **Expected result:** ARCHITEXTURE Stage-A, all-1742, 1742/1742, invariant mIoU=0.6803, ARI=0.6039
- **Croissant:** `croissant/controlnet_ptd_1742_croissant.json` (validated, no errors)
- **License:** CC-BY 4.0

### DeTexture ADE20K Refined Validation v1 (56 samples)

- **Kaggle:** https://www.kaggle.com/datasets/architexanonymous/architexture-detexture-ade20k-56
- **Paper:** Post-paper diagnostic extension (not in main paper tables)
- **Expected result:** ARCHITEXTURE Stage-A, 56 samples, invariant mIoU=0.7435, ARI=0.5532
- **Croissant:** `croissant/detexture_ade20k_56_croissant.json` (validated, no errors)
- **License:** CC-BY-NC 4.0 (ADE20K-derived)
- **Why 56 only:** The full DeTexture gallery has 753 entries (697 training + 56 validation). The 697 training-prefixed crops are intentionally excluded because the TextureSAM public checkpoint was trained on ADE20K training images. Evaluating on those crops would contaminate the baseline comparison. Only the 56 validation-prefixed crops are used.

## What is not hosted

The full DeTexture ADE20K gallery (753 entries including 697 training-prefixed crops) is not shipped as an evaluation artifact because it would contaminate the TextureSAM baseline. It can be reconstructed from the public DeTexture pipeline at https://github.com/aviadcohz/detexture_ADE20K using the committed `gallery_subset.json`.

The original 4.6GB ControlNet generated pool is not required for any reported result and is intentionally excluded.

## Reviewer access

For NeurIPS reviewer access, enable private link sharing on each Kaggle dataset (Kaggle UI → dataset page → Settings → sharing). Copy the private preview URL and add it to the OpenReview dataset URL field.

## Reproduction

```bash
# Proposal-space route (all 4 datasets from committed masks — no download needed)
python proposal_repro/verify_results.py

# Feature-clustering — ControlNet bridge (uses Kaggle dataset or bundle copy)
python -m scripts.feature_clustering.main eval-cstd-binary \
  --dataset-root ArchiTexture_NeurIPS_ED_submission_20260502/proposal-space-route/data/synthetic_texture_perlin_stitched_recovered/synthetic_texture_perlin_stitched \
  --variant feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2 \
  --device cuda --failure-policy skip \
  --output-dir outputs/bundle_controlnet_fc_eval

# Feature-clustering — DeTexture ADE20K (uses Kaggle dataset or bundle copy)
python scripts/eval_bundle_detexture_fc.py \
  --benchmark-root ArchiTexture_NeurIPS_ED_submission_20260502/proposal-space-route/experiments/detexture_ade20k_eval_20260317/benchmarks/detexture_validation_refined \
  --variant feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2 \
  --device cuda --failure-policy skip \
  --output-dir outputs/bundle_detexture_fc_eval
```
