# Artifact Map

Maps paper claims to hosted artifacts and evaluation evidence.

## Main paper tables — RWTD and STLD

These use committed prediction masks in the bundle. No separate download needed.

| Claim | Route | Verification |
|---|---|---|
| RWTD full-256 mIoU=0.4611 / ARI=0.6966 | Proposal-space | `python proposal_repro/verify_results.py` |
| STLD covered-182 mIoU=0.7195 / ARI=0.7791 | Proposal-space | `python proposal_repro/verify_results.py` |
| RWTD FC mIoU=0.8395 / ARI=0.7261 | Feature-clustering | `outputs/repro_notebook/*/feature_clustering/table_1/rwtd_eval/summary.json` |
| STLD FC mIoU=0.7522 / ARI=0.6176 | Feature-clustering | `outputs/repro_notebook/*/feature_clustering/table_1/stld_eval/summary.json` |

## Appendix routes — ControlNet bridge (Table 15)

| Claim | Route | Kaggle dataset | Evidence |
|---|---|---|---|
| ControlNet mIoU=0.6803 / ARI=0.6039 | Proposal-space | [architexture-controlnet-ptd-1742](https://www.kaggle.com/datasets/architexanonymous/architexture-controlnet-ptd-1742) | `proposal_repro/verified_results.json` |
| ControlNet FC mIoU=0.8424 / ARI=0.7314 | Feature-clustering | Same dataset | `outputs/bundle_controlnet_fc_eval/summary.json` |

## Post-paper diagnostic — DeTexture ADE20K

Not in the main paper tables. Validation-split only (56 samples) to avoid ADE20K train-set leakage into TextureSAM baseline.

| Claim | Route | Kaggle dataset | Evidence |
|---|---|---|---|
| DeTexture mIoU=0.5008 / ARI=0.3675 | Proposal-space | [architexture-detexture-ade20k-56](https://www.kaggle.com/datasets/architexanonymous/architexture-detexture-ade20k-56) | `proposal_repro/verified_results.json` |
| DeTexture FC mIoU=0.7435 / ARI=0.5532 | Feature-clustering | Same dataset | `outputs/bundle_detexture_fc_eval/summary.json` |

## Croissant metadata

| Dataset | File | Validation |
|---|---|---|
| ControlNet PTD 1742 | `croissant/controlnet_ptd_1742_croissant.json` | PASS (no errors) |
| DeTexture ADE20K 56 | `croissant/detexture_ade20k_56_croissant.json` | PASS (no errors) |
