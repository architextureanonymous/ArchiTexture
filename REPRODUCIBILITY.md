# Reproducibility Notebook

This is the terminal mirror of [notebooks/reproducibility.ipynb](notebooks/reproducibility.ipynb).
The notebook is unified and minimal:

1. setup
2. ArchiTexture headline table
3. proposal-space result verification (computed from committed masks)
4. case gallery visualization (computed from committed masks + dataset images)
5. live proposal-space training reproduction (optional)
6. live feature-clustering reproduction
7. DeTexture ADE20K smoke evaluation (optional)
8. standalone t-sweep experiment

## Goal

Show the reviewer the main final numbers first, then run a live proposal-space train/inference smoke path before the heavier feature-clustering and t-sweep sections.

## Setup

Recommended terminal flow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install -e scripts/feature_clustering
jupyter notebook notebooks/reproducibility.ipynb
```

The notebook includes a runnable setup cell. It installs the same requirements into the active kernel, then installs the editable feature-clustering bundle, then checks the environment in place. It does not switch kernels.

## ArchiTexture Headline Table

The first notebook section runs the proposal-bank readout script into a fresh output root under `outputs/repro_notebook/<run_id>/architexture/`, then shows:

1. the compact two-row headline table
2. the generated visuals
3. the supporting tables

| Benchmark | Evaluator / subset | mIoU | ARI |
| --- | --- | ---: | ---: |
| RWTD | official invariant, full-256 | `0.4611` | `0.6966` |
| STLD | direct foreground, all-200 | `0.6705` | `0.7249` |

The helper behind that section is:

```bash
python scripts/repro/proposal_bank_readout.py \
  --output-root outputs/repro_notebook/<run_id>/architexture
```

It writes fresh tables, copies the paper figures into the run root, and emits a small manifest for the run.

## Proposal Repro

The next section is the reviewer-safe live proposal-space reproduction. Training does **not** use RWTD or STLD labels. The script first generates texture-partition composites using the same PTD-style synthetic partition machinery used by the proposal-space training code, trains a lightweight proposal selector on those generated samples, and then evaluates only on RWTD and STLD.

The first output is the requested table:

| Dataset | MIOU | ARI |
| --- | ---: | ---: |
| RWTD | produced by the run | produced by the run |
| STLD | produced by the run | produced by the run |

The notebook writes a fresh run under:

- `outputs/repro_notebook/<run_id>/proposal_repro`

The terminal command is:

```bash
python proposal_repro/run_proposal_repro.py   --output-root outputs/repro_notebook/<run_id>/proposal_repro   --rwtd-root datasets/RWTD   --stld-root datasets/STLD
```

The first file to inspect is:

- `proposal_repro_table.csv`: rows `RWTD`, `STLD`; columns `MIOU`, `ARI`

Supporting files:

- `summary.json`: selected run configuration and aggregate table values
- `training_candidates.csv`: generated-training proposal candidates and targets
- `inference_metrics.csv`: per-sample RWTD/STLD held-out inference metrics
- `visuals/*/*.png`: input, proposal union, GT, and prediction panels

The public reviewer package does not include the heavyweight frozen SAM prompt-bank exports used by the paper-scale artifact readout. This live repro therefore builds deterministic image-derived proposal banks for evaluation, while keeping all selector training on generated texture composites rather than RWTD/STLD. The paper-scale RWTD/STLD numbers above remain the retained full-run readout.

## Case Gallery Visualization

The notebook computes two gallery figures **at runtime** from committed masks and local dataset images — not from pre-shipped PNGs. The gallery cell runs automatically after the verification cell and requires no flags.

Each gallery shows four columns per image row:

| Column | Source |
| --- | --- |
| Input | `datasets/RWTD/image/val/{id}.jpg` or `datasets/STLD/images/{id}.png` |
| GT | RWTD: `TextureSAM_upstream_20260303/Kaust256/labeles/{id}.png` · STLD: `datasets/STLD/labels/{id}.png` |
| Selector | `reports/final_round_learned_single_selector_{rwtd,stld}/…/mask…` |
| ArchiTexture | RWTD: `reports/release_swinb_full256_audit/official_export/mask_0_{id}.png` · STLD: `eval/strict_ptd_learned/masks/{id}.png` |

The "Fragments", "Core", and "Oracle single" columns from the paper figure require the frozen SAM proposal banks (not distributed) and are omitted.

There is no standalone terminal command for this section — it uses the same Python environment as the notebook and reads directly from the committed submission artifacts. To reproduce it outside the notebook, copy the gallery cell source and run it as a plain script with `REPO_ROOT` set to the checkout root.

## DeTexture ADE20K (optional)

Set `RUN_DETEXTURE = True` in the DeTexture notebook cell to download and smoke-test the curated benchmark.

### Download

```bash
# via huggingface-cli (requires huggingface_hub installed)
huggingface-cli download anon-detexture-neurips-2026/ADE20k_Detecture \
  --repo-type dataset \
  --local-dir datasets/ADE20k_Detexture
```

Expected layout after download:

```
datasets/ADE20k_Detexture/
  assets/
    crops/       ← crop images
    masks/       ← {id}_mask_a.png, {id}_mask_b.png
```

### Smoke test (5 samples)

```bash
rwtd-sam3 eval-detexture-binary \
  --dataset-root datasets/ADE20k_Detexture \
  --variant feature_cluster_coarse_to_fine_global_pooled_init_coarse_only \
  --limit 5 \
  --save-visuals \
  --output-dir outputs/detexture_smoke \
  --device cuda \
  --failure-policy skip
```

Writes `per_sample_metrics.csv`, `summary.json`, and `visuals/` under `outputs/detexture_smoke/`.

### Full evaluation

```bash
rwtd-sam3 eval-detexture-binary \
  --dataset-root datasets/ADE20k_Detexture \
  --variant feature_cluster_coarse_to_fine_global_pooled_init_coarse_only \
  --save-visuals \
  --output-dir outputs/detexture_full \
  --device cuda
```

## Feature Clustering

The feature-clustering section runs by default and writes fresh outputs under:

- `outputs/repro_notebook/<run_id>/feature_clustering/table_1`
- `outputs/repro_notebook/<run_id>/feature_clustering/figure_2`

The commands are:

```bash
python scripts/feature_clustering/repro_table_1.py \
  --output-root outputs/repro_notebook/<run_id>/feature_clustering/table_1 \
  --rwtd-root datasets/RWTD \
  --stld-root datasets/STLD

python scripts/feature_clustering/repro_figure_2.py \
  --output-root outputs/repro_notebook/<run_id>/feature_clustering/figure_2 \
  --rwtd-root datasets/RWTD \
  --stld-root datasets/STLD \
  --examples-per-dataset 3
```

The notebook then displays the fresh `table_1.csv` summary and the generated Figure 2-style panels.

## Standalone t-Sweep

The final live section in the notebook replays the standalone ControlNet `t` sweep with the bundled checkpoint and texture pool.

It writes a fresh export tree under:

- `outputs/repro_notebook/<run_id>/standalone_t_sweep_smoke`

The notebook uses the same row seeds and `t` values as the standalone bundle demo:

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

The section renders the resulting 4-row by 11-column image grid inline, with one hard-stitched column and ten `t` samples per row.

## Troubleshooting

- Missing `datasets/RWTD` or `datasets/STLD`
  - Mount the local drops before running the feature-clustering section.
- Missing the base diffusion model `runwayml/stable-diffusion-v1-5`
  - Let the notebook fetch it once with network access, or prepopulate the Hugging Face cache before running the t-sweep section.
- If the active kernel is not the environment you want, restart Jupyter from the intended virtual environment.
- Missing proposal-bank artifact trees
  - The helper mirrors the paper readout from the local `paper/` tables and figures in this checkout, so it does not depend on the heavyweight final-round artifact tree.
- Missing feature-clustering imports such as `einops`
  - Re-run the setup cell. It installs both the top-level repo and the editable `scripts/feature_clustering` bundle.

## Notes

- The setup cell installs requirements into the active kernel. If you prefer a separate virtual environment, create `.venv` first and launch Jupyter from that environment.
- The readout section runs the helper first, then renders the headline table, visuals, and supporting tables from the fresh run root.
- All outputs are written under `outputs/repro_notebook/<run_id>/`. The notebook reads only from the fresh run tree it just created.
