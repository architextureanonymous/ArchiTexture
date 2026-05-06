# SAM-2 Coarse Feature Clustering Bundle

This directory is a standalone copy of the code needed to run the registered
cross-dataset coarse feature clustering experiment with a frozen SAM-2 backbone:

`feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2`

It is intentionally copyable as-is. The repository code is vendored under
`src/rwtd_sam3/`, and the local launcher is `main.py`.

## What This Bundle Covers

- RWTD
- STLD
- CAID
- DeTexture ADE20K
- CSTD
- GlaS

The experiment variant is the same across all of those dataset adapters.
If you want the pooled-feature PCA overlay used in the coarse-feature figure
script, add `--save-pooled-feature-pca-overlay` to any eval or predict command.
It writes an extra
`visuals/<sample>_pooled_coarse_features_pca_rgb_overlay.png` file beside the
standard prediction panel.

## Install

Use a Python 3.10+ environment.

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install "SAM-2 @ git+https://github.com/facebookresearch/sam2.git"
```

## Run

The root launcher mirrors the repository CLI.

RWTD:

```bash
python main.py eval-sam3-auto \
  --variant feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2
```

STLD or CAID:

```bash
python main.py eval-architexture-binary \
  --route stld \
  --variant feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2
```

DeTexture ADE20K:

```bash
python main.py eval-detexture-binary \
  --dataset-root /path/to/detexture \
  --variant feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2
```

CSTD:

```bash
python main.py eval-cstd-binary \
  --dataset-root /path/to/cstd \
  --variant feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2
```

GlaS:

```bash
python main.py eval-glas-binary \
  --dataset-root /path/to/glas \
  --variant feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2
```

## Outputs

The bundle uses the same artifact contract as the repository code. Each run
creates a dedicated output directory with files such as:

- `config.json`
- `experiment_terms.md`
- `summary.json`
- `summary.md`
- `fairness.md`
- `paper_row.json`
- `benchmark_manifest.csv`
- `protocol.json`
- `per_sample_metrics.csv`
- `per_sample_metrics.jsonl`
- `eval/failures.csv` when a run has failures

## Smoke Test

This bundle includes a small local test that checks:

- the SAM-2 coarse-only variant is registered for all supported datasets
- the CLI accepts the variant on each dataset adapter

Run it with:

```bash
python smoke_test.py
```

## Reproducibility Helpers

Use these two scripts when you want the notebook flow from the terminal:

Table 1, full RWTD/STLD evaluation:

```bash
python repro_table_1.py \
  --output-root outputs/repro_notebook/<run_id>/feature_clustering/table_1
```

Figure 2-style examples, computed live:

```bash
python repro_figure_2.py \
  --output-root outputs/repro_notebook/<run_id>/feature_clustering/figure_2
```

The table script writes `table_1.csv`, `table_1.md`, and a manifest. The
figure script writes one clean `*_figure2.png` per selected sample plus a CSV
and manifest. Both scripts compute the outputs live from the local datasets and
the shared `feature_cluster_coarse_to_fine_global_pooled_init_coarse_only_sam2`
variant.

## Runtime Notes

- The SAM-2 model weights are loaded from HuggingFace Hub (`facebook/sam2-hiera-small`)
  on first run. The same model checkpoint is used in the proposal-space route.
- Dataset roots are external inputs. The code is bundled; the data is not.
- The proof replay fixtures are bundled only for verification. They are a
  minimal snapshot of real repository outputs, not synthetic GT-based stand-ins.
