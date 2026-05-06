# proposal_repro

This directory is the **reviewer-safe live smoke path** for the ArchiTexture proposal-space route.

It is NOT the full paper-scale artifact. See `AUDIT_paper_contract.md` for the distinction.

---

## What this package does

1. Generates deterministic texture-partition composites as synthetic training data (no RWTD or STLD labels used during training).
2. Trains a lightweight random-forest proposal selector on those composites.
3. Evaluates the selector on held-out RWTD and STLD images using image-derived proposal banks.
4. Writes a compact `proposal_repro_table.csv` with rows `RWTD`, `STLD` and columns `MIOU`, `ARI`.

**Important:** The numbers produced here are NOT the paper headline numbers. The paper numbers come from the full artifact-backed path using frozen SAM2.1 prompt-mask banks and the SwinB/ConvNext PTD encoders (see `AUDIT_paper_contract.md` §2).

---

## Prerequisites

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Datasets

You must supply:

- `datasets/RWTD/` — RWTD Kaust256 benchmark (images + GT labels). Available from the RWTD dataset authors.
- `datasets/STLD/` — STLD benchmark (images + GT). Available from the STLD dataset authors.

The script will exit with a clear error if either root is missing or empty.

### GPU

Handcrafted descriptor mode (default) runs on CPU. PTD/SwinB descriptor modes require a CUDA GPU. The smoke path uses handcrafted descriptors by default.

---

## Usage

```bash
python proposal_repro/run_proposal_repro.py \
  --output-root outputs/repro_run_$(date +%Y%m%d_%H%M%S)/proposal_repro \
  --rwtd-root datasets/RWTD \
  --stld-root datasets/STLD
```

Or from the reproducibility notebook:

```bash
jupyter notebook notebooks/reproducibility.ipynb
```

Navigate to the "Proposal repro" section and run cell 6.

---

## Output files

| File | Description |
|---|---|
| `proposal_repro_table.csv` | Lead table: rows `RWTD`, `STLD`; columns `MIOU`, `ARI` |
| `summary.json` | Run configuration and aggregate results |
| `training_candidates.csv` | Synthetic training proposal candidates and targets |
| `inference_metrics.csv` | Per-sample held-out inference metrics |
| `visuals/rwtd/` | Input / proposal-union / GT / prediction panels for RWTD samples |
| `visuals/stld/` | Same for STLD samples |

---

## Paper headline numbers

For the paper headline numbers, see:

- `ArchiTexture_NeurIPS_ED_submission_20260502/rwtd_miner_github_repo/paper_repro/source_data/main_results/rwtd_architexture_full256_official.json` — RWTD full-256: mIoU=0.4611, ARI=0.6966
- `ArchiTexture_NeurIPS_ED_submission_20260502/rwtd_miner_github_repo/paper_repro/source_data/main_results/rwtd_architexture_common253_official.json` — RWTD common-253: mIoU=0.4645, ARI=0.7013
- `ArchiTexture_NeurIPS_ED_submission_20260502/rwtd_miner_github_repo/paper_repro/source_data/main_results/stld_architexture_summary.json` — STLD all-200: mIoU=0.6705, ARI=0.7249; common-182: mIoU=0.7195, ARI=0.7791

These were produced by the full artifact-backed pipeline documented in `results/EXPERIMENT_LEDGER.md`.

---

## Known issues

- `run_proposal_repro.py` must be present in this directory for the notebook live section to work.
  If it is missing, the smoke path cannot run (see `AUDIT_status.md`).
- Unit test `test_consolidator.py::TestConsolidator::test_empty_input` fails in the submission
  package (empty-input fallback returns a non-zero mask instead of zero). This is a bug in the
  consolidator's default-mask behavior under an empty proposal list.

---

## Relationship to submission package

The consolidation code used by the live repro is the same library as
`ArchiTexture_NeurIPS_ED_submission_20260502/proposal-space-route/proposal_space_route/`.
The smoke path uses only handcrafted descriptors and a random forest, not the heavyweight
SwinB/ConvNext checkpoints, to allow a fast reviewer-accessible run without large GPU assets.
