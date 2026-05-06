# AUDIT_inventory.md — Proposal-Repro Directory

**Date:** 2026-05-03  
**Auditor:** automated artifact audit

---

## Status before this audit

`proposal_repro/` did **not exist** as a tracked or untracked git directory prior to this audit session. The directory was referenced by:

- `REPRODUCIBILITY.md` (line 71): `python proposal_repro/run_proposal_repro.py ...`
- `notebooks/reproducibility.ipynb` (cell 6 saved output): showed a prior run from `<REDACTED_PATH>`

The notebook cell 6 source is **empty** — only saved output from a prior manual run is present. The script that produced those outputs does not exist in the repository.

---

## Expected artifacts (from REPRODUCIBILITY.md and notebook)

The `proposal_repro/` package is the **reviewer-safe live smoke path**. It is documented to contain:

| Artifact | Description | Status |
|---|---|---|
| `run_proposal_repro.py` | Entry-point script | **MISSING** |
| `README.md` or equivalent | Usage instructions | **MISSING** |
| Any supporting modules | Helper code for the live repro | **MISSING** |

The entire directory was absent from the repository.

---

## Related artifacts found elsewhere in the repo

### Submission package code (proposal-space-route)

Location: `ArchiTexture_NeurIPS_ED_submission_20260502/proposal-space-route/`

| File | Status |
|---|---|
| `proposal_space_route/consolidator.py` | Present |
| `proposal_space_route/pipeline.py` | Present |
| `proposal_space_route/proposals.py` | Present |
| `proposal_space_route/metrics.py` | Present |
| `proposal_space_route/dtd_encoder.py` | Present |
| `proposal_space_route/ptd_encoder.py` | Present |
| `proposal_space_route/ptd_learned.py` | Present |
| `proposal_space_route/ptd_v3.py` | Present |
| `proposal_space_route/ptd_v4_set.py` | Present |
| `proposal_space_route/ptd_v6_coverage.py` | Present |
| `proposal_space_route/ptd_v8_partition.py` | Present |
| `proposal_space_route/features.py` | Present |
| `proposal_space_route/merge.py` | Present |
| `proposal_space_route/reranker.py` | Present |
| `proposal_space_route/cli.py` | Present |
| `pyproject.toml` | Present, anonymized (`name = "proposal-space-route"`) |
| `tests/test_consolidator.py` | Present — 1 FAILING test |
| `tests/test_descriptor_mode.py` | Present |
| `tests/test_mpcl.py` | Present |

### Submission package artifacts

Location: `ArchiTexture_NeurIPS_ED_submission_20260502/proposal-space-route/artifacts/`

| File | Description |
|---|---|
| `dtd_small_cnn.pt` | DTD descriptor CNN checkpoint |
| `ptd_encoder_swinb_pre_ring.pt` | RWTD SwinB PTD encoder |
| `ptd_learned_swinb_pre_ring.pkl` | RWTD learned consolidation bundle |
| `ptd_learned_swinb_pre_ring_metrics.json` | Bundle metrics |
| `ptd_v3_graph_bundle_sanitized.pkl` | V3 graph bundle |
| `ptd_v3_graph_metrics.json` | V3 metrics |
| `ptd_v4_set_bundle_sanitized.pkl` | V4 set bundle |
| `ptd_v4_set_metrics.json` | V4 metrics |
| `ptd_acute_rescue_repairmix_s256_safe_logreg_candidate.pkl` | Rescue classifier |
| `ptd_acute_rescue_repairmix_s256_safe_logreg_candidate_metrics.json` | Rescue metrics |
| `ptd_convnext_tiny.pt` | ConvNext descriptor checkpoint |

### Main result JSON evidence

Location: `ArchiTexture_NeurIPS_ED_submission_20260502/rwtd_miner_github_repo/paper_repro/source_data/main_results/`

| File | Numbers |
|---|---|
| `rwtd_architexture_full256_official.json` | mIoU=0.4611, ARI=0.6966 ✓ matches paper |
| `rwtd_architexture_common253_official.json` | mIoU=0.4645, ARI=0.7013 ✓ matches paper |
| `stld_architexture_summary.json` | mIoU=0.6705/all, 0.7195/covered; ARI=0.7249/all, 0.7791/covered ✓ matches paper |

### Prior live-repro run outputs

Location: `outputs/repro_notebook/20260503_130943/proposal_repro/`

| File | Notes |
|---|---|
| `proposal_repro_table.csv` | RWTD mIoU=0.687/ARI=0.5, STLD mIoU=0.135/ARI=0.017 — deliberately different from paper headline numbers (smoke-path only) |
| `summary.json` | Mode: `rwtd_stld_live_train_and_infer` |
| `inference_metrics.csv` | Per-sample metrics |
| `training_candidates.csv` | Synthetic training data |
| `visuals/` | Input/proposal/GT/prediction panels |

---

## Anonymity issues

| Location | Issue | Severity |
|---|---|---|
| `notebooks/reproducibility.ipynb` cell 6 saved output | `<REDACTED_PATH>` absolute paths | **HIGH** |
| `notebooks/reproducibility.ipynb` cell 6 saved output | Author-specific path in saved output | **HIGH** |
| `scripts/feature_clustering/src/rwtd_sam3/eval/README.md` | `<REDACTED_PATH> representations/...` links | **HIGH** |
| Submission `rwtd_*.json` source data | `pred_folder: /home/user/...` (uses `/home/user/` which is safe) | OK |

---

## Summary

The `proposal_repro/` directory is entirely missing. The proposal-space code lives in the submission package. The paper number evidence exists as committed JSON files inside the submission package. The live smoke-test script (`run_proposal_repro.py`) that the notebook references is absent from the repository tree.
