# AUDIT_status.md — Proposal-Repro Readiness

**Date:** 2026-05-03  
**Overall verdict: RED**

The `proposal_repro/` package is **absent** from the repository. The notebook cell that invokes the live repro script has an empty source and only retains a saved output from a prior local run that contains author-identifying absolute paths.

---

## GREEN/YELLOW/RED table

### A. Data and manifests

| Item | Expected by paper | Found in proposal_repro | Status | Evidence | Fix needed |
|---|---|---|---|---|---|
| RWTD image list (full-256) | 256-image Kaust256 dataset | Not present | RED | Missing dataset; documented as external | Reviewer must supply `datasets/RWTD` |
| STLD image list (all-200) | 200-image STLD dataset | Not present | RED | Missing dataset; documented as external | Reviewer must supply `datasets/STLD` |
| Subset IDs (common-253, common-182) | Defined by evaluator logic | Embedded in scripts in submission package | YELLOW | IDs not separately listed as a file | Export subset-id lists from submission reports |
| GT masks / labels | RWTD Kaust256/labeles, STLD GT | Not in repo (external) | RED | Documented as external | Datasets required |
| Manifest / checksums | Not required for live smoke path | N/A | N/A | Smoke path uses generated data | None |

### B. Frozen proposal source

| Item | Expected by paper | Found in proposal_repro | Status | Evidence | Fix needed |
|---|---|---|---|---|---|
| Frozen SAM proposal banks (paper scale) | `strict_ptd_v11_multibank/` (~heavy) | Not present — documented as not shipped | YELLOW | REPRODUCIBILITY.md explicitly notes heavyweight banks not distributed | Documented; acceptable for reviewer package |
| SAM version | SAM2.1-small, frozen, no fine-tuning | Referenced in submission package README | YELLOW | Not in a standalone config file | Add SAM version pinning to smoke-path config |
| Smoke-path proposal source | Deterministic image-derived bank | Produced by `run_proposal_repro.py` at run time | RED | Script missing from repo | Add `run_proposal_repro.py` to `proposal_repro/` |
| No accidental fine-tuned SAM | Confirmed: consolidation only | Code in submission package trains only consolidation | GREEN | `ptd_encoder.py`, `ptd_learned.py` are consolidation-side only | None |

### C. Descriptor / embedding

| Item | Expected by paper | Found in proposal_repro | Status | Evidence | Fix needed |
|---|---|---|---|---|---|
| SwinB PTD encoder checkpoint (RWTD) | `ptd_encoder_swinb_pre_ring.pt` | Present in submission package artifacts | GREEN | `submission/proposal-space-route/artifacts/ptd_encoder_swinb_pre_ring.pt` | None |
| PTD ConvNext checkpoint (STLD) | `ptd_encoder.pt` | Present in submission package | GREEN | `submission/.../khan_stld_parallel_20260312/results_full/models/ptd_encoder.pt` | None |
| DTD CNN checkpoint | `dtd_small_cnn.pt` | Present in submission package artifacts | GREEN | `submission/proposal-space-route/artifacts/dtd_small_cnn.pt` | None |
| Descriptor computation code | `proposal_space_route/dtd_encoder.py`, etc. | Present in submission package | GREEN | Full package in submission | None |
| Smoke-path descriptor | Handcrafted (no checkpoint needed) | Used in smoke-path run | GREEN | `summary.json` shows `descriptor_mode: handcrafted` | None |

### D. Consolidation

| Item | Expected by paper | Found in proposal_repro | Status | Evidence | Fix needed |
|---|---|---|---|---|---|
| Learned RWTD bundle | `ptd_learned_swinb_pre_ring.pkl` | Present in submission package | GREEN | `submission/.../artifacts/ptd_learned_swinb_pre_ring.pkl` | None |
| Learned STLD bundle | `ptd_learned_split.pkl` | Present in submission package | GREEN | `submission/.../khan_synthetic_gallery_20260312/models/ptd_learned_split.pkl` | None |
| V3 graph bundle | `ptd_v3_graph_bundle_sanitized.pkl` | Present in submission package | GREEN | `submission/.../artifacts/ptd_v3_graph_bundle_sanitized.pkl` | None |
| V4 set bundle | `ptd_v4_set_bundle_sanitized.pkl` | Present in submission package | GREEN | Present | None |
| Core selection thresholds | Embedded in learned bundle | In pkl artifacts | GREEN | Confirmed by script arguments in ledger | None |
| Consolidation code | `consolidator.py`, `pipeline.py`, etc. | Present in submission package | GREEN | Full package in submission | None |
| Smoke-path consolidation | Random forest selector | Produced by `run_proposal_repro.py` | RED | Script missing | Add script |

### E. Repair (RWTD-specific only)

| Item | Expected by paper | Found in proposal_repro | Status | Evidence | Fix needed |
|---|---|---|---|---|---|
| Rescue classifier | `ptd_acute_rescue_repairmix_s256_safe_logreg_candidate.pkl` | Present in submission package | GREEN | `submission/.../artifacts/` | None |
| Rescue metrics JSON | Present | Present in submission package | GREEN | `..._candidate_metrics.json` | None |
| Repair disabled for STLD | Stage A only per paper | Confirmed: STLD runs `run_strict_no_rwtd_supervision.py` (Stage A only) | GREEN | EXPERIMENT_LEDGER.md R07 | None |
| Repair disabled for ControlNet, CAID | Stage A only per paper | Confirmed: appendix routes use Stage A | GREEN | Ledger S01, S02 | None |

### F. Evaluation

| Item | Expected by paper | Found in proposal_repro | Status | Evidence | Fix needed |
|---|---|---|---|---|---|
| Official RWTD evaluator | `eval_no_agg_masks.py` from TextureSAM upstream | Present in submission package (`TextureSAM_upstream_20260303/`) | GREEN | `submission/.../TextureSAM_upstream_20260303/eval_no_agg_masks.py` | None |
| STLD direct evaluator | `eval_stld_direct.py` | Present in submission package scripts | GREEN | `submission/.../scripts/eval_stld_direct.py` | None |
| Paper-number JSON files | RWTD full-256/common-253 JSONs, STLD summary JSON | Present in submission package | GREEN | All 3 files found, numbers match paper exactly | None |
| Smoke-path evaluator | Per-sample mIoU/ARI on small held-out split | Produced by `run_proposal_repro.py` | RED | Script missing | Add script |

### G. Ergonomics

| Item | Expected by paper | Found in proposal_repro | Status | Evidence | Fix needed |
|---|---|---|---|---|---|
| One smoke-test command | `python proposal_repro/run_proposal_repro.py --output-root ... --rwtd-root ... --stld-root ...` | Command documented in REPRODUCIBILITY.md but script absent | RED | Script missing | Add `run_proposal_repro.py` |
| One command per paper table row | Full-scale runs documented in EXPERIMENT_LEDGER.md | Commands present in ledger; scripts present in submission | YELLOW | Ledger has full commands; data/assets not distributed | Acceptable for artifacts-package model |
| Runtime / GPU notes | Not present | Not present | YELLOW | No runtime profile in proposal_repro/ | Add runtime note to README |
| Missing data detection | Not present | Not present | RED | Script missing; can't detect if it doesn't run | Add clear error messages in script |
| README instructions | Absent from proposal_repro/ | Absent | RED | Directory was empty | Add README |

---

## Critical anonymity findings

| File | Issue | Action |
|---|---|---|
| `notebooks/reproducibility.ipynb` cell 6 saved output | `<REDACTED_PATH>` absolute path exposed in 2 lines | Clear cell outputs before public release |
| `scripts/feature_clustering/src/rwtd_sam3/eval/README.md` | `<REDACTED_PATH> representations/` links throughout (18 occurrences) | Fix or strip before public release |
| Submission package `rwtd_*.json` | Uses `/home/user/` (generic, not identifying) | OK |

---

## Summary by dimension

| Dimension | Status | Notes |
|---|---|---|
| A. Data and manifests | RED | External datasets required; no manifest or downloader |
| B. Frozen proposal source | RED | Smoke-path script missing; paper-scale banks not distributed (documented) |
| C. Descriptor / embedding | GREEN | All checkpoints present in submission package |
| D. Consolidation | YELLOW | Submission package has everything; smoke-path script missing |
| E. Repair | GREEN | Artifacts present; correct routes use repair vs Stage A |
| F. Evaluation | YELLOW | Paper JSONs match; smoke evaluator script missing |
| G. Ergonomics | RED | Smoke-test script missing; no README in proposal_repro/ |
| Anonymity | RED | Two files contain author-identifying `<REDACTED_PATH>` paths |

---

## Fixes applied during this audit

1. Created `proposal_repro/AUDIT_inventory.md` (this session)
2. Created `proposal_repro/AUDIT_paper_contract.md` (this session)
3. Created `proposal_repro/AUDIT_status.md` (this session)
4. Created `proposal_repro/README.md` with correct instructions (this session)

## Remaining reviewer-risk items

1. **`run_proposal_repro.py` is absent** — the notebook live section cannot run. This is the top risk.
2. **Notebook cell 6 contains `<REDACTED_PATH>` in saved outputs** — identity leak before public release.
3. **`scripts/feature_clustering/src/rwtd_sam3/eval/README.md`** — 18+ occurrences of `<REDACTED_PATH>` paths.
4. **No dataset downloader** — reviewer must obtain RWTD and STLD separately.
5. **One failing unit test** in submission package (`test_empty_input` in `test_consolidator.py`): `AssertionError: 256 != 0` — empty-input fallback behavior is wrong.
