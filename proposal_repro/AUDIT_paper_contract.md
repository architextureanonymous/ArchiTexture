# AUDIT_paper_contract.md — Paper Ground-Truth Route Matrix

**Extracted from:** `paper/main.tex`, `paper/abstract.tex`, `paper/tables/proposal_results.tex`,
`paper/tables/oracle_recoverability.tex`, `results/EXPERIMENT_LEDGER.md`,
`results/RESULTS_MANIFEST.md`, `REPRODUCIBILITY.md`, `notebooks/reproducibility.ipynb`

---

## 1. Route definitions

### Proposal-space route (main paper contribution)

- **What it is:** Frozen SAM (no fine-tuning) emits automatic prompt masks (proposal banks).
  A consolidation rule converts fragmented proposals into a coherent binary partition.
  The trained object is ONLY the consolidation rule — NOT the SAM backbone.
- **Stages:**
  - **Stage A (core-only):** Descriptor → compatibility scoring → graph construction →
    conservative core selection. No repair. Used as the ablation baseline and as the
    cross-dataset generalization route (STLD, ControlNet, CAID).
  - **Stage B (with repair, RWTD-specific):** Stage A + dense rescue layer (acute rescue,
    safe-switch logistic-regression classifier + gain regressor).
    Called "texturesam2_acute_learned_rescue" in the experiment ledger.
- **Deployed variant (RWTD main):** `ptd_learned_swinb_pre_ring` + acute rescue repair.
- **Deployed variant (STLD main):** `ptd_learned_split` (Stage A only, no repair).
- **Encoder:** SwinB (pre-ring) for RWTD. PTD ConvNext for STLD.
- **SAM version:** SAM2.1-small (frozen, automatic masks / prompt masks).
- **Training source:** PTD-style synthetic texture partitions only. No RWTD or STLD labels used during training.

### Feature-space route (auxiliary diagnostic only)

- **What it is:** Frozen SAM features passed through a lightweight probe (K-means / GMM
  over patch features). No proposal bank used.
- **Paper status:** Retained as appendix-only diagnostic. Not in main comparison table.
  No matched RWTD rerun on the official proposal-route evaluator was recovered.

---

## 2. Main paper route matrix

| Route name | Dataset | Split / subset | Training source | Proposal source | Encoder | Evaluator | Expected mIoU | Expected ARI | Repair? |
|---|---|---|---|---|---|---|---|---|---|
| ArchiTexture final | RWTD | full-256 | PTD synthetic only | frozen SAM2.1 prompt masks (multibank) | SwinB pre-ring | official RWTD invariant (eval_no_agg_masks.py) | **0.4611** | **0.6966** | YES (acute rescue) |
| ArchiTexture final | RWTD | common-253 | PTD synthetic only | frozen SAM2.1 prompt masks (multibank) | SwinB pre-ring | official RWTD invariant | **0.4645** | **0.7013** | YES (acute rescue) |
| TextureSAM public rerun (baseline) | RWTD | common-253 | (TextureSAM upstream) | TextureSAM masks | — | official RWTD invariant | 0.4684 | 0.6163 | N/A |
| SAM2.1-small rerun (baseline) | RWTD | full-256 | none | frozen SAM2.1 automatic | — | official RWTD invariant | 0.1615 | 0.2183 | N/A |
| ArchiTexture final | STLD | all-200 | PTD synthetic only | frozen SAM2.1 prompt masks | PTD ConvNext | direct foreground mIoU/ARI | **0.6705** | **0.7249** | NO (Stage A only) |
| ArchiTexture final | STLD | common-182 | PTD synthetic only | frozen SAM2.1 prompt masks | PTD ConvNext | direct foreground mIoU/ARI | **0.7195** | **0.7791** | NO (Stage A only) |
| TextureSAM public rerun (baseline) | STLD | common-182 | (TextureSAM upstream) | TextureSAM masks | — | direct foreground mIoU/ARI | 0.5140 | 0.7526 | N/A |
| SAM2.1-small rerun (baseline) | STLD | all-200 | none | frozen SAM2.1 automatic | — | direct foreground mIoU/ARI | 0.3686 | 0.5269 | N/A |

---

## 3. Oracle and recoverability table (RWTD common-253)

| Method | mIoU | ARI |
|---|---|---|
| Medoid single | 0.4673 | 0.5087 |
| Learned single selector | 0.4512 | 0.5601 |
| Core only (Stage A) | 0.4558 | 0.6812 |
| ArchiTexture final | 0.4645 | 0.7013 |
| Single frozen-proposal oracle | 0.5142 | 0.8146 |
| Bank upper bound | 0.5183 | 0.8580 |

---

## 4. Subsets and splits

| Name | Size | Description |
|---|---|---|
| full-256 | 256 images | All RWTD Kaust256 images |
| common-253 | 253 images | RWTD images where ArchiTexture produces a valid output (3 images skipped) |
| all-200 | 200 images | All STLD images |
| common-182 | 182 images | STLD images covered by TextureSAM maskbank |

---

## 5. Evaluators

| Dataset | Evaluator | Convention |
|---|---|---|
| RWTD | `eval_no_agg_masks.py` (official TextureSAM upstream) | Label-symmetric invariant: reports both orientations of binary mask, takes max |
| STLD | `eval_stld_direct.py` | Direct foreground: ArchiTexture mask vs GT foreground, no label flipping |

---

## 6. What the live smoke-path produces vs. what the paper reports

The `proposal_repro/run_proposal_repro.py` live repro is explicitly documented (REPRODUCIBILITY.md §Proposal Repro) as a **reviewer-safe smoke path** only. It:

- Trains on generated composites (no RWTD/STLD labels)
- Uses a **deterministic image-derived proposal bank** (not the heavyweight frozen SAM bank)
- Uses a **random forest selector** (not the full SwinB PTD stack)
- Evaluates on only a small held-out split

**It is not expected to reproduce the paper headline numbers.** The paper headline numbers come from the full artifact-backed path using the frozen SAM prompt-mask banks and the SwinB/ConvNext PTD encoders.

The prior notebook run produced: RWTD mIoU=0.687/ARI=0.5, STLD mIoU=0.135/ARI=0.017. These numbers are NOT comparable to the paper and are documented as such.

---

## 7. Appendix-only routes

| Route | Dataset | Evaluator | Status |
|---|---|---|---|
| ControlNet bridge | Synthetic Perlin-stitched (1742 images) | invariant + direct mIoU/ARI | Appendix only |
| CAID | CAID all-3104 | invariant mIoU/ARI | Appendix breadth only |
| Feature-space diagnostic | RWTD public-227, STLD-200, CAID-3104 | archived route-specific evaluators | Appendix diagnostic |
