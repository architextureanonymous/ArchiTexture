# Updated Sources Used

This file is the final evidence map for the revised NeurIPS manuscript in `paper_neurips_unified`. It supersedes the earlier inventory note and reflects the narrowed scope of the rewritten paper: proposal-space recoverability is the main claim; feature-space evidence is diagnostic only; ControlNet bridge and CAID are appendix-only supporting routes; DeTexture / Detector / ADE20K are excluded.

## Scope lock

Main-paper routes:

- RWTD
- STLD
- proposal-space recoverability via `ArchiTexture`
- feature-space diagnostic note only

Appendix-only supporting routes:

- ControlNet-stitched bridge benchmark
- CAID breadth check

Explicitly excluded:

- DeTexture / Detector / ADE20K

## Method sources

- `TextureSAM-v2/texturesam_v2/consolidator.py`
  - Main implementation evidence for proposal encoding, compatibility merging, conservative component scoring, and Stage-A selection.
- `TextureSAM-v2/texturesam_v2/features.py`
  - Confirms the 15-channel handcrafted texture map and 60-D region summary used inside the hybrid proposal descriptor.
- `TextureSAM-v2/texturesam_v2/ptd_encoder.py`
  - Confirms the PTD masked-crop encoder, ring-context construction, and descriptor dimensionality used by the retained pre-ring runs.
- `TextureSAM-v2/texturesam_v2/ptd_learned.py`
  - Confirms the 10-D pair features, 10-D component features, and random-forest Stage-A model families.
- `TextureSAM-v2/artifacts/ptd_learned_swinb_pre_ring_metrics.json`
  - Used to verify RWTD Stage-A training size, validation threshold, and train/validation example counts.
- `TextureSAM-v2/experiments/khan_synthetic_gallery_20260312/models/ptd_learned_split_metrics.json`
  - Used to verify the STLD Stage-A training layout (`mpeg7_shapes`) and sample counts.
- `TextureSAM-v2/scripts/run_texturesam2_acute_learned_rescue.py`
  - Main implementation evidence for the retained RWTD rescue path, including candidate-level features and the `candidate_cls` decision rule.
- `TextureSAM-v2/artifacts/ptd_acute_rescue_repairmix_s256_safe_logreg_candidate_metrics.json`
  - Used to verify the retained RWTD rescue bundle, model families, label mode, and PTD-only synthetic training regime.
- `TextureSAM-v2/scripts/run_stagea_learned_only.py`
  - Confirms the Stage-A-only deployment contract used on the supporting bridge and CAID routes.
- `TextureSAM-v2/scripts/run_strict_ptd_v11_dense_rescue.py`
  - Confirms the RWTD-only dense rescue layer above the frozen bank.
- `TextureSAM-v2/TextureSum2_paper/SUPPLEMENT_CONFIG_TABLE.md`
  - Used to verify PTD-only training scope, synthetic supervision scale, and deployment boundaries.

## Main matched comparison sources

- `rwtd_miner_github_repo/paper_repro/source_data/main_results/rwtd_architexture_full256_official.json`
  - RWTD full-256 `ArchiTexture` headline row: `0.4611 / 0.6966`.
- `rwtd_miner_github_repo/paper_repro/source_data/main_results/rwtd_architexture_common253_official.json`
  - RWTD common-253 `ArchiTexture` headline row: `0.4645 / 0.7013`.
- `rwtd_miner_github_repo/paper_repro/source_data/main_results/rwtd_texturesam_common253_official.json`
  - RWTD common-253 TextureSAM rerun row: `0.4684 / 0.6163`.
- `rwtd_miner_github_repo/paper_repro/source_data/main_results/rwtd_sam2_original_official.json`
  - RWTD raw SAM2.1-small baseline row: `0.1615 / 0.2183`.
- `TextureSAM-v2/experiments/khan_synthetic_gallery_20260312/direct_eval/direct_foreground_summary.json`
  - STLD all-200 and covered direct-foreground summaries.
- `TextureSAM-v2/experiments/khan_synthetic_gallery_20260312/stld_texturesam0p3_maskbank_summary.json`
  - STLD common-182 TextureSAM comparison row and coverage accounting.
- `TextureSAM-v2/experiments/khan_synthetic_gallery_20260312/stld_texturesam0p3_maskbank_per_image.csv`
  - STLD per-image overlap used for common-182 coverage and bootstrap alignment.
- `TextureSAM-v2/reports/reviewer_bootstrap_summary.json`
  - STLD bootstrap confidence intervals used in the results discussion.
- `TextureSAM-v2/reports/final_round_rwtd_bootstrap_summary.json`
  - RWTD final-round paired bootstrap support. The retained paper uses the `rwtd_common253_overlap_instances` block because it matches the official overlap-significance semantics.

## Oracle and stronger-baseline sources

- `TextureSAM-v2/scripts/build_rwtd_proposal_oracles.py`
  - Reviewer-facing RWTD oracle script.
- `TextureSAM-v2/reports/reviewer_oracles_rwtd_full256/rwtd_proposal_oracles_summary.json`
  - RWTD oracle summary used for the main paper oracle table and figure.
- `TextureSAM-v2/reports/reviewer_oracles_rwtd_full256/rwtd_proposal_oracles_per_image.csv`
  - RWTD per-image oracle metrics for auditability.
- `TextureSAM-v2/scripts/build_rwtd_generic_bank_baselines.py`
  - Reviewer-facing RWTD generic frozen-bank baseline script.
- `TextureSAM-v2/reports/reviewer_generic_bank_baselines_rwtd/rwtd_generic_baselines_summary.json`
  - RWTD medoid / spectral / agglomerative / greedy baseline summary.
- `TextureSAM-v2/reports/reviewer_generic_bank_baselines_rwtd/rwtd_generic_baselines_per_image.csv`
  - RWTD per-image generic baseline metrics.
- `TextureSAM-v2/scripts/build_rwtd_oracle_figure.py`
  - Generates the paper oracle decomposition figure.
- `paper_neurips_unified/figures/fig_rwtd_oracle_decomposition.png`
  - Main-paper Figure showing RWTD bank richness versus deployed recovery.
- `TextureSAM-v2/scripts/build_stld_proposal_oracles.py`
  - New STLD oracle script added in this revision.
- `TextureSAM-v2/reports/reviewer_oracles_stld/stld_proposal_oracles_summary.json`
  - STLD oracle summary used in the main text discussion and appendix table.
- `TextureSAM-v2/reports/reviewer_oracles_stld/stld_proposal_oracles_per_image.csv`
  - STLD per-image oracle metrics for auditability.
- `TextureSAM-v2/scripts/build_learned_single_selector.py`
  - Final-round learned top-1 selector script that reuses the retained Stage-A descriptors and score model while restricting inference to singleton proposals.
- `TextureSAM-v2/reports/final_round_learned_single_selector_rwtd/rwtd_learned_single_selector_summary.json`
  - RWTD learned single-selector summary used in the main-paper recoverability decomposition and ablation discussion.
- `TextureSAM-v2/reports/final_round_learned_single_selector_stld/stld_learned_single_selector_summary.json`
  - STLD learned single-selector summary used in the appendix support table and in the main-text discussion of top-1 selection versus full commitment.

## Route-level ablation and failure-analysis sources

- `rwtd_miner_github_repo/paper_repro/source_data/rwtd_controls/release_proposal_bank_baselines_summary.csv`
  - RWTD ablation rows for core-only, repair-only, rank-only, union, intersection, and lightweight reranker.
- `rwtd_miner_github_repo/paper_repro/source_data/rwtd_controls/release_common253_paired_significance_official_overlap.csv`
  - RWTD paired significance artifact retained for appendix-side auditability.
- `TextureSAM-v2/experiments/khan_synthetic_gallery_20260312/direct_eval/direct_foreground_summary.json`
  - STLD proposal-union / handcrafted / PTD-heuristic / learned comparison used for route-level ablation rows.
- `TextureSAM-v2/experiments/perlin_controlnet_eval_20260312/full_0p3/stageA_0p3/strict_ptd_learned/summary.json`
  - Bridge Stage-A union-versus-learned supporting ablation numbers.
- `rwtd_miner_github_repo/paper_repro/source_data/rwtd_audit/audit_summary.json`
  - RWTD audit counts and calibration summaries.
- `rwtd_miner_github_repo/paper_repro/source_data/rwtd_audit/wrong_partition_summary.json`
  - RWTD wrong-partition commitment counts used in the discussion.
- `rwtd_miner_github_repo/paper_repro/source_data/rwtd_audit/rescue_reliability_summary.json`
  - RWTD rescue activation and help/hurt breakdown.

## Feature-space diagnostic sources

- `research-site/experiments/Coarse Feature Clustering/rwtd/flip_averaged/experiment_terms.md`
  - Documents the clustering probe, flip-averaging, and coordinate-debiasing controls.
- `research-site/site/experiments/sam2-vs-sam3-frozen-feature-clustering/assets/tables/coarse_only_model_parity.csv`
  - Archived parity summary for coarse-only frozen-feature clustering.
- `research-site/site/experiments/sam2-vs-sam3-frozen-feature-clustering/assets/tables/flip_averaged_model_parity.csv`
  - Archived parity summary for flip-averaged clustering.
- `paper_neurips_unified/tables/feature_parity.tex`
  - Appendix table compiled from the archived parity artifacts.

## Supporting-route appendix sources

- `TextureSAM-v2/experiments/perlin_controlnet_eval_20260312/full_0p3/eval_0p3/comparison_summary.json`
  - Bridge comparison against TextureSAM for appendix reporting.
- `TextureSAM-v2/data/synthetic_texture_perlin_stitched_recovered/synthetic_texture_perlin_stitched/metadata.json`
  - Bridge construction metadata used to describe the route accurately and narrowly.
- `TextureSAM-v2/experiments/caid_eval_20260313/full_0p3/eval_0p3/binary_partition_summary.json`
  - CAID comparison summary used only in the appendix breadth table.

## Figure and asset reuse

- `paper_main/figures/fig_consolidation_logic.png`
  - Reused as the main proposal-space pipeline figure.
- `paper_main/figures/fig_ambiguity_commitment_full256.png`
  - Reused in the appendix RWTD failure-analysis section.
- `paper_neurips_unified/figures/fig_recoverability_loci.tex`
  - Revised conceptual figure with proposal-space visually dominant.

## Traceability and review docs

- `REVISION_REVIEWER_MEMO.md`
  - Pre-rewrite rejection-risk memo that locked the editorial hierarchy.
- `FINAL_ROUND_PLAN.md`
  - Final-round execution plan for the last manuscript pass.
- `FINAL_ROUND_RESULTS.md`
  - Final-round experiment and manuscript summary for the learned top-1 baseline, RWTD bootstrap support, and last-mile text changes.
- `EXPERIMENT_LEDGER.md`
  - Final experiment traceability ledger.

## Inspected but not retained as evidence

- `TextureSAM-v2/TextureSum2_paper/paper_body.tex`
  - Inspected for legacy wording and route naming only; not used as main evidence.
- `TextureSAM-v2/paper_imports/20260312_user_final/ArchiTexture_main/paper_body.tex`
  - Inspected for old proposal-route claims and audit wording only; not used as a headline evidence source.
- Any DeTexture / Detector / ADE20K artifacts
  - Inspected only to confirm exclusion from the revised paper.
