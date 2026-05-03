# Release Notes

This public release is intentionally scoped to the final paper:

- main story: proposal-space commitment above a frozen SAM proposal bank
- main-body datasets: RWTD and STLD
- appendix-only supporting routes: feature-space diagnostics, ControlNet bridge, and CAID

Out-of-scope material from earlier internal branches is not part of this release's main narrative. In particular, DeTexture / Detector / ADE20K routes and AdaSam-style adaptor results are not promoted here.

## Verified results

The headline proposal-space numbers are reproduced from committed prediction masks by running the official evaluators. To verify locally:

```bash
python proposal_repro/verify_results.py
```

Expected output:

```
Route                  Metric        Got     Paper  Match
--------------------------------------------------------------
  RWTD full-256
  mIoU                   0.4611    0.4611     OK
  ARI                    0.6966    0.6966     OK
  STLD all-200
  mIoU                   0.6705    0.6705     OK
  ARI                    0.7249    0.7249     OK
Verdict: ALL MATCH
```

The reproducibility notebook (`notebooks/reproducibility.ipynb`) runs the same check and displays the comparison table inline in the **Proposal repro** section.

Prediction masks are in `ArchiTexture_NeurIPS_ED_submission_20260502/proposal-space-route/reports/` (RWTD) and `experiments/khan_synthetic_gallery_20260312/eval/strict_ptd_learned/masks/` (STLD). Ground-truth labels for RWTD are in `TextureSAM_upstream_20260303/Kaust256/labeles/`.
