# Final Submission Results

## Descriptor ablation

This hardening-round ablation isolates the descriptor-sensitive Stage-A commitment layer. On RWTD, the dense rescue layer is unchanged across descriptor variants and operates on candidate masks rather than the 1084-D proposal descriptor, so the clean reviewer-facing comparison is Stage-A under the official evaluator.

Fresh Stage-A results:

| Benchmark / subset | Handcrafted-only | PTD-only | Hybrid |
| --- | --- | --- | --- |
| RWTD common-253 | `0.4469 / 0.6556` | `0.4486 / 0.6505` | `0.4558 / 0.6812` |
| RWTD full-256 | `0.4454 / 0.6533` | `0.4472 / 0.6484` | `0.4543 / 0.6786` |
| STLD common-182 | `0.7045 / 0.7808` | `0.6698 / 0.7740` | `0.7195 / 0.7791` |
| STLD all-200 | `0.6570 / 0.7265` | `0.6253 / 0.7203` | `0.6705 / 0.7249` |

Reading:

- RWTD is clearly the most descriptor-sensitive regime. Hybrid improves coherence by roughly `+0.026` to `+0.031` ARI over the single-family variants.
- STLD is less sensitive overall. Hybrid is the strongest mIoU setting, while handcrafted-only is marginally strongest on ARI.
- The reviewer-safe conclusion is narrower than “descriptor choice does not matter”: it does matter, especially on RWTD, but the proposal-space recoverability claim does not collapse to one hidden handcrafted shortcut.

## Practicality profile

The appendix now includes a compact practicality table built from retained profiling and audit artifacts.

Key values:

- RWTD Stage-A: `3.84` prompt proposals/image, `2.64` merged components/image, `0.025 s/image`, `430 MiB` peak RSS.
- RWTD full system: `38.05` dense proposals/image, `11.72` rescue candidates/image, `0.378 s/image`, `637 MiB` peak RSS, `21/256` images switched by the rescue layer.
- STLD Stage-A: `1.78` prompt proposals/image and `2.32` merged components/image. No separate runtime profile was recorded in the retained workspace.

## Feature appendix balance

The feature appendix was de-emphasized without deleting it:

- a new implementation/practicality section now comes before the feature appendix
- the archived parity table was reduced to a smaller single-column table
- the parity table now appears after the gallery and scale diagnostics instead of leading the section

## Manuscript sections changed

- `paper_neurips_unified/sections/05_proposal_space_recovery.tex`
- `paper_neurips_unified/sections/07_results.tex`
- `paper_neurips_unified/appendix.tex`
- `paper_neurips_unified/tables/algorithm_box.tex`
- `paper_neurips_unified/tables/descriptor_ablation.tex`
- `paper_neurips_unified/tables/practicality.tex`
- `paper_neurips_unified/tables/feature_parity.tex`

## Claims strengthened or softened

- Strengthened: the paper now explains explicitly why the modular tree-based decision layer is principled for limited synthetic supervision above a frozen bank.
- Strengthened: the RWTD vs STLD contrast is now stated as a scientific regime difference rather than left implicit.
- Strengthened: the appendix now answers the practicality objection directly.
- Softened/clarified: the descriptor ablation is reported at Stage-A on RWTD because that is the descriptor-sensitive layer. The dense rescue layer remains part of the main deployed RWTD system, but it is not the object of the descriptor-sensitivity test.
