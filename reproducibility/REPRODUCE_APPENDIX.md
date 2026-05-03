# Reproduce Appendix

This note covers the appendix-only figures and supporting tables retained in the final paper package.

## 1. Use the committed appendix assets directly

The appendix figures already shipped with this repo live in:

- `appendix_assets/`
- `paper/figures/`

The descriptor ablation, practicality, and selector-oracle supporting summaries live in:

- `results/artifacts/descriptor_ablation/`
- `results/artifacts/selector/`
- `results/artifacts/oracles/`
- `results/artifacts/bootstrap/`

## 2. Paper-side appendix tables and figures

The appendix sections in `paper/appendix.tex` consume:

- `paper/tables/descriptor_ablation.tex`
- `paper/tables/practicality.tex`
- `paper/tables/feature_parity.tex`
- `paper/figures/fig_feature_gallery.png`
- `paper/figures/fig_feature_scale_diagnostics.png`
- `paper/figures/fig_rwtd_case_gallery.png`
- `paper/figures/fig_stld_case_gallery.png`

## 3. Original figure builders

The original builders used during paper assembly are included under `scripts/`, including:

- `scripts/build_appendix_assets.py`
- `scripts/build_controlnet_bridge_figure.py`
- `scripts/build_caid_audit_figure.py`
- `scripts/build_descriptor_ablation.py`

Some of these builders expect local asset roots that are not distributed in this public package. The committed figures and summary CSV/JSON files above are the audit-ready public subset.
