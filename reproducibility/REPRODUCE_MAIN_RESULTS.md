# Reproduce Main Results

This note focuses on the retained main-paper evidence. The fastest audit path is summary-first, then script-first if you have the full external assets.

## 1. Build the paper

```bash
cd paper
tectonic main.tex
```

## 2. Verify the committed main-result summaries

The final matched comparison table is backed by the JSON summaries committed under:

- `results/artifacts/main_results/rwtd_architexture_full256_official.json`
- `results/artifacts/main_results/rwtd_architexture_common253_official.json`
- `results/artifacts/main_results/rwtd_texturesam_common253_official.json`
- `results/artifacts/main_results/rwtd_sam2_original_official.json`
- `results/artifacts/main_results/stld_architexture_summary.json`
- `results/artifacts/main_results/stld_texturesam_summary.json`

The manifest mapping these files into the paper is:

- `results/RESULTS_MANIFEST.md`

## 3. Run lightweight local checks

```bash
pip install -e .
pytest tests
```

## 4. Full analysis reruns

The repo includes the in-scope analysis scripts used by the final paper round:

- `scripts/build_rwtd_proposal_oracles.py`
- `scripts/build_rwtd_generic_bank_baselines.py`
- `scripts/build_learned_single_selector.py`
- `scripts/build_rwtd_paired_bootstrap.py`
- `scripts/build_stld_proposal_oracles.py`

These full reruns require local dataset roots, upstream SAM assets, and pretrained checkpoints that are not committed here. Use the command provenance in `results/EXPERIMENT_LEDGER.md` together with the checkpoint notes in `checkpoints_manifest/README.md` to recreate the exact local runs.
