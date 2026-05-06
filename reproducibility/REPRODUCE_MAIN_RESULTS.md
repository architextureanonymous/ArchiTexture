# Reproduce Main Results

This note focuses on the retained main-paper evidence. The fastest audit path is summary-first, then script-first if you have the full external assets.

## 1. Build the paper

```bash
cd paper
tectonic main.tex
```

## 2. Verify the committed main-result summaries

The definitive verification is the proposal-space verifier, which re-runs the evaluators against committed prediction masks and checks the output matches reported numbers:

```bash
python proposal_repro/verify_results.py
```

This writes `proposal_repro/verified_results.json` and prints a match/fail table. All four dataset rows must show `OK`.

The narrative audit trail (experiment provenance, result manifest) is in:

- `results/RESULTS_MANIFEST.md`
- `results/EXPERIMENT_LEDGER.md`

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
