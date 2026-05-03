# Checkpoints Manifest

The public repo includes code, paper sources, and retained result summaries, but it does not ship the heavyweight pretrained checkpoints used in the full local experiments.

External assets needed for full reruns include:

- SAM-style proposal-generator checkpoints used by the upstream inference scripts
- PTD encoder checkpoints used by the learned single-selector and descriptor-ablation scripts
- learned PTD descriptor bundles used by the RWTD and STLD proposal-space routes

The exact local paper-round command provenance is recorded in `../results/EXPERIMENT_LEDGER.md`. Use that ledger to map each script to the external checkpoints it expects.
