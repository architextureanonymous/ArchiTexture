# Paper Build

The final manuscript source and compiled PDF for the ArchiTexture submission live in this directory.

## Compile

```bash
cd paper
tectonic main.tex
```

This produces `paper/main.pdf`.

## Contents

- `main.tex`, `abstract.tex`, `sections/`, `tables/`, `figures/`: paper source
- `appendix.tex`: appendix included after references in the same PDF
- `scripts/build_appendix_assets.py`: builder used for the appendix galleries during the final paper round

The source is self-contained for the retained manuscript build. The prebuilt `main.pdf` is the same paper package referenced from the repo root.
