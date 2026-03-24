# Notebooks

Use this folder for exploratory debugging, visualizations, and one-off experiments.
Prefer committing lightweight notebooks or markdown summaries; keep large outputs in `data/`.

## Recommended setup

1. Install the project in editable mode in the same environment your notebook kernel uses:

```bash
pip install -e .[dev]
```

2. Start Jupyter from the repo root so relative paths resolve cleanly:

```bash
jupyter lab
```

3. In notebooks, import package modules directly (not from top-level `polydiff` unless exported):

```python
from polydiff.data import gen_polygons
from polydiff.training.train import train_from_config
from polydiff.sampling.sample import sample_from_config
```

## Variable-Size Note

The core codebase now supports ragged variable-size polygon datasets for `gat` and `gcn`.

Notebook status is mixed:

- utility code in `polydiff.data` now loads dense and ragged polygon files
- some older notebooks still assume a fixed `n_vertices` field and will need manual updates before they work on mixed-size datasets

In particular, comparison notebooks that align polygons vertex-by-vertex are still conceptually fixed-size unless they are rewritten to group or resample by polygon size.

## Common import issue

If Jupyter is launched from `notebooks/`, `import polydiff` can fail because the repo root is not on `sys.path`.
Use a bootstrap cell that finds `pyproject.toml` in parent directories and inserts that directory into `sys.path`.
`notebooks/exploration.ipynb` includes this pattern.
