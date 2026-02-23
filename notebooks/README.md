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

## Common import issue

If Jupyter is launched from `notebooks/`, `import polydiff` can fail because the repo root is not on `sys.path`.
Use a bootstrap cell that finds `pyproject.toml` in parent directories and inserts that directory into `sys.path`.
`notebooks/exploration.ipynb` includes this pattern.
