# Polydiff

DDPM diffusion model to generate 2D polygons. This repo is structured for clean data generation, repeatable experiments, and easy debugging.

**Structure**
- `polydiff/`: Python package (importable code)
- `scripts/`: CLI entrypoints
- `configs/`: YAML configs for training and sampling
- `data/`: generated data and outputs
- `models/`: local training artifacts and checkpoints
- `pretrained_models/`: externally trained checkpoints
- `notebooks/`: exploratory debugging and visualization
- `tests/`: pytest tests

**Quick Start**
1. Create an environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

2. Generate polygons (saved to `data/raw/` by default)
```bash
python -m polydiff.data.gen_polygons --n 6 --num 10000 --out hexagons.npz
```

Control how noisy/irregular generated polygons are:
- Less noisy (closer to regular polygons): lower `--radial_sigma` and `--angle_sigma`, optionally increase `--smooth_passes`.
- More noisy (more irregular polygons): raise `--radial_sigma` and `--angle_sigma`, optionally reduce `--smooth_passes`.

Examples:
```bash
# Less noisy
python -m polydiff.data.gen_polygons \
  --n 6 --num 10000 --out hexagons_clean.npz \
  --radial_sigma 0.08 --angle_sigma 0.04 --smooth_passes 5

# More noisy
python -m polydiff.data.gen_polygons \
  --n 6 --num 10000 --out hexagons_noisy.npz \
  --radial_sigma 0.30 --angle_sigma 0.20 --smooth_passes 1 --deform_dist uniform
```

3. Visualize polygons
```bash
python -m polydiff.data.plot_polygons hexagons.npz --num 16
```

Notes:
- Training/generated `.npz` files include stored `score`, so polygons are colored by score by default.
- Use `--show_scores` to draw numeric score labels on each subplot.

4. Run tests
```bash
pytest
```

**Training (DDPM)**
Install the ML deps and run training:
```bash
pip install -e .[dev]
# torch + pyyaml are in core deps
python -m polydiff.training.train --config configs/train_diffusion.yaml
```

**Sampling**
```bash
python -m polydiff.sampling.sample --config configs/sample_diffusion.yaml
```

To sample with fewer reverse steps, set `sampling.n_steps` in `configs/sample_diffusion.yaml`:
```yaml
sampling:
  num_samples: 64
  n_steps: 250
  out_path: "data/processed/samples.npz"
```
`sampling.n_steps` must be between `1` and the checkpoint's trained diffusion steps.

Visualize sampled outputs:
```bash
# samples are written to data/processed/samples.npz by default
python -m polydiff.data.plot_polygons data/processed/samples.npz --num 16

# for sampled files (no stored score), compute and show scores at plot time
python -m polydiff.data.plot_polygons data/processed/samples.npz --num 16 --compute_score
```

**CLI Wrappers**
If you prefer scripts instead of `-m` modules:
```bash
python scripts/generate_data/gen_polygons.py --n 6 --num 10000 --out hexagons.npz
python scripts/generate_data/plot_polygons.py hexagons.npz --num 16
python scripts/train_diffusion.py --config configs/train_diffusion.yaml
python scripts/sample_diffusion.py --config configs/sample_diffusion.yaml
```

**Data Conventions**
- If `--out` or input filenames are bare (no path), they resolve to `data/raw/`.
- Keep large files out of git unless using Git LFS.

**Debugging**
Use `notebooks/` for exploratory work and fast iteration. Keep notebook outputs small.



whats the analogy for adding in the context of the target protein? 
does the message passing go between the molecule and the protein? 
how do the features get added into the molecule (like the rd kit features)?
how to add in the atom type

