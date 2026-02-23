# From Noise to Geometry: Training a Diffusion Model to Draw Polygons

What if you could start with pure Gaussian noise and end up with clean, plausible geometry? That is exactly what I’m building in this project: a diffusion model that learns a distribution over 2D polygons and then generates new polygon shapes from scratch.

This is compelling to me for two reasons. First, it is a clean sandbox for understanding generative modeling mechanics without the overhead of images or huge datasets. Second, polygon generation is still genuinely useful: synthetic geometric data supports simulation, CAD-style workflows, shape priors, and future conditioning tasks where we care about controllable structure.

Project repository: [github.com/Jerenemy/polydiff](https://github.com/Jerenemy/polydiff)

## What I’m Building

I’m training a DDPM-style model (`polydiff`) on synthetic near-regular polygons (currently hexagons). The workflow is:

1. Generate a dataset of polygons with controlled deformation.
2. Train a diffusion denoiser to predict noise at random timesteps.
3. Sample new polygons by iteratively denoising from random noise.
4. Evaluate shape quality using the same regularity metric used in data generation.

## Why This Is Useful

This setup lets me iterate fast on core research questions:

- How well does a simple MLP denoiser recover geometric structure?
- How much quality do we lose or gain when we change diffusion steps or architecture?
- Which inductive biases matter most before moving to heavier models (e.g., GNNs or equivariant models)?

Because everything is small and inspectable, it is easier to debug failure modes and build intuition for diffusion behavior.

## Technical Details

### Data generation

Training data comes from `polydiff.data.gen_polygons`.

- Each sample is an `n`-gon with smooth radial and angular jitter.
- Polygons are canonicalized by centering, RMS-scale normalization, and CCW ordering.
- Optional rejection removes self-intersecting samples.
- A continuous regularity score is computed from edge/angle/radius variation:
  - higher score = more regular polygon
  - lower score = more distorted polygon

Current training dataset (`data/raw/hexagons.npz`) has:

- `10,000` polygons
- `n=6` vertices per polygon
- stored regularity scores for each sample

### Model and diffusion process

The denoiser in `polydiff/models/diffusion.py` is a timestep-conditioned MLP:

- input: flattened polygon coordinates (`2 * n_vertices`) + sinusoidal timestep embedding
- output: predicted Gaussian noise (`epsilon`)

Diffusion config used in training:

- `n_steps = 1000`
- linear beta schedule from `1e-4` to `2e-2`
- objective: MSE between true noise and predicted noise (standard DDPM epsilon prediction)

### Training config

From `configs/train_diffusion.yaml`:

- batch size: `128`
- epochs: `100`
- learning rate: `2e-4`
- hidden dim: `256`
- time embedding dim: `64`
- MLP layers: `3`

Checkpoints are saved every `500` steps with a final checkpoint at:

- `models/diffusion_final.pt`

Given `10,000` samples and batch size `128`, this run performs about `7,900` optimizer steps.

## Preliminary Results

I evaluated the current sample file (`data/processed/samples.npz`, 64 generated hexagons from `diffusion_final.pt`) against the training distribution using the project’s regularity score and self-intersection check.

| Split | Count | Mean score | Std | Median | Max | Self-intersection rate |
|---|---:|---:|---:|---:|---:|---:|
| Training (`hexagons.npz`) | 10,000 | 0.7866 | 0.1312 | 0.8058 | 0.9991 | 0.0000 |
| Generated (`samples.npz`) | 64 | 0.4706 | 0.1010 | 0.4675 | 0.7150 | 0.0000 |

What this says right now:

- The model is generating valid, non-self-intersecting polygons in this sample set.
- Quality is still below the training distribution on regularity (lower mean score), so there is clear headroom.
- This is a promising baseline, but not yet matching target geometric quality.

## What’s Next

The next obvious upgrades are:

- evaluate larger generated sets (not just 64) for tighter statistics
- compare reduced sampling steps (`sampling.n_steps`) vs quality/speed
- move from a flat MLP denoiser toward graph-aware and eventually E(2)-equivariant architectures

That progression should make it clearer which architectural inductive biases most improve geometric fidelity.

## Reproducibility Commands

```bash
# Train
python3 -m polydiff.training.train --config configs/train_diffusion.yaml

# Sample
python3 -m polydiff.sampling.sample --config configs/sample_diffusion.yaml

# Visualize generated polygons and compute scores
python3 -m polydiff.data.plot_polygons data/processed/samples.npz --num 16 --compute_score
```
