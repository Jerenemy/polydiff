# Configs

YAML config files for training and sampling runs.

Main files:

- `train_diffusion.yaml`: diffusion-denoiser training
- `sample_diffusion.yaml`: sampling, guidance, diagnostics, and animation export
- `train_guidance_model.yaml`: classifier/regressor guidance-model training

Important compatibility rules:

- `model.type: mlp` expects a fixed-size dataset
- `model.type: gat` and `model.type: gcn` support fixed-size or variable-size polygon datasets
- `train_guidance_model.yaml` is still fixed-size only because learned guidance models are currently MLPs

Important sampling knobs:

- `sampling.n_steps`: reverse-diffusion step count override
- `sampling.size_distribution.values/probabilities`: optional polygon-size prior for GAT/GCN sampling
- `sampling.guidance.*`: classifier, regressor, analytic regularity, analytic area, or restoration guidance
- restoration guidance also accepts `min_timestep_weight` and `timestep_power` to ramp its strength over reverse diffusion
- `sampling.guidance.components`: optional additive guidance terms while preserving the legacy single-guidance form
- `sampling.restoration.*`: explicit opt-in ligand -> restored protein -> DNA-binding scene used by restoration guidance, diagnostics, and GIF overlays
- `sampling.diagnostics.*`: reference comparison JSON output
- `sampling.animation.*`: GIF export of reverse trajectories

If `sampling.size_distribution` is omitted for a GAT/GCN checkpoint, sampling defaults to the empirical training-size histogram saved in the checkpoint metadata.
