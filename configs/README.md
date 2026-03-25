# Configs

YAML config files for training and sampling runs.

Main files:

- `train_diffusion.yaml`: diffusion-denoiser training
- `sample_diffusion.yaml`: sampling, guidance, diagnostics, and animation export
- `train_guidance_model.yaml`: classifier/regressor guidance-model training
- `study_minimum_high_honors.yaml`: example multi-case study manifest
- `study_guidance_characterization.yaml`: controlled MLP schedule/strength sweep, including extreme-scale regularity stress tests
- `study_architecture_noise_sweep.yaml`: architecture comparison across noisier training datasets

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

Important study-manifest knobs:

- `study.root_dir`: where numbered `study_*` folders are written
- `study.parallel.enabled`: opt into dependency-aware parallel case execution
- `study.parallel.max_workers`: maximum number of ready cases to run at once
- `study.parallel.require_cuda`: require `torch.cuda.is_available()` before enabling parallel mode
- `study.summary.reference_data_path`: reference dataset used for galleries and PCA
- `cases[*].kind`: `train_diffusion` | `train_guidance_model` | `sample_diffusion`
- `cases[*].overrides`: dotted-path config overrides applied to the case config
- `cases[*].tags`: lightweight metadata used for grouped study figures and interpretation guides
- placeholder values such as `{{train-mlp.run_name}}` let later cases consume earlier outputs

Recommended tag keys for thesis studies:

- `architecture`: `mlp` | `gat` | `gcn`
- `dataset_noise`: `baseline` | `noisy` | `very_noisy`
- `dataset_noise_rank`: numeric ordering for noise sweeps
- `analysis_group`: `architecture_baseline` | `guidance_schedule` | `guidance_strength` | `architecture_noise`
- `guidance_schedule`: `unguided`, `all`, `early`, `mid`, `late`, `linear_ramp`, ...
- `guidance_scale`: numeric strength value
- `guidance_baseline: true` on the unguided sample case when you want it included in both guidance sweeps

When those tags are present, the study runner can emit:

- combined architecture score overlays
- architecture metric panels
- guidance timing and strength sweeps
- architecture-vs-noise sweeps
- a heuristic outlier failure-mode figure

The current guidance characterization manifest uses analytic regularity guidance because it is aligned with the dataset geometry and therefore easier to interpret. Treat it as a favorable controllability probe, not as a claim that arbitrary guidance objectives will be equally safe.

If `sampling.size_distribution` is omitted for a GAT/GCN checkpoint, sampling defaults to the empirical training-size histogram saved in the checkpoint metadata.
