# Configs

YAML config files for training and sampling runs.

Main files:

- `train_diffusion.yaml`: diffusion-denoiser training
- `sample_diffusion.yaml`: sampling, guidance, diagnostics, and animation export
- `train_guidance_model.yaml`: classifier/regressor guidance-model training
- `train_surrogate_guidance_model.yaml`: ligand-context surrogate regressor training with noisy-coordinate corruption
- `study_minimum_high_honors.yaml`: example multi-case study manifest
- `study_guidance_characterization.yaml`: controlled MLP schedule/strength sweep, including extreme-scale regularity stress tests
- `study_architecture_noise_sweep.yaml`: architecture comparison across noisier training datasets
- `study_distribution_fidelity.yaml`: baseline-distribution study comparing unguided and guided architecture fidelity
- `study_pocket_fit_conditioning.yaml`: pocket-conditioned polygon analogue comparing analytic guidance to surrogate guidance with and without timestep conditioning
- `study_polygon_guidance_pooling.yaml`: variable-size polygon guidance-model study comparing `avg` / `max` / `sum` pooling with and without timestep conditioning
- `study_surrogate_guidance_pooling.yaml`: surrogate-study matrix comparing `avg` / `max` / `sum` pooling with and without timestep conditioning

Important compatibility rules:

- `model.type: mlp` expects a fixed-size dataset
- `model.type: gat` and `model.type: gcn` support fixed-size or variable-size polygon datasets
- `train_guidance_model.yaml` now supports `mlp`, `gat`, and `gcn`; graph guidance models can train on fixed-size or variable-size polygon datasets and expose `pooling: avg | max | sum`
- `train_surrogate_guidance_model.yaml` is separate from the polygon guidance path; it trains a ligand-context surrogate over pooled graph features instead of fixed-size polygons

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

Surrogate-study config notes:

- `data.pt_path` or `data.pt_dir` must point to saved ligand records
- `data.pdb_path` must point to the pocket structure used as context
- `noise.enabled: true` means ligand coordinates are corrupted during training and evaluation
- `model.timestep_conditioning` controls whether the surrogate receives the sampled diffusion timestep `t`
- `model.pooling`: `avg` | `max` | `sum`; this is the order-invariant ligand readout

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
- architecture-by-guidance fidelity sweeps
- architecture-vs-noise sweeps
- a heuristic outlier failure-mode figure

The current guidance characterization manifest uses analytic regularity guidance because it is aligned with the dataset geometry and therefore easier to interpret. Treat it as a favorable controllability probe, not as a claim that arbitrary guidance objectives will be equally safe.

If `sampling.size_distribution` is omitted for a GAT/GCN checkpoint, sampling defaults to the empirical training-size histogram saved in the checkpoint metadata.

The surrogate study runner lives at:

```bash
python -m predict_binding_affinity.studies.run --config configs/study_surrogate_guidance_pooling.yaml
```

The pocket-conditioned polygon analogue lives at:

```bash
python -m polydiff.pocket_conditioning.study --config configs/study_pocket_fit_conditioning.yaml
```

That study is the cleaner bridge to the binding-affinity pipeline when you want to test whether noisy-state surrogate training and timestep conditioning improve conditional guidance before paying the cost of molecule-scale experiments.

It is the config to cite when your thesis question is:

- should the learned guidance surrogate see noisy `x_t` and `t`?
- does that improve the true conditional objective, not just the surrogate score?
- does it do so with less off-manifold damage than a timestep-agnostic surrogate?

It also includes a `visualization:` block for:

- matched-seed unguided-vs-guided GIF comparisons
- per-case denoising GIFs
- a same-seed final-state comparison panel on one held-out pocket

For the thesis-facing explanation of that study, see:

- [`docs/pocket_fit_conditioning_study.md`](../docs/pocket_fit_conditioning_study.md)

The variable-size polygon guidance-model pooling study lives at:

```bash
python -m polydiff.studies.run --config configs/study_polygon_guidance_pooling.yaml
```

That is the polygon-domain counterpart to the molecular surrogate pooling study. It uses `train_guidance_model.yaml` with `classifier.type: gat`, ragged `mixed_polygons.npz`, and pooled graph readouts across `avg`, `max`, and `sum`, with and without timestep conditioning.
