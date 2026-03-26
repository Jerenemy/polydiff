# Thesis Experiment Pack

This note defines the concrete study set that now exists in the repo for the thesis claims:

- "Systematic characterization of guidance strength and timing effects, with identified failure modes"
- "Architecture comparison showing why MLP outperforms GNNs on globally-constrained denoising tasks"

The second claim should be treated as a hypothesis, not a preset conclusion.

The same caution applies to guidance: the current guidance study uses analytic regularity guidance, which is unusually aligned with the data manifold and evaluation target. That makes it a good first mechanism study, but not a general claim about arbitrary guidance objectives.

## Recommended Study Order

1. `configs/study_minimum_high_honors.yaml`
2. `configs/study_guidance_characterization.yaml`
3. `configs/study_distribution_fidelity.yaml`
4. `configs/study_architecture_noise_sweep.yaml`
5. `configs/study_polygon_guidance_pooling.yaml`
6. `configs/study_pocket_fit_conditioning.yaml`
7. `configs/study_surrogate_guidance_pooling.yaml`

The first study is the minimum package. The second isolates guidance timing and strength and now deliberately pushes regularity guidance into the plateau and breakdown regime with very large scales. The third asks which architecture best matches the baseline training distribution and whether guidance acts like a correction or a distortion for that architecture. The fourth tests whether architecture rankings change when the training polygons become noisier. The fifth ports the surrogate-style pooling question into the polygon guidance-model path itself on variable-size data. The sixth is the pocket-conditioned polygon analogue: can a noisy-state surrogate with timestep conditioning approach analytic conditional guidance more safely than a timestep-agnostic surrogate? The seventh is the molecular surrogate ablation: once ligand coordinates are noised, does explicit timestep conditioning help, and which permutation-invariant pooling rule is strongest for the surrogate regressor?

The dedicated note for the fifth study lives at [`docs/pocket_fit_conditioning_study.md`](pocket_fit_conditioning_study.md).

## Training Dataset Presets

Generate the fixed-size datasets used by the architecture-noise study with:

```bash
python scripts/generate_data/generate_thesis_datasets.py
```

The presets are:

| Preset | File | Purpose |
|---|---|---|
| `baseline` | `data/raw/hexagons.npz` | Default near-regular training set |
| `noisy` | `data/raw/hexagons_noisy.npz` | Rougher polygons with broader deformation coverage |
| `very_noisy` | `data/raw/hexagons_very_noisy.npz` | Stress-test dataset with sharper local irregularity |

## What The Figures Are For

- `study_tradeoff.png`: overall score vs shape-fidelity comparison.
- `architecture_score_overlay__*.png`: compare full score distributions for MLP, GAT, and GCN on the same dataset.
- `architecture_metric_panel__*.png`: compare mean score, p99, high-threshold success rate, and shape drift side by side.
- `guidance_schedule_sweep.png`: identify whether timing changes help at fixed scale.
- `guidance_strength_sweep.png`: identify whether guidance is monotonic, saturating, or destabilizing.
- because the strength sweep now reaches very large scales, this figure should be read as a plateau/breakdown search rather than just a local tuning curve.
- `distribution_guidance_sweep.png`: compare whether increasing guidance scale moves each architecture toward or away from the training distribution while also changing score quality.
- `architecture_noise_sweep.png`: test whether more difficult training data changes the architecture ranking.
- `outlier_failure_modes.png`: compare where the worst samples fail.
- `study_polygon_guidance_pooling.yaml` is the config to use when you want the polygon-domain analogue of the pooling study, but on ragged polygons inside the main guidance-model path rather than the molecule surrogate path.
- `surrogate_timestep_mae.png`: for the pocket analogue, compare surrogate prediction error by noise level with and without timestep conditioning.
- `pocket_guidance_metric_panel.png`: compare unguided, analytic, surrogate-no-`t`, and surrogate-with-`t` on true reward and safety metrics.
- `pocket_guidance_tradeoff.png`: compare true conditional reward against shape-distribution drift in the pocket analogue.
- `pocket_best_sample_gallery.png`: visual check that reward gains correspond to plausible pocket fits rather than reward hacking.
- `pocket_matched_seed_final_panel__*.png`: same initial noise and same pocket across cases, so the end-state comparison is visually fair.
- `figures/animations/trajectory_compare__*.gif`: matched-seed unguided-vs-guided denoising comparison.
- `figures/animations/trajectory__*.gif`: single-case denoising trajectories for presentations and qualitative inspection.
- `surrogate_metric_panel.png`: compare pooled surrogate quality across `avg`, `max`, and `sum` readout, with and without timestep conditioning on noisy ligand coordinates.
- `surrogate_validation_curves.png`: compare optimization stability and convergence across the same surrogate cases.

## Architecture Claim Guardrail

Use the studies to decide among three possible outcomes:

1. `mlp` dominates overall.
2. `mlp` and `gat` form a tradeoff, for example fidelity vs high-score tail.
3. graph models become more competitive as the training data become rougher.

Any of those outcomes is thesis-usable if the evidence is clear.

## Surrogate Guidance Claim Guardrail

The pocket-conditioned polygon study is the main learned-guidance design test because the true conditional reward is available analytically. The clean claim is:

1. if a surrogate sees noisy states and timestep information, it should better approximate the guidance target that diffusion actually encounters
2. that better approximation should translate into a better true-reward vs fidelity tradeoff than a surrogate trained without `t`
3. analytic guidance serves as the upper-bound reference for what a well-aligned guidance signal could do in this conditional setting

That is the study that most directly informs a concrete binding-affinity pipeline decision:

- whether to train the molecular surrogate on noisy coordinates
- whether to condition it on diffusion timestep
- and whether the added complexity is justified by a better reward-safety tradeoff

The molecular surrogate study is not a direct diffusion-sampling comparison. It is a readiness test for a future learned guidance model. The clean claim there is:

1. once ligand coordinates are noised, timestep conditioning may or may not be necessary for a stable surrogate
2. the graph readout must be permutation-invariant, so pooling is a real architectural choice rather than an implementation detail
3. `avg`, `max`, and `sum` should be treated as competing inductive biases, not as equivalent defaults
