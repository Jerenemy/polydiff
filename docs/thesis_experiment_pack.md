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

The first study is the minimum package. The second isolates guidance timing and strength and now deliberately pushes regularity guidance into the plateau and breakdown regime with very large scales. The third asks which architecture best matches the baseline training distribution and whether guidance acts like a correction or a distortion for that architecture. The fourth tests whether architecture rankings change when the training polygons become noisier.

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

## Architecture Claim Guardrail

Use the studies to decide among three possible outcomes:

1. `mlp` dominates overall.
2. `mlp` and `gat` form a tradeoff, for example fidelity vs high-score tail.
3. graph models become more competitive as the training data become rougher.

Any of those outcomes is thesis-usable if the evidence is clear.
