# Pocket-Fit Conditioning Study

This note explains why `configs/study_pocket_fit_conditioning.yaml` exists, what it is meant to prove, and how it fits into the thesis.

## Thesis Role

This is the main bridge between the polygon work and the binding-affinity guidance problem.

It is not meant to show that polygon generation itself is the final application. It is meant to answer a design question that matters for the molecular pipeline:

- if you train a learned guidance surrogate on noisy diffusion states, should it also receive timestep `t`?
- does that make guided sampling better on the true conditional objective?
- does it do so without introducing more off-manifold damage than a timestep-agnostic surrogate?

That is the core reason to run this study before spending more time on the molecular system.

## Why This Study Is Better Than Regularity-Only Guidance

Regularity guidance is useful for studying mechanics, but it is unusually aligned with the polygon manifold. That makes it a favorable controllability probe, not a strong analogue of binding-affinity guidance.

The pocket-fit study introduces real tension between:

- fitting the conditioning context
- and staying on the learned ligand-shape manifold

That makes it much closer to the actual risk in the binding-affinity setting, where a surrogate can improve its target while pushing samples away from realistic structures.

## Main Hypothesis

The main hypothesis is:

- a pocket-conditioned surrogate trained on noisy ligand states and conditioned on `t` should better approximate the guidance signal encountered during reverse diffusion than the same surrogate trained without `t`

The main decision rule is:

- prefer the `with t` surrogate only if it improves true pocket-fit relative to `no t` without creating a worse reward-fidelity tradeoff

Analytic pocket-fit guidance is included as an upper-bound reference, because polygons let us compute the true reward exactly.

## What The Study Runs

The study performs four guided-sampling cases on the same held-out pockets:

- `unguided`
- `analytic-pocket-fit`
- `surrogate-no-t`
- `surrogate-with-t`

Before that, it builds the synthetic conditional dataset and trains:

- a ligand prior diffusion model
- a pocket-conditioned surrogate trained on noisy `x_t` without `t`
- a pocket-conditioned surrogate trained on noisy `x_t` with `t`

## What To Look At

Primary figures:

- `surrogate_timestep_mae.png`
  - tests whether the `with t` surrogate predicts the true reward better, especially in higher-noise bins
- `pocket_guidance_metric_panel.png`
  - compares reward gain, success rate, manifold drift, and invalidity across the four guidance cases
- `pocket_guidance_tradeoff.png`
  - the main thesis figure for this study; right and down is better
- `pocket_best_sample_gallery.png`
  - checks whether high-scoring samples are visually plausible fits rather than shrinkage or protrusion hacks
- `pocket_matched_seed_final_panel__*.png`
  - compares final outputs for the same initial noise and same pocket across unguided and guided cases
- `figures/animations/trajectory_compare__*.gif`
  - the most direct unguided-versus-guided comparison; same starting noise, same pocket, different guidance path
- `figures/animations/trajectory__*.gif`
  - single-case denoising movies for inspection and presentations

Primary metrics:

- `fit_score_mean`: true pocket-fit reward on final samples
- `fit_success_rate`: fraction above the success threshold
- `distribution_distances.shape_distribution_shift_mean_normalized_w1`: manifold drift relative to the ligand training distribution
- `generated_summary.self_intersection_rate`: explicit invalidity
- `surrogate_true_gap_mean`: whether surrogate-guided cases are merely optimizing the surrogate or actually improving the true reward

## What This Can Support In The Thesis

This study can support a claim like:

- in a controlled pocket-conditioned analogue, training the guidance surrogate on noisy states and conditioning it on timestep `t` improves true conditional-objective optimization more safely than a timestep-agnostic surrogate

It should not be used to claim:

- that polygons are the final application
- that this directly beats TargetDiff, KGDiff, or MOOD on molecular generation benchmarks
- that arbitrary guidance objectives are always safe

## How It Fits With The Other Studies

Use the studies together like this:

1. `study_guidance_characterization.yaml`
   - shows basic timing and strength behavior under an aligned objective
2. `study_distribution_fidelity.yaml`
   - shows which denoiser architectures best match the polygon distribution
3. `study_architecture_noise_sweep.yaml`
   - checks whether architecture rankings change as the data get rougher
4. `study_pocket_fit_conditioning.yaml`
   - answers how to train and deploy a learned conditional surrogate before moving back to the binding-affinity task

The first three establish the polygon diffusion story. The pocket-fit study is the methodological bridge back to the molecular guidance setup.

## Run Command

```bash
cd /Users/jeremyzay/Desktop/thayer_lab_research/2026/spring_2026/polydiff
source .venv/bin/activate
python -m polydiff.pocket_conditioning.study --config configs/study_pocket_fit_conditioning.yaml
```

If you already have a good ligand prior checkpoint, set `diffusion.checkpoint_path` in the config to skip retraining the prior.

The study config also has a `visualization:` block so you can choose:

- which held-out pocket is animated
- which cases get single-case GIFs
- which unguided-vs-guided cases are compared side by side
- animation frame count and FPS
