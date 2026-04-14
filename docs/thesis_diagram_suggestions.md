# Paper-Quality Diagram Suggestions For The Senior Thesis

This note is not a generic figure wishlist. It is a thesis-specific figure plan for the argument this project is best positioned to make:

> sampling-time guidance in coordinate-based diffusion models creates a tradeoff between target optimization and fidelity, and that tradeoff depends on guidance strength, guidance timing, and denoiser architecture.

The strongest thesis figures should do one of three jobs:

- explain the mechanism clearly
- support a core empirical claim
- make a failure mode legible

Do not treat every figure as a decoration. Each one should earn its space by carrying part of the argument.

## Global Figure Standards

Use these across the whole thesis so the document feels coherent instead of like a stack of lab notebook exports.

- Export vector graphics whenever possible: `PDF` or `SVG`, not screenshots.
- Use one consistent font family across all figures and the thesis body if possible.
- Keep final in-thesis text size readable at print scale: usually 8 to 10 pt after placement.
- Use direct labels on lines when possible instead of large detached legends.
- Keep the same color assignment throughout the thesis.
- Keep axis limits fixed across comparable panels.
- Use panel labels `A`, `B`, `C`, ... only when the panels are meaningfully discussed in the text.
- Prefer white backgrounds and light axis frames.
- Avoid rainbow colormaps, default matplotlib styling, and thick opaque grids.
- Every multi-panel figure should have one main message, not three unrelated ones.

## Suggested Color Roles

If you want one stable visual language across the thesis, this is a good starting scheme:

- training/reference data: black or charcoal
- unguided sampling: medium gray
- MLP: deep blue
- GAT: teal
- GCN: rust or dark orange
- analytic guidance: dark green
- learned surrogate without timestep: amber
- learned surrogate with timestep: blue-green
- failure / invalid / off-manifold regime: crimson

## Must-Have Figures

These are the figures most likely to strengthen the thesis substantially.

### 1. Project Overview Figure

**Purpose**

Explain the whole system in one page at the start of the methods chapter.

**What to show**

- synthetic polygon data generation
- diffusion-model training
- unguided sampling
- optional sampling-time guidance
- evaluation metrics and study outputs

**Best layout**

- left-to-right pipeline
- one row for data/training/sampling
- one lower row for diagnostics, studies, and restoration/pocket extensions

**Why it matters**

This gives the reader the map before they hit details. It also makes the project look like a controlled experimental platform instead of a single toy demo.

**Implementation notes**

- redraw this as a clean vector figure, not Mermaid
- use simple boxes, restrained arrows, and only a few equations
- no code identifiers inside the final thesis version unless they are genuinely informative

### 2. Guided Reverse Diffusion Mechanism Figure

**Purpose**

Explain exactly how guidance enters the reverse process.

**What to show**

- noisy sample `x_t`
- denoiser prediction `\hat{x}_{0,\theta}(x_t, t)` for the current default architecture
- optional note that older epsilon-trained checkpoints first convert `\hat{\epsilon}_\theta(x_t, t)` to `\hat{x}_{0,\theta}(x_t, t)`
- DDPM posterior mean
- external guidance gradient
- timestep schedule weight
- updated reverse mean used to sample `x_{t-1}`

**Best layout**

- circular or left-to-right loop for one reverse step
- equations placed next to the blocks they describe
- small inset showing early / mid / late schedules

**Why it matters**

This is one of the central mathematical ideas in the thesis. If this figure is clean, the rest of the guidance chapter becomes much easier to follow.

**Key message**

Guidance is not a second model replacing the denoiser. It is a gradient-based perturbation of the reverse transition.

### 3. Inductive Bias Comparison Figure

**Purpose**

Make the MLP vs GNN comparison intuitive before presenting quantitative results.

**What to show**

- MLP sees the whole flattened polygon at once
- GCN passes messages locally around the cycle
- GAT weights local neighbors by attention
- optional global pooled feature path for GAT/GCN

**Best layout**

- three aligned columns: `MLP`, `GCN`, `GAT`
- same input polygon at the top of each
- arrows showing information flow
- one-line caption under each architecture describing its bias

**Why it matters**

If the thesis claim is that this task is globally constrained and that local message passing is a mismatch, the reader should be able to see that argument before reading the metric tables.

**Key message**

Regularity-oriented denoising may reward global coordination more than local relational processing.

### 4. Baseline Fidelity Figure

**Purpose**

Show whether unguided models learn the training distribution well enough for guidance experiments to mean anything.

**What to show**

- score distribution overlay: train vs generated
- one embedding plot such as PCA or UMAP of train vs generated canonical polygons
- representative sample gallery
- outlier gallery

**Best layout**

- panel A: distribution overlay
- panel B: low-dimensional embedding
- panel C: representative samples
- panel D: worst failures

**Why it matters**

This establishes the baseline. Without it, the guidance story is under-motivated because the reader cannot tell whether the model had a coherent prior to begin with.

**Existing repo artifacts to reuse**

- `architecture_score_overlay__*.png`
- `outlier_failure_modes.png`

### 5. Architecture Comparison Figure

**Purpose**

Support the main architecture chapter with a single high-information figure.

**What to show**

- MLP, GAT, GCN side-by-side on the same dataset
- mean score, p95 or p99 score, self-intersection rate, distribution shift, and maybe best-tail success rate
- a few matched-size sample galleries beneath the quantitative panel

**Best layout**

- top: compact metric panel with shared ordering of architectures
- bottom: sample strips for each architecture

**Why it matters**

This is probably the most thesis-important empirical comparison outside the guidance tradeoff plots.

**Key message**

Architecture choice changes both average fidelity and the shape of the failure distribution.

### 6. Guidance Strength Tradeoff Figure

**Purpose**

Show that stronger guidance is not monotonically better.

**What to show**

- x-axis: guidance strength
- left y-axis or panel: target property improvement
- right y-axis or companion panel: fidelity or distribution shift
- optionally self-intersection or failure rate as a red overlay

**Best layout**

- preferred: two aligned panels
- panel A: line chart vs scale
- panel B: Pareto-style frontier of target gain vs fidelity cost

**Why it matters**

This is one of the core thesis claims. If the curve shows a knee, plateau, or breakdown, that is a strong mechanistic result.

**Existing repo artifacts to reuse**

- `guidance_strength_sweep.png`
- `study_tradeoff.png`

### 7. Guidance Timing Figure

**Purpose**

Show that when guidance is applied matters, not just how much.

**What to show**

- schedule shapes: all, early, mid, late, ramp
- resulting performance for each schedule at fixed scale
- one matched-seed qualitative example if possible

**Best layout**

- panel A: tiny schedule cartoons
- panel B: metric comparison across schedules
- panel C: matched-seed final outputs or trajectories

**Why it matters**

This is the figure most likely to produce a general conclusion that feels broader than polygons.

**Key message**

Late guidance acts on partially formed structure and can be much more effective than early guidance on near-pure noise.

**Existing repo artifacts to reuse**

- `guidance_schedule_sweep.png`

### 8. Failure Mode Taxonomy Figure

**Purpose**

Make “off-manifold” concrete.

**What to show**

- oversmoothed polygons
- collapsed or degenerate shapes
- overly average-looking outputs
- spiky or distorted high-guidance failures
- self-intersecting failures if present

**Best layout**

- a grid of thumbnail polygons
- one short label under each failure family
- optional metric summaries beside each family

**Why it matters**

A thesis becomes stronger when it analyzes why things fail, not just whether the score went up or down.

**Key message**

Guidance does not only change averages. It changes the geometry of failure.

### 9. Restoration Causal Diagram

**Purpose**

Explain why restoration is not the same as pocket fit.

**What to show**

- ligand-like polygon
- mutant protein reference
- restored protein state
- DNA-binding competence
- DNA bound pose

**Best layout**

- explicit causal chain:
  `ligand contact -> protein restoration -> DNA-binding activation -> DNA bound state`
- show the intermediate latent variables, not just the endpoints

**Why it matters**

This is the conceptual figure that makes the restoration extension scientifically legible. Without it, the reader may interpret the whole section as “another guidance score.”

**Key message**

The restoration prototype encodes a causal decomposition, not just a geometric fit heuristic.

### 10. Pocket Conditioning Comparison Figure

**Purpose**

Compare unguided, analytic guidance, surrogate without timestep, and surrogate with timestep.

**What to show**

- true pocket-fit reward
- shape-fidelity or distribution-drift cost
- representative best samples
- matched-seed comparisons for fairness

**Best layout**

- panel A: metric panel
- panel B: tradeoff scatter
- panel C: best-sample gallery
- panel D: matched-seed final panel

**Why it matters**

This figure connects the controlled polygon work to the later molecular-guidance motivation.

**Existing repo artifacts to reuse**

- `pocket_guidance_metric_panel.png`
- `pocket_guidance_tradeoff.png`
- `pocket_best_sample_gallery.png`
- `pocket_matched_seed_final_panel__*.png`

### 11. Surrogate Error By Noise Level Figure

**Purpose**

Show whether timestep conditioning actually helps the surrogate approximate noisy-state rewards.

**What to show**

- MAE by timestep bin
- with-timestep vs without-timestep surrogate
- optional analytic upper-bound reference if appropriate

**Best layout**

- clean line chart with timestep bins on x-axis
- shaded confidence region if you have repeated evaluations

**Why it matters**

This is the strongest figure for the claim that noisy-state surrogate training should explicitly account for diffusion timestep.

**Existing repo artifacts to reuse**

- `surrogate_timestep_mae.png`

### 12. Thesis Summary Figure

**Purpose**

End the results chapter with a compact “design rules” figure.

**What to show**

- architecture outcome
- best guidance-strength regime
- best timing regime
- main failure mode beyond the optimum
- lesson for future molecular guidance

**Best layout**

- either a table-like visual summary
- or a phase-diagram style chart with annotated safe / risky regions

**Why it matters**

Readers remember synthesis figures. This is where you convert many experiments into a few durable conclusions.

## Strong Appendix Figures

These are useful if space allows, but they should not displace the core figures above.

### A. Training-Diagnostics Figure

- training loss
- EMA loss
- gradient norm
- sample diagnostics over training

Use this to support stability claims, not as a centerpiece.

### B. Variable-Size Pipeline Figure

- ragged storage
- graph-batch construction
- variable-size sampling by histogram or requested size distribution

Useful if variable-size polygons are a substantial part of the final thesis narrative.

### C. Pose Canonicalization Figure

- raw polygon
- centered / RMS-normalized polygon
- anchored-in-scene polygon for restoration

Useful if you want to explain why absolute pose is not meaningful by default.

### D. Matched-Seed Trajectory Figure

- same initial noise
- unguided vs guided denoising path
- 4 to 6 selected frames rather than every timestep

This works best in appendix or presentation slides unless the qualitative contrast is unusually strong.

## Figures To Avoid

These usually weaken the thesis unless they directly support a point.

- screenshots of notebooks
- Mermaid exports in the final PDF
- giant legends disconnected from the data
- tables that should be plots
- plots with more than 5 to 6 meaningful series in one axis
- default matplotlib styling with small fonts and thin saturated colors
- redundant figures that repeat the same result with only a minor metric change

## Recommended Chapter Placement

### Introduction / Methods setup

- project overview figure
- guided reverse diffusion mechanism
- restoration causal diagram

### Baseline modeling chapter

- baseline fidelity figure
- architecture comparison figure
- optional inductive-bias cartoon

### Guidance characterization chapter

- guidance strength tradeoff figure
- guidance timing figure
- failure mode taxonomy

### Conditional guidance / future-work chapter

- pocket conditioning comparison figure
- surrogate error by noise level
- thesis summary figure

## Practical Production Advice

If you want these to look paper-quality, the last 20% matters a lot.

- Generate the quantitative base in Python.
- Export to `SVG` or `PDF`.
- Do final annotation and spacing cleanup in Illustrator, Inkscape, Figma, or TikZ.
- Align panel widths and baselines manually.
- Trim repeated axis labels in multi-panel figures.
- Use short captions with one claim sentence first, then one methods sentence.
- Put the key claim in the caption, not just the surrounding text.

## Fastest High-Value Figure Set

If time is limited, make these six figures excellent instead of making twelve figures merely adequate:

1. project overview figure
2. guided reverse diffusion mechanism
3. architecture comparison figure
4. guidance strength tradeoff figure
5. guidance timing figure
6. pocket conditioning comparison figure

That set is enough to make the thesis feel deliberate, mechanistic, and research-driven.
