# How to Turn PolyDiff into a High-Honors Senior Thesis

## Core goal

The thesis should stop being framed as:

> I built a diffusion model for polygons and tried some guidance.

It should become:

> I built a controlled testbed to study the mechanics of sampling-time guidance in coordinate-based diffusion models, and I used it to identify when guidance improves target properties, when it destabilizes generation, and which architectural biases help or hurt this tradeoff.

That is thesis-level. It gives you:

- a clear research question
- a controlled methodology
- multiple non-obvious findings
- conclusions that transfer to future work in the lab

The point is not to make PolyDiff look more “realistic.” The point is to make it produce enough **mechanistic knowledge** that it would have been irresponsible *not* to build it before trying restoration guidance in a molecular model.

---

## What the finished thesis needs to claim

At minimum, the final thesis should support something close to the following claim:

> In coordinate-based diffusion models, sampling-time guidance introduces a measurable tradeoff between target optimization and fidelity to the learned data distribution. This tradeoff depends on guidance strength, reverse-diffusion timing, and denoiser architecture. In a controlled polygon domain, these effects can be isolated and quantified, yielding practical design rules for later guidance in more complex generative systems.

If you can support that claim cleanly, PolyDiff is absolutely enough for a senior thesis.

---

## What is currently missing

Right now PolyDiff has good ingredients, but the thesis is still missing three things:

1. **Systematic experiments instead of one-off observations**
2. **Failure-mode analysis instead of just “it worked” or “it got worse”**
3. **At least one strong mechanistic discovery that feels broader than polygons**

So the work from here should be about converting your current project into a **controlled study**.

---

## The experiments you need

## Experiment 1: Establish a rigorous unguided baseline

### Question
Can the unguided diffusion model actually learn the polygon distribution well enough for the guidance experiments to mean anything?

### What to run
For the fixed-size hexagon setting, train and evaluate:

- MLP baseline
- GCN baseline
- GAT baseline

Use the same data distribution and same train/sample protocol for all three.

### What to measure
For a large generated set, not 64 samples. Use at least 5,000 and ideally 10,000 generated samples per run.

Compare generated vs training distributions on:

- regularity score
- edge-length coefficient of variation
- angle coefficient of variation
- radius coefficient of variation
- self-intersection rate
- centering drift
- scale drift

Also include:

- PCA or UMAP projection of train vs generated polygons
- gallery of representative generated polygons
- gallery of worst outliers

### What would count as a strong result
A strong result would be something like:

- the MLP matches the training distribution substantially better than GCN and somewhat better than GAT
- GCN shows obvious oversmoothing or outlier-heavy behavior
- GAT improves over GCN but still underperforms MLP on this globally constrained task

### Why this matters
This gives you a trustworthy base model and supports the thesis claim that architecture matters before guidance is even added.

---

## Experiment 2: Show the MLP vs GNN architecture mismatch clearly

### Question
Why does the simpler MLP outperform graph models on regular polygon denoising?

### What to run
Take the best checkpoint from each architecture and compare them on:

- same fixed-size hexagon dataset
- same number of diffusion steps
- same evaluation metrics
- same sampling budget

Then ablate GAT/GCN structure:

- GCN without residuals vs with residuals
- GAT without global pooled feature vs with pooled feature
- GAT with more message passing layers
- optional: GAT with longer-range edges if feasible

### What to look for
You are trying to show that the target task is **globally constrained** and that local message passing is a poor inductive bias for it.

### Strong enough thesis discovery
A very strong finding would be:

> For regularity-oriented denoising, access to the whole shape at once matters more than graph-local inductive bias. The MLP outperforms GNNs because the denoising target is a globally coordinated geometric correction, not a local relational decision.

That is a clean and interesting conclusion.

### What evidence would support it
- MLP has lower score-distribution error
- GCN has worse outliers and lower best-case regularity
- GAT reduces extreme failures but tends toward average-looking shapes
- adding a global pooled feature reduces outliers but suppresses perfect or near-perfect polygons

That would already be a respectable core chapter.

---

## Experiment 3: Quantify the guidance-strength tradeoff

### Question
As guidance strength increases, how do target-property improvement and distribution fidelity trade off?

### What to run
Using your best unguided architecture, probably the MLP in the fixed-size setting, sample at a sweep of guidance strengths.

For example:

- 0.0
- 0.5
- 1.0
- 2.0
- 4.0
- 8.0

Do this for:

- analytic regularity guidance
- learned regressor guidance if it is stable enough
- classifier guidance only if it works cleanly

### What to measure
For each guidance scale:

- mean regularity score
- median regularity score
- upper-tail score performance, e.g. fraction above 0.9 or 0.95
- self-intersection rate
- distance between generated and training score distributions
- any validity or degeneracy metrics
- sample diversity, if you can define a sensible one
- visual examples

### Strong enough thesis discovery
A strong result would look like:

- weak guidance improves target score modestly with little damage
- moderate guidance gives best tradeoff
- strong guidance sharply improves the score but causes off-manifold artifacts, reduced diversity, or degenerate geometry

Then your claim becomes:

> Guidance is not monotonic in usefulness. After a certain point, stronger gradients stop improving meaningful generation and start pushing samples off the learned manifold.

That is a real thesis result.

---

## Experiment 4: Test whether guidance timing matters

### Question
Is guidance equally useful at all reverse-diffusion steps, or does timing matter?

### What to run
Compare several schedules:

- no guidance
- guidance at all steps
- guidance only in early steps
- guidance only in middle steps
- guidance only in late steps
- ramped guidance increasing over time
- optional: decay schedule

For example, if you have T total steps:

- early-only: first 30%
- mid-only: middle 40%
- late-only: final 30%
- linear ramp
- quadratic ramp

### What to measure
Same metrics as the strength sweep:

- regularity score
- validity
- self-intersection
- off-manifold drift
- visual quality
- diversity

### Strong enough thesis discovery
A very strong result would be:

> Early guidance is unstable because the sample is still too noisy for the property gradient to be meaningful, while late guidance is much more effective because it acts on partially denoised structure.

This is one of the best possible results you could get, because it feels general, not polygon-specific.

If the data support it, this could easily become one of the thesis’s main findings.

---

## Experiment 5: Compare learned guidance vs analytic guidance

### Question
Does a learned property model behave differently from a directly computed differentiable objective?

### What to run
Compare:

- analytic regularity guidance
- learned regressor guidance
- learned classifier guidance

Use matched guidance scales as much as possible.

### What to measure
- final property scores
- stability
- failure cases
- sensitivity to noise level
- agreement between learned predictor and analytic score
- calibration on noisy inputs, if possible

### Strong enough thesis discovery
A strong result would be:

> Analytic guidance is more faithful to the target objective, while learned guidance is more flexible but less reliable under noisy intermediate states.

Or, if it goes the other way:

> Learned guidance provides smoother gradients and can outperform a harsh analytic objective, but only after being trained on noisy states matching the diffusion process.

Either way, this experiment gives real insight.

---

## Experiment 6: Identify and categorize failure modes

### Question
When guidance fails, how does it fail?

### What to run
For the worst-performing settings from the previous experiments, collect and categorize outputs.

Build a small failure taxonomy. For example:

- self-intersections
- spikes or collapsed edges
- large-scale blow-up
- near-regular but off-distribution shapes
- repeated average-looking shapes
- extreme asymmetry
- mode collapse

### What to produce
A figure panel with representative examples for each failure mode.

For each failure mode, note:

- which architecture produced it
- which guidance kind produced it
- which guidance scale produced it
- whether it appears early or late in the reverse process

### Strong enough thesis discovery
A strong thesis does not just report success. It shows that you understand the bad cases.

If you can say:

> Strong regularity guidance improved scores, but the model began exploiting the objective by producing low-diversity, inflated, or geometrically brittle samples

that makes the work much more mature.

---

## Experiment 7: Show trajectory-level behavior during sampling

### Question
What happens to samples over reverse diffusion, not just at the end?

### What to run
Track a batch of samples through reverse diffusion for:

- unguided
- weak guided
- strong guided
- late-only guided

Record metrics at intervals, such as every 25 or 50 steps.

### What to measure over time
- regularity score
- edge CV
- angle CV
- radius CV
- sample norm / scale
- self-intersection count or proxy
- optional: distance to training distribution proxy

### Strong enough thesis discovery
A strong result would be:

- unguided samples improve smoothly but plateau below the data distribution
- guided samples diverge early if guidance is too strong
- well-timed guidance produces late-stage correction without wrecking geometry

This would make your thesis feel much more like a real study of dynamics rather than just endpoint optimization.

---

## Experiment 8: Variable-size experiment to justify GNN relevance

### Question
If GNNs underperform on fixed-size regular hexagons, do they become more justified once the task actually requires variable-size graph handling?

### What to run
Use the variable-size ragged dataset for GAT and GCN.

Train on mixtures such as:

- 5, 6, 7, 8-gons
- optionally broader mixtures if stable

Compare:

- variable-size GAT
- variable-size GCN
- an MLP baseline only if you create a padded baseline for fairness
- or explain clearly why MLP is less natural here

### What to measure
- per-size score distributions
- overall score distribution
- whether size-conditioned generation matches training frequencies
- whether some polygon sizes fail systematically

### Strong enough thesis discovery
A strong result would be:

> GNNs are not universally worse. Their advantage becomes clearer when the representation must natively support variable graph size, even if they remain weaker than MLPs on the simpler fixed-size regularity task.

This saves you from sounding like “GNN bad, MLP good,” which would be too dumb and too broad.

---

## Experiment 9: Add a second analytic objective to show generality

### Question
Does the guidance mechanism only work for regularity, or can it steer toward qualitatively different goals?

### What to run
Use a second objective that is meaningfully different from regularity.

Candidates:

- area
- compactness
- perimeter target
- radial asymmetry
- restoration analogue

If area is too stupid and mostly causes scale blow-up, that is still useful if documented correctly. The restoration analogue would be stronger if stable enough.

### What to measure
Same full evaluation as before, but now compare whether:

- objective can be increased
- validity is preserved
- degenerate hacks emerge
- objective meaningfully differs from regularity guidance

### Strong enough thesis discovery
A very good result would be:

> Guidance is general enough to optimize multiple differentiable objectives, but different objectives induce different geometric pathologies, showing that objective design matters as much as guidance implementation.

That is a thesis-grade statement.

---

## Experiment 10: Restoration analogue, but only if it becomes a real study

### Question
Can you demonstrate that guidance can optimize an external-effect objective rather than an intrinsic geometric score?

### What to run
If the restoration analogue is stable enough, compare:

- no restoration guidance
- restoration-only guidance
- regularity-only guidance
- restoration + regularity guidance

Also test:

- constant restoration strength
- timestep-ramped restoration strength

### What to measure
- restoration score / proxy
- DNA-distance objective
- contact drift
- validity of generated polygons
- regularity score
- success rate by chosen threshold

### Strong enough thesis discovery
A strong result would be:

> External-effect guidance is possible, but it is structurally multi-objective: optimizing rescue alone produces degenerate samples, while combining rescue with a validity prior yields a more usable tradeoff.

This would tie your work most directly back to the lab’s long-term goal without overclaiming that you solved restoration.

If this experiment remains half-baked, do not make it central. Better to have one clean objective study than one messy fake-biology chapter.

---

## What discoveries would be strong enough for high honors

You probably do not need all ten experiments at full depth. For high honors, you need around three or four **strong findings**, each backed by systematic evidence.

Here are examples of result sets that would be strong enough.

### Strong result set A: Architecture + guidance dynamics
1. MLP beats GCN and GAT on fixed-size regular polygon denoising
2. Increasing guidance strength improves target score up to a point, then pushes samples off-manifold
3. Late-stage guidance outperforms early-stage guidance in the score-fidelity tradeoff
4. Failure modes can be categorized and linked to objective and architecture

This is already enough for a strong thesis if presented well.

### Strong result set B: Guidance mechanism study
1. Unguided model approximates the training distribution reasonably but under-samples high-regularity tail
2. Analytic regularity guidance successfully shifts the distribution upward
3. Learned predictor guidance behaves differently from analytic guidance on noisy states
4. Objective timing and scale determine whether improvement is meaningful or degenerate

This is probably the cleanest path.

### Strong result set C: Broader generality
1. Guidance works for regularity
2. Guidance also works for a second objective, but with different failure modes
3. Variable-size setup changes the architecture tradeoffs
4. Restoration-style external-effect guidance is feasible only when paired with a validity prior

This would be ambitious and impressive if it works.

---

## What “strong enough” looks like concretely

A senior thesis with high honors should probably have something like the following headline results.

### Example headline result 1
On 10,000 sampled hexagons, the MLP produces a generated regularity distribution close to the training set, while GCN and GAT underperform, with GCN showing pronounced oversmoothing and GAT reducing extreme failures only when given a pooled global feature.

Why this is good:
- clear
- non-obvious
- supported by architecture reasoning

### Example headline result 2
Analytic regularity guidance increases the fraction of near-perfect samples, but above a moderate guidance strength the generated distribution drifts away from the training manifold and develops identifiable geometric pathologies.

Why this is good:
- it is a real tradeoff, not just “higher is better”
- it feels generalizable

### Example headline result 3
Applying guidance only in the later portion of reverse diffusion yields better score improvement per unit of distributional damage than early-step or uniform guidance.

Why this is good:
- very strong mechanistic insight
- directly relevant to later molecular guidance work

### Example headline result 4
In the variable-size setting, graph models become more representation-appropriate, even though they remain less effective than the MLP on the simpler fixed-size regularity task, showing that architecture preference depends on the structure of the generation problem rather than data type alone.

Why this is good:
- avoids simplistic overclaiming
- shows mature interpretation

---

## Priority order: what to do first

If time is limited, do these in this order.

### Tier 1: absolutely necessary
1. Large-sample unguided baseline comparison
2. Guidance-strength sweep
3. Guidance-timing sweep
4. Failure-mode catalog with visual examples

Without these, the thesis is still too soft.

### Tier 2: highly valuable
5. Analytic vs learned guidance comparison
6. Variable-size experiment
7. Trajectory-level plots over sampling steps

These are what push it into stronger territory.

### Tier 3: only if stable
8. Second objective such as area or compactness
9. Restoration analogue study
10. More complex architectural ablations

Do these only if the earlier experiments are already clean.

---

## What figures your thesis should contain

At minimum, I would want these figures in a high-honors version.

1. **Train vs generated score distribution** for MLP, GAT, GCN
2. **Representative sample gallery** from each architecture
3. **Guidance-strength sweep plot**
4. **Guidance-timing sweep plot**
5. **Failure-mode figure panel**
6. **Trajectory-over-time plot** for guided vs unguided samples
7. **Variable-size per-n plot** if you include that section
8. **Restoration diagnostics plot** if that experiment becomes real

If your thesis has these figures and they are actually interpreted, it will look much more serious.

---

## What to avoid

Do not let the thesis become:

- a giant code dump
- a vague “future work toward molecules” essay
- a tour of every feature in the repo
- a fake claim that polygons are molecules

Also do not overclaim that PolyDiff proves restoration guidance will work for p53. It does not.

What it can prove is more modest and still valuable:

> before deploying expensive, noisy, domain-specific guidance in molecular diffusion, it is useful to understand the mechanics of guidance in a fully controlled coordinate-based setting.

That is completely defensible.

---

## Recommended thesis structure once the experiments are done

1. **Introduction**
   - research question
   - why guidance matters
   - why molecular systems are too entangled to study guidance cleanly

2. **Background**
   - DDPMs
   - guidance
   - molecule diffusion context
   - why restoration is hard

3. **PolyDiff as a controlled testbed**
   - domain design
   - data generation
   - objectives
   - architectures

4. **Unguided generation study**
   - MLP vs GCN vs GAT
   - fixed-size results
   - architecture interpretation

5. **Guidance mechanics study**
   - strength sweep
   - timing sweep
   - analytic vs learned guidance
   - failure modes

6. **Extensions**
   - variable-size
   - second objective
   - restoration analogue if good enough

7. **Discussion**
   - what transfers to real diffusion guidance
   - what does not
   - implications for later work in the lab

8. **Conclusion**
   - direct answer to the thesis question
   - what PolyDiff established
   - what remains open

---

## A realistic high-honors target

If you want a concrete target to aim for, here is a version that would be strong enough.

### Minimum high-honors package
- clean baseline comparison among MLP, GAT, GCN
- guidance-strength sweep with real quantitative tradeoff curves
- timing-schedule sweep showing late guidance is best
- visual and quantitative failure-mode analysis
- clear discussion tying these to future guidance in molecular diffusion

That alone could be excellent.

### Stronger high-honors package
Everything above, plus:
- analytic vs learned guidance comparison
- variable-size experiment
- restoration analogue as a smaller extension chapter

That would be very solid.

---

## Final answer

Yes, PolyDiff can become a strong senior thesis, but only if you stop treating it like a project and start treating it like a controlled scientific study.

The thesis has to answer:

- what guidance does
- when it helps
- when it breaks
- why architecture changes that behavior

Once you have that, the work is no longer “diffusion for polygons.”

It becomes:

> a mechanistic study of guidance in coordinate-based diffusion models

And that is absolutely thesis-worthy.
