# Why The Restoration Prototype Is Worth Building

This note summarizes what the PolyDiff restoration prototype has already taught us about how to build a model of ligand-mediated functional rescue, and why that work is not wasted even though the current system is still a toy.

The short argument is:

- the prototype forced the restoration idea into an implementable mathematical object
- it exposed several non-obvious failure modes that would also matter in a more realistic system
- it produced a reusable guidance, diagnostics, and visualization framework
- it clarified which parts of the restoration story must be modeled explicitly and which parts can be deferred

So the prototype is not just a demo. It is a de-risking instrument.

## The Main Conceptual Discovery

The most important discovery was that "restoration" cannot be left as a vague downstream intuition. It has to be written as a concrete conditional objective.

In practice, that meant the project had to move from:

```text
ligand binds mutant target
```

to:

```text
ligand contact
-> restored protein state
-> DNA-binding competence
-> DNA bound pose
```

That change matters because it distinguishes restoration from plain affinity guidance. The prototype made that distinction operational rather than rhetorical.

Without the prototype, it would have been easy to keep talking about "functional rescue" while actually optimizing only for pocket fit.

## The Prototype Forced A Useful Causal Decomposition

One of the clearest lessons is that the rescue mechanism has to be decomposed into intermediate variables.

The current toy model explicitly separates:

- ligand contact at a chosen site
- protein restoration score
- protein geometry interpolation from mutant toward WT-like
- DNA-binding activation
- final DNA-distance objective

That decomposition is useful even if every term is still hand-designed, because it tells you what a future realistic model would need to represent.

In other words, the prototype does not merely answer "can we guide generation?" It answers:

- what state variables are necessary to make restoration legible at all
- where the rescue signal should enter the generative process
- which hidden assumptions were previously being collapsed into one shortcut score

## The Guidance Law Became Clear

The prototype also clarified how restoration guidance can be made mathematically compatible with diffusion guidance.

What emerged is an energy-based conditional guidance view:

- define a differentiable restoration objective
- treat it as a log-potential or Boltzmann-style conditional term
- add its gradient to the reverse process

This is important because it means the prototype already supports a defensible thesis claim:

- restoration guidance can be framed as a conditional log-density contribution, not just an arbitrary heuristic force

That is a real modeling insight. It establishes the mathematical bridge between score-based generation and external-effect guidance.

## A Major Engineering Discovery: Absolute Pose Is Not Free

Another major discovery was that the diffusion model does not learn a meaningful world-frame pose by default.

That showed up immediately in early restoration GIFs:

- the polygon, target, and DNA-like body could appear jumbled or far apart
- the sample could seem to "fly away"
- visually dramatic trajectories could be pure translation artifacts rather than meaningful restoration behavior

The solution was scene anchoring:

```text
x_scene = x - mean(x) + scene_anchor
```

This is a very important lesson. Any future restoration model that relies on spatial relations must address frame choice explicitly. The prototype surfaced that requirement early and cheaply.

That alone makes it useful. It prevented a misleading interpretation of the generative trajectory and forced the project to separate shape generation from world-coordinate semantics.

## Another Key Discovery: Strong Guidance On Pure Noise Behaves Badly

The prototype also made clear that a timestep-agnostic restoration objective is unstable early in reverse diffusion.

What happens is:

- sampling starts from Gaussian noise
- the restoration proxy still tries to measure contact geometry on that noise
- if guidance is too strong, it overreacts before the denoiser has formed a plausible polygon

This produced weird initial states in the animations and diagnostics.

That discovery led directly to a better design:

- restoration guidance now receives the diffusion timestep
- its effective strength ramps up over reverse diffusion
- early noisy steps are treated cautiously
- late near-denoised steps receive the strongest restoration pressure

This is exactly the kind of design correction a prototype is supposed to reveal. It would have been much harder to infer only from theory.

## Validity And Restoration Are Competing Objectives

The project also discovered that restoration cannot be optimized in isolation.

If the model only optimizes for the external rescue score, it can try to "cheat" geometrically:

- collapse shapes
- distort polygons unnaturally
- ignore the training distribution
- exploit quirks of the proxy rather than solving the intended problem cleanly

That is why additive guidance matters:

- `restoration` supplies the functional objective
- `regularity` supplies a validity prior

This is not a minor implementation detail. It is a structural insight:

- restoration is a multi-objective generation problem, not a single-score maximization problem

That lesson will carry over to any more realistic version, where geometric plausibility, chemistry, and functional rescue will all compete.

## Diagnostics Turned Restoration Into Something Measurable

A prototype is valuable when it converts a qualitative story into a measurable process.

This one now gives you restoration-specific diagnostics such as:

- `protein_restoration_mean`
- `dna_binding_activation_mean`
- `contact_drift_mean`
- `dna_distance_mean`
- `restoration_success_rate`
- full reverse-trajectory summaries

That is important because it lets you ask real research questions:

- Does restoration rise monotonically during denoising?
- Does higher guidance help final rescue or just create unstable early behavior?
- Does regularity improve rescue quality or only sample validity?
- At what point in the reverse process does rescue become stable?

Those are exactly the kinds of questions that make a prototype scientifically useful rather than decorative.

## Visualization Was Not Cosmetic, It Was Diagnostic

The restoration GIF work was also not wasted.

The animations forced the model to be interpretable in a way scalar metrics alone would not.

They exposed:

- incorrect causal storytelling
- pose-frame problems
- layout problems in the renderer
- unstable early-step behavior under strong guidance
- ambiguity about what exactly was being "restored"

The current overlay now shows:

- mutant protein reference
- WT-like protein reference
- current restored-protein state
- ligand contact point
- DNA unbound reference
- DNA bound reference
- current DNA position

That visualization is useful because it makes causal errors obvious. When the system was telling the wrong story, the GIF made that visible immediately.

## The Prototype Preserved The Existing Pipeline Instead Of Forking It

Another useful discovery is architectural:

- restoration did not require a separate generative model
- it could be integrated as an opt-in conditional guidance path
- the old sampling, diagnostics, and non-restoration analysis modes were preserved

That is valuable because it shows the restoration concept is compatible with a clean modular system.

The project now has:

- explicit restoration toggles
- additive guidance composition
- per-run diagnostics
- per-run GIF export
- backward-compatible config handling where useful

This means the prototype has already generated reusable infrastructure, not just one-off code.

## The Prototype Clarified What A Future Realistic Version Would Need

A common mistake is to treat a toy model as useless because it is not already realistic.

That is the wrong standard here.

What the prototype has actually done is identify the next realism gaps precisely. A future stronger model would likely need:

- a learned or physics-based conditional expert instead of a fully hand-built proxy
- a richer protein state than one scalar restoration coordinate
- a real molecular representation instead of polygons
- a more realistic DNA-binding model than interpolation between two positions
- better calibration of the conditional guidance strength
- possibly a learned distribution over binding poses rather than imposed anchoring

This is exactly what makes the prototype useful: it turned "make a restoration model" into a concrete list of missing components.

## Why This Is Not A Waste Of Time

The prototype is worth building because it has already accomplished several high-value tasks:

- It separated restoration from affinity in a mathematically explicit way.
- It revealed that the causal chain must include an internal protein-restoration state.
- It discovered that pose anchoring is necessary for meaningful restoration geometry.
- It discovered that timestep-aware guidance is necessary to avoid pathological early behavior.
- It showed that restoration is inherently multi-objective and must be balanced against validity.
- It produced restoration-specific diagnostics and animations that make the process inspectable.
- It fit into the existing diffusion pipeline without breaking the rest of the project.
- It created a credible bridge from an abstract biological story to an implementable generative-guidance framework.

That is already a successful prototype outcome.

The right way to judge it is not:

- "does this already simulate real molecular rescue?"

The right way to judge it is:

- "did this make the restoration problem better posed, more measurable, and easier to extend?"

The answer is yes.

## The Strongest Thesis-Style Claim The Prototype Supports

The prototype supports a modest but meaningful claim:

- a diffusion generator can be guided not only by direct fit or intrinsic sample quality, but by a differentiable external-effect objective representing functional rescue

And after the revisions, it supports a stronger version:

- that external-effect objective can be structured to reflect a causal chain in which ligand binding restores a mutant protein toward a WT-like state, which then re-enables downstream DNA binding

That is already a nontrivial modeling result, even in toy form.

## The Best Way To Present Its Value

The most defensible way to present this prototype is:

- not as a realistic biophysical simulator
- not as a validated drug-design engine
- but as a mechanistically structured guidance prototype that de-risks the mathematical, algorithmic, and representational problems involved in restoration-oriented generative modeling

That framing is honest and strong at the same time.

It says:

- the prototype is simplified
- the proxy is hand-designed
- the geometry is toy
- but the central research question has been operationalized successfully

That is exactly the kind of progress a good prototype is supposed to deliver.

## Relationship To The Other Restoration Note

This file is about why the prototype is valuable.

For the exact toy mechanism, equations, and current visualization semantics, see:

- [`restoration_analogue.md`](restoration_analogue.md)
