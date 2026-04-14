# `polydiff/models/diffusion.py` Reference

## Status Update

The codebase no longer assumes fixed-size polygons everywhere.

Current behavior:

- `DenoiseMLP` is still fixed-size and operates on flattened `(B, 2n)` tensors
- `DenoiseGAT` and `DenoiseGCN` now support variable-size polygons through ragged graph batches
- variable-size GAT/GCN training and sampling use concatenated node tensors plus per-graph `num_vertices`, not padded storage
- `DiffusionConfig.prediction_target` selects whether the denoiser learns `x0` directly or `epsilon`; the current training default is `x0`
- checkpoint loading preserves that target, and legacy checkpoints without it are treated as epsilon-trained

The conceptual discussion below still contains fixed-size examples because that was the original baseline setup, but the implementation status has changed. For a focused description of the current ragged GAT/GCN pipeline, see [`../../docs/variable_size_pipeline.md`](../../docs/variable_size_pipeline.md).

This document explains the current diffusion stack and all three denoisers:

- `DenoiseMLP`
- `DenoiseGAT`
- `DenoiseGCN`

It is meant to answer three different questions at once:

1. What is the model doing intuitively?
2. What equations is it trying to approximate?
3. Where is the corresponding code?

## Source Map

Main files:

- [`diffusion.py`](diffusion.py) lines 15-455: timestep embedding, denoisers, denoiser builder, and DDPM wrapper
- [`gat.py`](gat.py) lines 6-303: graph attention implementation used by `DenoiseGAT`
- [`gcn.py`](gcn.py) lines 8-34: base `GraphConvolution` used by `DenoiseGCN`

Train/sample integration:

- [`../training/train.py`](../training/train.py) lines 52-54: read `model_cfg` and build the denoiser
- [`../sampling/runtime.py`](../sampling/runtime.py) lines 57-61: rebuild the denoiser from checkpoint `model_cfg`

Configs:

- [`../../configs/train_diffusion.yaml`](../../configs/train_diffusion.yaml) lines 12-16: choose `model.type`, `hidden_dim`, `time_emb_dim`, `num_layers`
- [`../../configs/sample_diffusion.yaml`](../../configs/sample_diffusion.yaml) lines 6-13: choose checkpoint and sampling options

## Notation

- `B`: batch size
- `n`: number of polygon vertices
- `D = 2n`: flattened polygon dimension
- `x_0`: clean polygon
- `x_t`: polygon after adding noise at timestep `t`
- `eps`: Gaussian noise used in the forward process
- `x0_theta(x_t, t)`: model prediction of the clean polygon
- `eps_theta(x_t, t)`: model prediction of the added noise
- `u_theta(x_t, t)`: raw model output, which is either `x0_theta` or `eps_theta` depending on config
- `T`: number of diffusion steps
- `beta_t`: noise variance added at step `t`
- `alpha_t = 1 - beta_t`
- `bar_alpha_t = prod_{s=0}^t alpha_s`

Unless stated otherwise:

- flattened polygon tensors have shape `(B, D)`
- per-node graph tensors have shape `(B, n, F)` before batching graphs together

## `SinusoidalPosEmb`

Relevant code:

- [`diffusion.py`](diffusion.py) lines 15-31

### Intuition

The denoiser needs to know whether it is removing a tiny amount of noise near the end of sampling or a large amount of noise near the beginning. `SinusoidalPosEmb` turns the scalar timestep `t` into a vector with multiple frequencies so downstream layers can tell those regimes apart.

### Math

For embedding width `d`, let `half = floor(d / 2)` and define frequencies

```text
omega_i = exp(-log(10000) * i / (half - 1)),   i = 0, ..., half - 1
```

Then for sample `b`:

```text
phi_b = [sin(t_b * omega_0), ..., sin(t_b * omega_{half-1}),
         cos(t_b * omega_0), ..., cos(t_b * omega_{half-1})]
```

If `d` is odd, one zero is appended to keep the width exact.

### What the code does

- Builds the frequency vector in [`diffusion.py`](diffusion.py) lines 23-28
- Concatenates sine and cosine features in [`diffusion.py`](diffusion.py) lines 28-30

## `DenoiseMLP`

Relevant code:

- [`diffusion.py`](diffusion.py) lines 34-57

### Intuition

This is the baseline that ignores graph structure. It treats the whole polygon as one long coordinate vector and asks a standard MLP to infer the configured diffusion target directly from that vector plus the timestep.

This is a reasonable baseline when:

- the polygon always has the same number of vertices
- the vertex ordering is consistent
- you want the simplest possible denoiser first

### Features

The MLP sees exactly two kinds of information:

1. the full flattened noisy polygon `x_t`
2. one shared timestep embedding `phi(t)`

It does not see:

- explicit adjacency
- local neighborhoods
- per-vertex positional encodings
- pooled global graph features

### Math

Let `phi(t)` be the learned timestep embedding produced by:

```text
phi(t) = SiLU(W_t * SinusoidalPosEmb(t))
```

Then the model is:

```text
u_theta(x_t, t) = f_MLP([x_t ; phi(t)])
```

where `[ ; ]` means concatenation.

### Architecture

1. Build `phi(t)` using `SinusoidalPosEmb -> Linear -> SiLU`
2. Concatenate `[x_t ; phi(t)]`
3. Run `num_layers` hidden `Linear -> SiLU` blocks
4. Project back to `D`

### Current limitations

- no graph inductive bias
- depends on stable vertex ordering
- no explicit permutation symmetry handling
- no explicit geometric equivariance

## `DenoiseGAT`

Relevant code:

- Wrapper and feature assembly: [`diffusion.py`](diffusion.py) lines 59-169
- Base GAT implementation: [`gat.py`](gat.py) lines 6-303

### Intuition

This model treats the polygon as a ring graph. Each vertex can communicate with:

- itself
- its previous neighbor
- its next neighbor

The model is trying to do two things at once:

1. local cleanup: use nearby vertices to infer what a denoised local geometry should look like
2. global coordination: use a pooled summary so the whole polygon stays coherent

The GAT is meant to be a better inductive bias than the MLP because polygons really are graphs, not arbitrary flat vectors.

### Exact features given to each node

For vertex index `k` in a polygon with `n` vertices, the initial node feature is:

```text
h_k^(0) = [x_k, y_k,
           sin(2*pi*k/n), cos(2*pi*k/n),
           sin(4*pi*k/n), cos(4*pi*k/n),
           phi(t)]
```

So the node gets:

1. noisy `x` coordinate
2. noisy `y` coordinate
3. first-harmonic sine position on the ring
4. first-harmonic cosine position on the ring
5. second-harmonic sine position on the ring
6. second-harmonic cosine position on the ring
7. the timestep embedding `phi(t)`, repeated at every node

Relevant code:

- timestep MLP: [`diffusion.py`](diffusion.py) lines 73-77
- fixed cycle positional features: [`diffusion.py`](diffusion.py) lines 78-82 and 127-138
- node feature concatenation: [`diffusion.py`](diffusion.py) lines 157-160

### Why these features were chosen

Intuition:

- `(x_k, y_k)` tells the model what the current noisy polygon looks like.
- `phi(t)` tells it how much denoising is needed at this timestep.
- the sine/cosine cycle features tell it where the node sits around the ring

The cycle features are important because without them, a high-noise cycle graph can look like an anonymous ring of almost interchangeable nodes.

Why fixed instead of learned?

- the polygon start vertex is not globally meaningful
- fixed circular features give relative phase on the ring
- they break symmetry without pretending vertex `0` has a stable world-space meaning

### Graph topology

The edge set is:

```text
E = {(k, k), (k, k+1 mod n), (k, k-1 mod n)}
```

So every node sees:

- itself
- one step clockwise
- one step counterclockwise

Relevant code:

- cycle edge construction: [`diffusion.py`](diffusion.py) lines 114-125
- batched edge index: [`diffusion.py`](diffusion.py) lines 140-148

### Attention math

Inside `GATLayer`, the implementation first projects node features:

```text
z_u = W h_u
```

Then for source node `u` and target node `v`, it computes an additive attention score:

```text
e_uv = LeakyReLU(a_src^T z_u + a_trg^T z_v)
```

and normalizes it over the neighborhood of `v`:

```text
alpha_uv = softmax_{u in N(v)}(e_uv)
```

Finally it aggregates:

```text
m_v = sum_{u in N(v)} alpha_uv * z_u
```

After that, the code applies:

- skip connection
- head concatenation or averaging
- bias
- activation on non-final layers

Relevant code:

- linear projection and scoring: [`gat.py`](gat.py) lines 82-103
- neighborhood softmax: [`gat.py`](gat.py) lines 133-180
- neighbor aggregation: [`gat.py`](gat.py) lines 182-193
- residual/concat/bias handling: [`gat.py`](gat.py) lines 235-259
- layer stack construction: [`gat.py`](gat.py) lines 271-295

### Wrapper architecture

The wrapper in `DenoiseGAT` chooses:

- `num_layers` graph layers
- 4 heads on intermediate layers if `hidden_dim >= 4`, otherwise 1
- 1 final head
- dropout `0.0` in the wrapper

The output of the GAT stack is a hidden vector per node, not the final noise directly.

Relevant code:

- head count and hidden sizing: [`diffusion.py`](diffusion.py) lines 84-98

### Global feature path status

The constructor still defines a pooled global feature path:

- `global_head`
- `global_gate`

Relevant code:

- [`diffusion.py`](diffusion.py) lines 99-106

If `use_global_features` is `false`, the model predicts the per-node diffusion target directly from `node_hidden`:

```text
u_hat_k = MLP(h_k^(L))
```

If `use_global_features` is `true`, the model pools a graph summary, gates it back into each node, and predicts the diffusion target from the concatenated readout:

```text
g = global_head(mean_k h_k^(L))
gamma_k = sigmoid([h_k^(L) ; g])
u_hat_k = MLP([h_k^(L) ; gamma_k odot g])
```

Intuition:

- with `use_global_features = false`, the experiment is a pure node-level GAT readout
- with `use_global_features = true`, each node also receives a gated pooled graph summary before the final head

### Current concerns

1. The graph is still very local.
   Even with several layers, global structure still has to emerge mostly through multi-hop local propagation.

2. The optional mean-pooled global path is blunt.
   It can reduce bad outliers, but it can also shrink variance and pull generation toward average shapes if it dominates.

3. There are no explicit long-range edges.
   Opposite or multi-hop relationships are not encoded directly.

4. There are no edge features.
   Messages only know the graph topology, not relative displacement, edge length, or turning angle.

5. The model is not E(2)-equivariant.
   It still relies on preprocessing and learned invariances rather than built-in geometric symmetry handling.

## `DenoiseGCN`

Relevant code:

- Wrapper and graph assembly: [`diffusion.py`](diffusion.py) lines 172-257
- Base graph convolution: [`gcn.py`](gcn.py) lines 8-34

### Intuition

This is the simpler graph baseline. Instead of learning attention weights, each node just averages information from a fixed 3-node neighborhood:

- itself
- previous neighbor
- next neighbor

Then a small MLP turns the final hidden state into the predicted 2D diffusion target.

This makes the GCN easier to reason about, but also much more prone to oversmoothing.

### Exact features given to each node

Each node gets:

```text
h_k^(0) = [x_k, y_k, phi(t)]
```

So the GCN currently sees:

1. noisy `x`
2. noisy `y`
3. timestep embedding `phi(t)`

It does not currently see:

- cycle positional features
- edge features
- long-range edges

If `use_global_features` is enabled, it does receive a pooled graph summary after message passing before the final node head.

Relevant code:

- timestep MLP: [`diffusion.py`](diffusion.py) lines 184-188
- node feature concatenation: [`diffusion.py`](diffusion.py) lines 245-249

### Graph math

The adjacency matrix encodes:

```text
A_{k,k} = A_{k,k+1} = A_{k,k-1} = 1/3
```

for each vertex `k`.

The base graph convolution in [`gcn.py`](gcn.py) does:

```text
support = H W
output = A support
```

or equivalently:

```text
tilde_H = A H W
```

The wrapper then adds a residual path and nonlinearity:

```text
H^{(l+1)} = SiLU(A H^(l) W_l + R_l(H^(l)))
```

where `R_l` is:

- identity if widths match
- linear projection otherwise

Finally a per-node MLP predicts the 2D diffusion target:

```text
u_hat_k = MLP(h_k^(L))
```

Relevant code:

- adjacency indices and values: [`diffusion.py`](diffusion.py) lines 206-237
- residual graph stack: [`diffusion.py`](diffusion.py) lines 251-254
- node head: [`diffusion.py`](diffusion.py) lines 201-205 and 256-257
- base `GraphConvolution`: [`gcn.py`](gcn.py) lines 28-34

### Current concerns

1. Oversmoothing is built into the operator.
   Every layer mixes each node with a fixed local average.

2. Residuals help optimization, but not expressivity enough.
   They keep information alive, but the core message passing is still low-pass.

3. No positional features, and global context is optional rather than structural.
   That leaves the model weak on symmetry breaking and whole-polygon coordination.

4. Empirically it has been the least promising denoiser so far.
   This matches the theory: fixed averaging is a poor fit for detailed geometric denoising-target prediction.

## Why GNNs Can Be Counterproductive For Regular Hexagon Denoising

This section is about the current task specifically: denoising near-regular, fixed-size hexagons represented as ordered vertex coordinates.

### Short answer

For this task, the MLP has two major advantages:

1. it sees the entire polygon immediately
2. the polygon is tiny, fixed-size, and consistently ordered

Those two facts remove much of the reason to use a graph network in the first place.

### Why the MLP can be stronger on regular hexagons

For a hexagon, the input is only 12 numbers:

```text
[x_1, y_1, x_2, y_2, ..., x_6, y_6]
```

The MLP denoiser gets the whole 12D vector at once, so it can learn direct global rules such as:

- if the right side moves out, pull the opposite side into a matching place
- if one top vertex shifts, adjust the adjacent and opposite vertices consistently
- if the whole shape looks like a noisy hexagon, predict a coordinated 12D correction

That is easy for an MLP because every output coordinate can depend on every input coordinate immediately.

The GNN, by contrast, starts from per-node features and has to reconstruct those global relationships through message passing over a tiny cycle graph.

For 6 vertices, that can be a bad trade:

- the graph is too small for locality to be a strong advantage
- the ring is very symmetric
- many vertices look locally similar
- the model has to infer whole-shape coordination from repeated local updates

So in this regime, the graph inductive bias can be restrictive rather than helpful.

### Concrete hexagon example: what one hidden node vector means

Suppose vertex `k` is the upper-right corner of a noisy near-regular hexagon. Before message passing, its initial feature is:

```text
h_k^(0) = [x_k, y_k,
           sin(2*pi*k/6), cos(2*pi*k/6),
           sin(4*pi*k/6), cos(4*pi*k/6),
           phi(t)]
```

This is still mostly raw input information:

- where this vertex currently is
- where it sits on the ring
- what diffusion timestep the model is at

After one GAT layer, the hidden vector is no longer just `(x, y)` with neighbors appended. It becomes a learned feature vector of width `hidden_dim`, for example 256 channels:

```text
h_k^(1) in R^256
```

That vector is produced by:

1. projecting the current node features into a hidden space
2. projecting neighbor features into the same hidden space
3. attention-weighting the neighbors
4. summing those weighted neighbor messages
5. mixing that result with a skip connection from the current node

Relevant code:

- GAT projection and attention: [`gat.py`](gat.py) lines 82-120
- skip connection and output mixing: [`gat.py`](gat.py) lines 235-259

So the hidden vector is best thought of as a learned summary of things like:

- estimated local radius
- whether the left and right neighbors are symmetric
- whether the local corner angle looks too sharp or too flat
- whether this node is in the upper-right, lower-left, etc.
- how much denoising should happen at this timestep

These are not explicit named channels in the code. They are learned hidden coordinates. But that is the kind of information they can represent.

### Are neighbor messages added to `x, y`, or are there more features?

The answer is: there are more features.

The GNN does not usually keep working in raw `x, y` space after the first step. Instead:

- raw node features are projected into a hidden feature space
- neighbor information is aggregated in that hidden space
- the output hidden vector has width `hidden_dim`, not width 2

So for the current GAT:

```text
h_k^(0) in R^(2 + 4 + time_emb_dim)
h_k^(1), h_k^(2), ..., h_k^(L) in R^hidden_dim
```

The neighbors are not simply stuck onto the current `x, y`, and they are not just numerically added to `x, y` either. They are transformed, weighted, aggregated, and mixed into a new learned representation.

For the current GCN, the same high-level idea holds, but the aggregation is simpler:

```text
H^{(l+1)} = SiLU(A H^(l) W_l + R_l(H^(l)))
```

so neighbor information is averaged through the adjacency matrix and then mapped into hidden channels.

### Why this can still fail on the hexagon case

A near-regular hexagon has extremely repetitive local structure:

- every node has degree 2 on the cycle
- every local corner is similar
- every local edge pair is similar
- opposite-side coordination matters, but that is not directly local

So after message passing, many node embeddings can become too similar in the very regime where the model actually needs precise global coordination.

That is one way to say the graph bias is counterproductive here:

- it emphasizes local similarity
- the task actually needs easy global coordination across all 6 vertices
- the MLP gets that global coupling for free

### A polygon generation task where a GNN would be the right tool

A GNN becomes much more compelling when the polygon behaves like a sampled boundary curve rather than a tiny fixed-size object.

Example:

- polygons with 50 to 200 vertices
- variable numbers of vertices across samples
- smooth local deformations such as dents, bulges, bends, or locally sharp corners
- optional edge features like segment length or relative displacement

Examples of real tasks in that regime:

- cell boundary generation
- coastline-like contour generation
- object silhouette denoising from many boundary samples
- shape completion where local neighborhood geometry matters

Why a GNN is better there:

- local geometry is genuinely informative
- message passing lets the same local denoising rule apply around the boundary
- the model scales more naturally with larger `n`
- it can handle richer edge/node attributes than a flat MLP baseline

In short:

- regular hexagons are small enough that the MLP's immediate global view is often better
- large, locally structured contours are where a GNN starts to make much more sense

### What if polygon size varied?

If the task changed from "denoise fixed-size hexagons" to "denoise polygons with varying numbers of vertices and make them regular," then a GNN would usually become the better conceptual fit.

Why:

1. An MLP expects a fixed input width.
   If `n` changes, then the flattened polygon width `2n` changes too, so a plain MLP cannot naturally use one shared architecture across all polygon sizes.

2. A GNN naturally works at the node level.
   The same message-passing rule can be applied to a polygon with 5, 20, or 100 vertices, as long as you can build the graph structure for each polygon.

3. "Make this polygon locally and globally more regular" is a graph-shaped problem.
   Each vertex should become consistent with its neighbors, while the whole cycle should become more uniform overall. That is exactly the kind of structure a graph model is meant to express.

So if you wanted one model to handle many polygon sizes, the ranking would usually be:

- conceptually: GNN is more appropriate
- conceptually: plain MLP is less appropriate

### Current implementation status

That conceptual change is now reflected in the code:

- `DenoiseMLP` is still fixed-size
- `DenoiseGAT` and `DenoiseGCN` now support variable-size polygons through ragged graph batches

Implemented pieces:

- per-sample ragged graph batching
- dynamic cycle-edge construction per polygon
- positional features computed from each polygon's own `n`
- diffusion loss and sampling paths that operate on concatenated node tensors plus per-graph sizes

So the current codebase now matches the conceptual recommendation:

- fixed-size polygons: MLP, GAT, and GCN all work
- variable-size polygons: use GAT or GCN

### Would an MLP still ever be reasonable?

Only in a weaker sense.

If you padded every polygon to a maximum size and used masks, you could still force an MLP to operate on variable-size data. But compared with a graph model, that would usually be:

- less natural
- less parameter-efficient
- more brittle to changes in `n`
- less likely to generalize cleanly across polygon sizes

So for:

- fixed `n = 6` regular hexagons: MLP can easily win
- variable `n` regular polygons with one shared model: GNN is usually the better direction

### Is a GNN relevant to classifier guidance?

Not inherently.

Classifier guidance is a sampling strategy, not a denoiser architecture choice. The core idea is:

- train a classifier or conditional model on noisy inputs `x_t`
- at sampling time, use the gradient of that model with respect to `x_t`
- bias the reverse diffusion trajectory toward the desired condition

That mechanism does not require a GNN. You could do classifier guidance with:

- an MLP classifier
- a CNN
- a Transformer
- a GNN

The GNN only matters if the classifier itself would benefit from graph structure.

So for your polygon setup:

- classifier guidance is compatible with a GNN denoiser
- classifier guidance is also compatible with an MLP denoiser
- using guidance does not by itself make a GNN more necessary

The architectural question and the guidance question are mostly orthogonal.

## `build_denoiser(...)`

Relevant code:

- [`diffusion.py`](diffusion.py) lines 260-288

### Intuition

This is the switchboard that keeps the rest of the training and sampling code architecture-agnostic.

### Behavior

It reads:

- `model.type`
- `hidden_dim`
- `time_emb_dim`
- `num_layers`

and returns one of:

- `DenoiseMLP`
- `DenoiseGAT`
- `DenoiseGCN`

If `model.type` is missing, it defaults to `gat`.

## `DiffusionConfig`

Relevant code:

- [`diffusion.py`](diffusion.py) lines 292-296

This stores:

- `n_steps`
- `beta_start`
- `beta_end`
- `prediction_target`

Those determine the linear beta schedule and how the denoiser output is interpreted by the DDPM wrapper.

## `Diffusion`

Relevant code:

- schedule setup: [`diffusion.py`](diffusion.py) lines 299-339
- forward noising: [`diffusion.py`](diffusion.py) lines 347-353
- reverse step: [`diffusion.py`](diffusion.py) lines 358-379
- full sampling loop: [`diffusion.py`](diffusion.py) lines 381-425
- training loss: [`diffusion.py`](diffusion.py) lines 427-455

### Intuition

The `Diffusion` class is the outer algorithm around the denoiser:

- it decides how much noise to add during training
- it turns the denoiser output into a reverse diffusion step at sampling time
- it computes training statistics for debugging

The denoiser predicts a configurable diffusion target, but the `Diffusion` wrapper is what makes that prediction useful for DDPM training and ancestral sampling.

### Forward process

The forward process adds noise in closed form:

```text
x_t = sqrt(bar_alpha_t) * x_0 + sqrt(1 - bar_alpha_t) * eps
```

This is implemented in `q_sample(...)`.

### Reverse process

The model output is first interpreted as either:

```text
u_theta(x_t, t) = x0_theta(x_t, t)      # if prediction_target = x0
u_theta(x_t, t) = eps_theta(x_t, t)     # if prediction_target = epsilon
```

If the checkpoint is epsilon-trained, the wrapper converts it to a clean-sample estimate:

```text
x0_theta(x_t, t) = (x_t - sqrt(1 - bar_alpha_t) * eps_theta(x_t, t)) / sqrt(bar_alpha_t)
```

Then DDPM uses the posterior mean induced by `x0_theta`:

```text
mu_theta(x_t, t) =
    [beta_t * sqrt(bar_alpha_{t-1}) / (1 - bar_alpha_t)] * x0_theta(x_t, t)
  + [sqrt(alpha_t) * (1 - bar_alpha_{t-1}) / (1 - bar_alpha_t)] * x_t
```

Sampling uses:

```text
x_{t-1} = mu_theta(x_t, t) + sqrt(tilde_beta_t) * z
```

At `t = 0`, `tilde_beta_0 = 0`, so this reduces to the mean automatically.

### Training loss

Training samples a random timestep and random Gaussian noise, constructs `x_t`, and minimizes one of:

```text
L_x0  = E[ || x_0 - x0_theta(x_t, t) ||^2 ]     # default
L_eps = E[ || eps - eps_theta(x_t, t) ||^2 ]    # alternate
```

The code also logs:

- `t_mean`, `t_std`
- `x_t_std`
- `noise_std`
- `model_output_std`
- `eps_pred_std`
- `x0_pred_mse`

Those are useful for diagnosing collapse, under-dispersion, or weak denoising.

## Current Architecture Summary

If you want the shortest possible summary:

- `MLP`: sees the whole polygon at once, but has no graph structure.
- `GAT`: best current graph denoiser, uses coordinates + cycle harmonics + time embedding + gated pooled global feature.
- `GCN`: simpler graph baseline, but fixed neighborhood averaging makes it prone to oversmoothing.

## Likely Next Improvements

The next upgrades that fit this codebase naturally are:

1. Long-range edges
   Intuition: let nodes talk directly to opposite or multi-hop vertices.
   Concern: can overconstrain a tiny graph.

2. Better global context
   Intuition: use a global token or attention pooling instead of blunt mean pooling.
   Concern: can still collapse diversity if the global path dominates.

3. Edge features
   Intuition: let messages depend on geometry, not just adjacency.
   Concern: requires careful normalization and invariance choices.

4. Equivariant geometry models
   Intuition: build rotational / translational structure into the model.
   Concern: more complex and needs clean integration with existing preprocessing.
