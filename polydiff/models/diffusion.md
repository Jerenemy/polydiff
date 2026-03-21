# `polydiff/models/diffusion.py` Reference

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
- `eps_theta(x_t, t)`: model prediction of that noise
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

This is the baseline that ignores graph structure. It treats the whole polygon as one long coordinate vector and asks a standard MLP to infer the noise directly from that vector plus the timestep.

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
eps_theta(x_t, t) = f_MLP([x_t ; phi(t)])
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

### Gated pooled global feature

After message passing, the model adds a graph-level summary:

```text
g = psi(mean_k h_k^(L))
```

where `psi` is `global_head`.

Then each node decides how much of that summary to use:

```text
gamma_k = sigmoid(W_g [h_k^(L) ; g])
g_k = gamma_k odot g
```

Finally:

```text
eps_hat_k = MLP([h_k^(L) ; g_k])
```

Intuition:

- `mean_k h_k^(L)` asks: what is the average hidden state of this whole polygon?
- `g` is a coarse whole-graph summary
- `gamma_k` is a gate that stops every node from using the global summary equally
- the final per-node MLP turns `[local hidden ; gated global]` into a 2D noise vector

Relevant code:

- global head: [`diffusion.py`](diffusion.py) lines 99-102
- global gate: [`diffusion.py`](diffusion.py) lines 103-106
- node head: [`diffusion.py`](diffusion.py) lines 107-111
- forward pass for pooled global feature and gate: [`diffusion.py`](diffusion.py) lines 162-169

### Current concerns

1. The graph is still very local.
   Even with several layers, global structure still has to emerge mostly through multi-hop local propagation.

2. Mean-pooling is blunt.
   It can suppress catastrophic outliers, but it can also shrink sample variance and pull generation toward average shapes.

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

Then a small MLP turns the final hidden state into the predicted 2D noise.

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
- pooled global graph features
- edge features
- long-range edges

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

Finally a per-node MLP predicts the 2D noise:

```text
eps_hat_k = MLP(h_k^(L))
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

3. No positional features and no global context.
   That leaves the model weak on symmetry breaking and whole-polygon coordination.

4. Empirically it has been the least promising denoiser so far.
   This matches the theory: fixed averaging is a poor fit for detailed noise prediction.

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

Those determine the linear beta schedule used by the DDPM wrapper.

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

The denoiser predicts `eps`, but the `Diffusion` wrapper is what makes that prediction useful for DDPM training and ancestral sampling.

### Forward process

The forward process adds noise in closed form:

```text
x_t = sqrt(bar_alpha_t) * x_0 + sqrt(1 - bar_alpha_t) * eps
```

This is implemented in `q_sample(...)`.

### Reverse process

The model predicts `eps_theta(x_t, t)`, and DDPM converts that into the reverse mean:

```text
mu_theta(x_t, t) = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - bar_alpha_t)) * eps_theta(x_t, t))
```

Then for `t > 0`:

```text
x_{t-1} = mu_theta(x_t, t) + sqrt(tilde_beta_t) * z
```

For `t = 0`, the code returns the mean directly.

### Training loss

Training samples a random timestep and random Gaussian noise, constructs `x_t`, and minimizes:

```text
L_simple = E[ || eps - eps_theta(x_t, t) ||^2 ]
```

The code also logs:

- `t_mean`, `t_std`
- `x_t_std`
- `noise_std`
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
