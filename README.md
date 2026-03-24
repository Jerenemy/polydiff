# Polydiff

DDPM diffusion models for generating 2D polygons.

The repo supports three denoisers behind one training and sampling pipeline:

- `mlp`: fixed-size flattened-vector baseline
- `gat`: graph attention denoiser over the polygon cycle graph
- `gcn`: graph convolution denoiser over the same cycle graph

The important new split is:

- `mlp` still requires a fixed number of vertices
- `gat` and `gcn` now support both fixed-size and variable-size polygons without padded storage

Variable-size polygon data is stored raggedly as concatenated `coords` plus a per-polygon `num_vertices` array. During training and sampling, GAT/GCN batches are built as concatenated node sets with per-graph sizes, so message passing operates on the natural polygon graphs rather than masked dense tensors.

## Structure

- `polydiff/`: importable package code
- `configs/`: YAML configs for training and sampling
- `scripts/`: CLI wrappers
- `data/`: raw datasets and generated samples, organized per run under `data/processed/run_*`
- `models/`: local checkpoints, logs, and config snapshots, organized per run under `models/run_*`
- `pretrained_models/`: externally trained checkpoints
- `notebooks/`: exploratory analysis
- `tests/`: pytest tests

## Variable-Size GNN Pipeline

For the full design and file-format details of the new ragged GAT/GCN path, see:

- [`docs/variable_size_pipeline.md`](docs/variable_size_pipeline.md)

## Quick Start

1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

2. Generate polygons

```bash
python -m polydiff.data.gen_polygons --n 6 --num 10000 --out hexagons.npz
```

Variable-size example:

```bash
python -m polydiff.data.gen_polygons \
  --size_values 5,6,7,8 \
  --size_probabilities 0.1,0.2,0.4,0.3 \
  --num 10000 \
  --out mixed_polygons.npz
```

Important details about the generated data:

- Each polygon is generated from a random start angle `theta0`.
- Vertices are then placed around the cycle with smooth radial and angular perturbations.
- The raw generator recenters each polygon, RMS-normalizes its scale, and enforces CCW orientation.
- The generator does not anchor the polygon to a fixed start vertex.

Control how irregular the polygons are:

- Lower `--radial_sigma` and `--angle_sigma` for more regular polygons.
- Raise them for more irregular polygons.
- Increase `--smooth_passes` for smoother deformations.

Examples:

```bash
# cleaner polygons
python -m polydiff.data.gen_polygons \
  --n 6 --num 10000 --out hexagons_clean.npz \
  --radial_sigma 0.08 --angle_sigma 0.04 --smooth_passes 5

# rougher polygons
python -m polydiff.data.gen_polygons \
  --n 6 --num 10000 --out hexagons_noisy.npz \
  --radial_sigma 0.30 --angle_sigma 0.20 --smooth_passes 1 --deform_dist uniform

# variable-size mixture
python -m polydiff.data.gen_polygons \
  --size_values 5,6,7,8 \
  --size_probabilities 0.1,0.2,0.4,0.3 \
  --num 10000 --out mixed_noisy.npz \
  --radial_sigma 0.30 --angle_sigma 0.20 --smooth_passes 1 --deform_dist uniform
```

3. Plot polygons

```bash
python -m polydiff.data.plot_polygons hexagons.npz --num 16
```

Notes:

- Training `.npz` files usually contain stored `score`, so polygons are colored by score by default.
- Use `--show_scores` to render numeric score labels.

4. Run tests

```bash
.venv/bin/pytest
```

## Training

```bash
python -m polydiff.training.train --config configs/train_diffusion.yaml
```

The training config now chooses the denoiser explicitly:

```yaml
model:
  type: gat   # mlp | gat | gcn
  hidden_dim: 256
  time_emb_dim: 64
  num_layers: 3
```

Architecture/data compatibility:

- `mlp`: fixed-size datasets only
- `gat` / `gcn`: fixed-size dense datasets and variable-size ragged datasets

For variable-size GAT/GCN training, batches are collated as graph-native ragged node lists instead of padded `(B, V_max, 2)` tensors.

Training writes:

- `models/run_####__.../train.log`: human-readable logs
- `models/run_####__.../train_metrics.jsonl`: structured per-step metrics
- `models/run_####__.../config.source.yaml`: original config snapshot
- `models/run_####__.../config.resolved.yaml`: resolved config actually used
- `models/run_####__.../run_metadata.json`: run metadata including linked output folders
- `models/run_####__.../diffusion_final.pt`: final checkpoint

Checkpoints store:

- model weights
- diffusion schedule
- `model_cfg`
- `n_vertices` for fixed-size runs
- `max_vertices` for GAT/GCN checkpoint reconstruction
- training vertex-count histogram inside the training-data summary
- `run_name`
- training data path
- training data summary statistics

Each training invocation creates a new numbered run directory. The suffix is generated from the experiment name, model type, and dataset stem, so runs stay sortable but still human-readable.

That means sampling rebuilds the same denoiser architecture from the checkpoint automatically. You do not have to restate `model.type` in `configs/sample_diffusion.yaml`; the checkpoint drives reconstruction.

Logged metrics include:

- loss
- EMA loss
- gradient norm
- parameter norm
- timestep statistics
- `eps_pred_std`
- `noise_std`
- `x0_pred_mse`

Optional expensive generative diagnostics can be enabled during training with `training.sample_diagnostics_every`.

Relevant files:

- [`configs/train_diffusion.yaml`](configs/train_diffusion.yaml)
- [`polydiff/training/train.py`](polydiff/training/train.py)
- [`polydiff/models/diffusion.py`](polydiff/models/diffusion.py)
- [`polydiff/data/polygon_dataset.py`](polydiff/data/polygon_dataset.py)

## Sampling

```bash
python -m polydiff.sampling.sample --config configs/sample_diffusion.yaml
```

By default, sampling uses the most recent `models/run_*` directory. You can override that with:

```bash
python -m polydiff.sampling.sample --run 7
python -m polydiff.sampling.sample --run run_0007__polydiff-train-gat-hexagons-noisy
```

The sampling config can still point to an explicit checkpoint or run if needed:

```yaml
model:
  # checkpoint: "models/run_0007__.../diffusion_final.pt"
  # run: "run_0007__..."
```

Variable-size GAT/GCN sampling is controlled by `sampling.size_distribution`:

```yaml
sampling:
  num_samples: 10000
  size_distribution:
    values: [5, 6, 7, 8]
    probabilities: [0.1, 0.2, 0.4, 0.3]
```

Sampling behavior by model family:

- `mlp`: always samples one fixed size matching the checkpoint
- `gat` / `gcn`: if `size_distribution` is omitted, sampling defaults to the empirical training-size histogram stored in the checkpoint
- sampled polygon size stays fixed for each sample throughout the reverse diffusion trajectory

Sampling writes into a numbered sampling subfolder under the model run:

- `data/processed/run_####__.../sample_0001__.../samples.npz`
- `data/processed/run_####__.../sample_0001__.../samples.diagnostics.json`
- `data/processed/run_####__.../sample_0001__.../media/animations/sample_0000.gif` and friends when animation export is enabled
- `data/processed/run_####__.../sample_0001__.../media/notebooks/compare_polygon_distributions/*.png` for saved notebook comparison figures

If you sample from the same model run multiple times, each invocation gets a new `sample_####__...` directory. That keeps unguided, guided, and parameter-sweep sampling outputs from overwriting each other.

The diagnostics JSON compares generated samples against the training reference distribution when that reference is available from the checkpoint or explicitly configured.

Sample file formats:

- fixed-size output: dense `coords` with shape `(num_polygons, n_vertices, 2)` plus scalar `n`
- variable-size output: ragged `coords` with shape `(total_vertices, 2)` plus per-polygon `num_vertices`

Reduced-step sampling is supported with `sampling.n_steps`, but the current implementation uses naive schedule truncation rather than a respaced sampler. For architecture comparisons, full-step sampling is the cleanest reference.

Relevant files:

- [`configs/sample_diffusion.yaml`](configs/sample_diffusion.yaml)
- [`polydiff/sampling/sample.py`](polydiff/sampling/sample.py)
- [`polydiff/sampling/runtime.py`](polydiff/sampling/runtime.py)

## Guidance

The repo now supports optional sampling-time guidance. The denoiser still predicts noise as usual, but during reverse diffusion the sampler can add an extra gradient in `x`-space from a separate guidance model.

Intuition:

- the diffusion model says how to denoise
- the guidance model says which direction in polygon-space looks more desirable
- sampling combines both signals

The guidance hook lives inside the reverse DDPM step. In [`polydiff/models/diffusion.py`](polydiff/models/diffusion.py), `p_sample(...)` computes the usual reverse mean and then, if a guidance callback is present, shifts that mean by:

```text
model_mean <- model_mean + posterior_variance_t * guidance_grad(x_t, t)
```

That keeps the sampler generic: it only needs a gradient with respect to `x_t`, not a specific guidance architecture.

### Guidance Model Types

The current code supports four guidance modes:

- `classifier`
- `regressor`
- `regularity`
- `area`

The first two are time-conditioned MLPs over the flattened noisy polygon. They live in [`polydiff/models/guidance_models.py`](polydiff/models/guidance_models.py).
The last two are analytic, differentiable PyTorch objectives in [`polydiff/models/regularity_torch.py`](polydiff/models/regularity_torch.py).

Shared idea:

- input: noisy polygon `x_t`
- input: timestep `t`
- output: either class logits, a scalar score prediction, or a directly evaluated analytic geometry objective

Current limitations:

- the learned classifier/regressor guidance models are still fixed-size MLPs
- checkpoint-backed classifier/regressor guidance only supports uniform-size sampled batches
- analytic `regularity` and `area` guidance now work on mixed-size GAT/GCN batches
- guidance-model training is still fixed-size only
- the analytic regularity path is differentiable, but it still omits the discrete self-intersection test because that part of the original NumPy score is not autograd-friendly
- the analytic area path can easily encourage scale blow-up, because increasing polygon area is often easiest by just making the sample larger

### Classifier Guidance

Classifier guidance trains a model to predict a class from noisy polygons.

The current training path uses a threshold on the stored regularity score:

- class `1`: score above threshold
- class `0`: score below threshold

This is configured in [`configs/train_guidance_model.yaml`](configs/train_guidance_model.yaml) lines 23-25 with:

```yaml
labels:
  type: score_threshold
  threshold: 0.80
```

At sampling time, classifier guidance computes:

```text
guidance_grad(x_t, t) = scale * ∇_x log p_phi(y = target_class | x_t, t)
```

Relevant code:

- classifier-guidance gradient: [`polydiff/sampling/guidance.py`](polydiff/sampling/guidance.py)
- guidance checkpoint loading: [`polydiff/sampling/guidance.py`](polydiff/sampling/guidance.py)

Use this when you want a coarse target like:

- "make the polygon more regular than this threshold"
- "bias toward high-quality samples"

### Regressor Guidance

Regressor guidance trains a model to predict a continuous scalar, currently the polygon regularity score.

This is configured in [`configs/train_guidance_model.yaml`](configs/train_guidance_model.yaml) lines 23-25 with:

```yaml
labels:
  type: score_regression
```

At sampling time there are two modes:

1. If `target_value` is omitted:

```text
guidance_grad(x_t, t) = scale * ∇_x f_phi(x_t, t)
```

This simply pushes samples toward higher predicted score.

2. If `target_value` is provided:

```text
guidance_grad(x_t, t) = scale * ∇_x ( - (f_phi(x_t, t) - target_value)^2 )
```

This pushes samples toward a specific target score rather than just "higher is better."

Relevant code:

- regressor-guidance gradient: [`polydiff/sampling/guidance.py`](polydiff/sampling/guidance.py)
- regressor checkpoint loading: [`polydiff/sampling/guidance.py`](polydiff/sampling/guidance.py)

Use this when you want:

- continuous control over regularity
- the option to target a specific score rather than a thresholded class

### Analytic Regularity Guidance

Regularity guidance does not use a learned guidance network at all. Instead, it evaluates the polygon regularity score directly in PyTorch and backpropagates through that score with respect to `x_t`.

This uses the same smooth score shape as the current NumPy regularity function:

```text
score(x) = exp(-alpha * edge_cv(x) - beta * angle_cv(x) - gamma * radius_cv(x))
```

At sampling time there are again two modes:

1. If `target_value` is omitted:

```text
guidance_grad(x_t, t) = scale * ∇_x score(x_t)
```

2. If `target_value` is provided:

```text
guidance_grad(x_t, t) = scale * ∇_x ( - (score(x_t) - target_value)^2 )
```

This mode is useful when you want:

- a guidance signal without training a separate network
- a sanity-check baseline against learned regressor guidance
- direct control over the exact analytic regularity objective

Important caveat:

- the differentiable Torch score intentionally leaves out the discrete self-intersection rejection logic from data generation, because that part is not differentiable

Relevant code:

- differentiable score: [`polydiff/models/regularity_torch.py`](polydiff/models/regularity_torch.py)
- analytic guidance wrapper: [`polydiff/sampling/guidance.py`](polydiff/sampling/guidance.py)
- sampling config example: [`configs/sample_diffusion.yaml`](configs/sample_diffusion.yaml)

### Analytic Area Guidance

Area guidance also skips the learned network. It computes polygon area directly with a differentiable Torch shoelace formula and backpropagates that objective with respect to `x_t`.

By default it uses a smooth absolute area, so it still gives a usable gradient if the polygon orientation flips under noise:

```text
area(x) = sqrt(signed_area(x)^2 + eps)
```

At sampling time:

1. If `target_value` is omitted:

```text
guidance_grad(x_t, t) = scale * ∇_x area(x_t)
```

2. If `target_value` is provided:

```text
guidance_grad(x_t, t) = scale * ∇_x ( - (area(x_t) - target_value)^2 )
```

Important caveat:

- this objective does exactly what it says, which means it often rewards making the polygon larger rather than making it more regular

Relevant code:

- differentiable area: [`polydiff/models/regularity_torch.py`](polydiff/models/regularity_torch.py)
- analytic guidance wrapper: [`polydiff/sampling/guidance.py`](polydiff/sampling/guidance.py)

### Training A Guidance Model

The guidance training entrypoint is `train_guidance_model.py`, and it supports both classifier and regressor guidance models.

Run:

```bash
python -m polydiff.training.train_guidance_model --config configs/train_guidance_model.yaml
```

What it does:

1. Loads polygons from the dataset
2. Samples random diffusion timesteps
3. Corrupts clean polygons into `x_t` using the same forward diffusion schedule as the denoiser
4. Trains a time-conditioned guidance model on those noisy polygons

So the guidance model learns to operate on the same noisy inputs it will see during guided sampling.

Relevant code:

- training entrypoint: [`polydiff/training/train_guidance_model.py`](polydiff/training/train_guidance_model.py) lines 34-238
- task selection (`score_threshold` vs `score_regression`): [`polydiff/training/train_guidance_model.py`](polydiff/training/train_guidance_model.py) lines 53-68
- noisy-input construction with `q_sample(...)`: [`polydiff/training/train_guidance_model.py`](polydiff/training/train_guidance_model.py) lines 139-148

### Enabling Guidance During Sampling

Sampling config now supports:

```yaml
sampling:
  guidance:
    enabled: true
    kind: classifier   # classifier | regressor
    checkpoint: "models/classifier_final.pt"
    scale: 1.5
    target_class: 1    # classifier only
    # target_value: 0.95  # regressor only
```

Relevant code:

- config parsing: [`polydiff/sampling/runtime.py`](polydiff/sampling/runtime.py) lines 161-204
- request object wiring: [`polydiff/sampling/runtime.py`](polydiff/sampling/runtime.py) lines 207-253
- sampling entrypoint loading the guidance checkpoint: [`polydiff/sampling/sample.py`](polydiff/sampling/sample.py) lines 57-80
- example config block: [`configs/sample_diffusion.yaml`](configs/sample_diffusion.yaml) lines 14-23

Practical guidance:

- start with a small `scale`
- verify the guided distribution against the unguided baseline
- if you use regressor guidance, try both:
  - maximizing the predicted score
  - targeting a specific score with `target_value`

### Why The Interface Was Structured This Way

The sampler is deliberately decoupled from the guidance model architecture. It only consumes a gradient callback of the form:

```text
guidance_grad(x_t, t) -> tensor with shape x_t.shape
```

That means:

- today's fixed-size MLP classifier works
- today's fixed-size MLP regressor works
- a future variable-size GNN guidance model can reuse the same sampler interface

So the part that will likely change later is the guidance model builder, not the diffusion sampling loop.

## Diffusion Math

All three denoisers plug into the same DDPM wrapper in [`polydiff/models/diffusion.py`](polydiff/models/diffusion.py).

Intuition:

- Training picks a clean polygon `x_0`, adds a known amount of Gaussian noise, and asks the denoiser to predict exactly which noise was added.
- Sampling starts from pure noise and repeatedly subtracts the model's estimated noise.
- The denoiser architecture changes how the model reasons about polygon structure, but not the diffusion math around it.

Core equations:

```text
q(x_t | x_0) = N(sqrt(bar_alpha_t) * x_0, (1 - bar_alpha_t) I)
x_t = sqrt(bar_alpha_t) * x_0 + sqrt(1 - bar_alpha_t) * eps
eps ~ N(0, I)
```

Training objective:

```text
L_simple = E[ || eps - eps_theta(x_t, t) ||^2 ]
```

Reverse step used at sampling time:

```text
mu_theta(x_t, t) = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - bar_alpha_t)) * eps_theta(x_t, t))
x_{t-1} = mu_theta(x_t, t) + sqrt(tilde_beta_t) * z
z ~ N(0, I)
```

Relevant code:

- [`polydiff/models/diffusion.py`](polydiff/models/diffusion.py) lines 299-339 build the diffusion schedule.
- [`polydiff/models/diffusion.py`](polydiff/models/diffusion.py) lines 347-353 implement `q_sample(...)`.
- [`polydiff/models/diffusion.py`](polydiff/models/diffusion.py) lines 358-379 implement one reverse step `p_sample(...)`.
- [`polydiff/models/diffusion.py`](polydiff/models/diffusion.py) lines 427-455 implement the training loss and diagnostic stats.

## Denoisers

All denoisers implement the same interface:

- input `x_t`: shape `(B, D)` where `D = 2 * n_vertices`
- input `t`: shape `(B,)`
- output: predicted noise with shape `(B, D)`

The diffusion wrapper itself does not care which denoiser is underneath. The architectural differences are entirely in how the model converts `(x_t, t)` into a noise prediction.

### `DenoiseMLP`

Relevant code:

- [`polydiff/models/diffusion.py`](polydiff/models/diffusion.py) lines 34-57

Intuition:

- Treat the polygon as one long vector.
- Give the network the noisy coordinates and a description of "what timestep am I at?"
- Let a standard MLP learn the mapping from noisy polygon to predicted noise.

Math view:

```text
phi(t) = SiLU(W_t * SinusoidalPosEmb(t))
eps_hat = f_MLP([x_t ; phi(t)])
```

Features used:

1. Every noisy coordinate in the flattened polygon vector
2. One timestep embedding for the whole polygon

Architecture:

1. Build `phi(t)` with `SinusoidalPosEmb -> Linear -> SiLU`
2. Concatenate `[x_t ; phi(t)]`
3. Run `num_layers` hidden `Linear -> SiLU` blocks
4. Project back to `data_dim`

Why it is still useful:

- simplest baseline
- full global receptive field immediately
- easy to debug against the graph models

Current limitations:

- no explicit graph structure
- no local neighborhood bias
- depends on stable vertex ordering
- not permutation-equivariant
- not E(2)-equivariant

### `DenoiseGAT`

Relevant code:

- Wrapper and feature construction: [`polydiff/models/diffusion.py`](polydiff/models/diffusion.py) lines 59-169
- Attention implementation: [`polydiff/models/gat.py`](polydiff/models/gat.py) lines 6-303

Intuition:

- Turn the polygon into a ring graph where each vertex can talk to itself and its two immediate neighbors.
- Give each vertex its current noisy coordinates, its place on the ring, and the diffusion timestep.
- Use attention to decide which neighboring messages matter more for the current denoising step.
- After local message passing, add a coarse whole-polygon summary and let each node decide how much of that global context to use.

Math view:

Initial node features for vertex `k`:

```text
h_k^(0) = [x_k, y_k,
           sin(2*pi*k/n), cos(2*pi*k/n),
           sin(4*pi*k/n), cos(4*pi*k/n),
           phi(t)]
```

where `phi(t)` is the shared timestep embedding.

Cycle graph:

```text
E = {(k, k), (k, k+1 mod n), (k, k-1 mod n)}
```

Inside one GAT layer, with source node `u` and target node `v`:

```text
z_u = W h_u
e_uv = LeakyReLU(a_src^T z_u + a_trg^T z_v)
alpha_uv = softmax_{u in N(v)}(e_uv)
m_v = sum_{u in N(v)} alpha_uv * z_u
```

Then the implementation applies skip connections and head concat or averaging inside the layer.

After the GAT stack:

```text
g = psi(mean_k h_k^(L))
gamma_k = sigmoid(W_g [h_k^(L) ; g])
eps_hat_k = MLP([h_k^(L) ; gamma_k odot g])
```

where `psi` is `global_head`, `gamma_k` is the node-wise gate, and the final MLP predicts 2 numbers per node.

Every feature currently given to the GAT:

1. Noisy `x` coordinate
2. Noisy `y` coordinate
3. `sin(2 * pi * k / n)`
4. `cos(2 * pi * k / n)`
5. `sin(4 * pi * k / n)`
6. `cos(4 * pi * k / n)`
7. The timestep embedding `phi(t)` repeated at every node
8. A pooled graph feature after message passing, injected back through the gated global path

Important architectural details:

- Intermediate layers use 4 heads when `hidden_dim >= 4`, otherwise 1 head.
- The final GAT layer uses 1 head and outputs a `hidden_dim` vector per node.
- Skip connections are already enabled inside the imported `GAT` implementation.
- The wrapper sets GAT dropout to `0.0`.
- There are still no explicit edge features.

Why these features help, intuitively:

- Coordinates tell the model what the noisy polygon currently looks like.
- Time embedding tells it how hard the denoising step should be.
- Cycle harmonics tell it where a node sits around the ring without tying that position to absolute world-space orientation.
- The pooled global feature gives a coarse "what kind of polygon is this overall?" summary.
- The gate stops that global summary from being used at full strength everywhere.

Current concerns and limitations:

- The topology is still only local cycle edges plus self-loops.
- Global shape has to emerge through multi-hop local propagation, except for the coarse pooled summary.
- The pooled summary can reduce outliers but also reduce variance if it dominates.
- There are no long-range edges, no opposite-vertex connections, and no edge geometry features.
- The positional features are fixed, not learned.
- The model is still not E(2)-equivariant.
- It still assumes stable cyclic ordering.

### `DenoiseGCN`

Relevant code:

- Wrapper and graph construction: [`polydiff/models/diffusion.py`](polydiff/models/diffusion.py) lines 172-257
- Base graph convolution: [`polydiff/models/gcn.py`](polydiff/models/gcn.py) lines 8-34

Intuition:

- This is the simplest graph baseline: each node repeatedly averages information from itself and its two immediate neighbors.
- Residuals keep the node state from being replaced entirely by that local average.
- A small per-node MLP makes the final `(x, y)` prediction after message passing.

Math view:

Initial node features:

```text
h_k^(0) = [x_k, y_k, phi(t)]
```

Adjacency:

```text
A_{k,k} = A_{k,k+1} = A_{k,k-1} = 1/3
```

One residual GCN block is:

```text
tilde_h^(l+1) = A h^(l) W_l + b_l
h^(l+1) = SiLU(tilde_h^(l+1) + R_l(h^(l)))
```

where `R_l` is either identity or a learned linear projection.

Final node prediction:

```text
eps_hat_k = MLP(h_k^(L))
```

Every feature currently given to the GCN:

1. Noisy `x` coordinate
2. Noisy `y` coordinate
3. The timestep embedding `phi(t)` repeated at every node

What it does not currently get:

- cycle positional features
- pooled global features
- edge features
- long-range edges

Architecture details:

- `num_layers` graph convolutions
- residual path around every graph convolution
- shared `SiLU` after each residual block
- per-node head `Linear -> SiLU -> Linear`

Current concerns and limitations:

- The core message passing is still a local averaging operator, so oversmoothing is the central risk.
- In current experiments this model has been much less successful than the GAT and often plateaued early.
- Residuals and the node MLP help, but they do not change the basic low-pass nature of the graph convolution.
- No positional features and no global path means the model has weak symmetry-breaking and weak whole-graph context.

## Denoiser Selection and Checkpoint Behavior

Training chooses the denoiser with `model.type`:

- `mlp`
- `gat`
- `gcn`

This is implemented through `build_denoiser(...)` in `polydiff/models/diffusion.py`.

The checkpoint stores `model_cfg`, and sampling reconstructs the denoiser from that stored config before loading weights. That makes the train and sample paths architecture-consistent by default.

One consequence is that architecture edits can make old checkpoints incompatible if parameter shapes change. That is expected for the recent GAT and GCN experiments.

## Current Architectural Concerns

Across all three denoisers, the main open issues are:

- no E(2)-equivariant geometry handling
- reliance on fixed cyclic ordering
- no explicit edge features
- no learned mechanism for long-range polygon relationships
- DDPM MSE training can reward safe average predictions even when sample diversity is too low

Architecture-specific concerns:

- `mlp`: weak inductive bias but full global access
- `gat`: better local structure modeling but still prone to either unstable tails or over-regularized samples depending on how global context is injected
- `gcn`: simplest graph baseline, but currently the least convincing denoiser because of oversmoothing

Relevant code:

- [`polydiff/models/diffusion.py`](polydiff/models/diffusion.py) lines 260-288 select the denoiser.
- [`polydiff/training/train.py`](polydiff/training/train.py) lines 52-54 use that builder during training.
- [`polydiff/sampling/runtime.py`](polydiff/sampling/runtime.py) lines 57-61 use the stored `model_cfg` during checkpoint loading.

## Plausible Next Improvements

These are the most natural next upgrades from the current codebase.

### 1. Long-range graph edges

Examples:

- connect opposite vertices for even `n`
- connect `k` to `k +/- 2`
- use a denser graph than the simple cycle

Why it may help:

- gives nodes more direct access to whole-polygon structure
- reduces the burden on multi-hop local propagation

Concerns:

- easy to overconstrain a tiny graph
- the right edge pattern may depend on `n`
- denser connectivity can wash out the interpretation of attention weights

### 2. Learnable global context

Examples:

- global token node
- attention pooling instead of mean pooling
- lower-dimensional global bottleneck

Why it may help:

- richer graph-level summary than blunt mean pooling
- could preserve diversity better than broadcasting a single large pooled vector

Concerns:

- still can collapse toward mean-shape behavior if the global path dominates
- introduces more moving parts into an already sensitive generative setup

### 3. Richer positional information

Examples:

- more harmonics
- learnable circular embeddings
- canonicalized start vertex plus learned index embeddings

Why it may help:

- lets the model coordinate vertex roles around the ring more precisely

Concerns:

- naive learned absolute index embeddings can encode fake semantics if the start vertex is arbitrary
- more positional structure can make the model less robust to reindexing

### 4. Edge features

Examples:

- relative displacement vectors
- edge length
- turning angle proxies
- normalized pairwise distance features

Why it may help:

- gives message passing explicit geometric relationships rather than topology alone

Concerns:

- feature design can accidentally leak preprocessing assumptions
- some edge features are not naturally invariant or equivariant without extra care

### 5. Equivariant geometry models

Examples:

- EGNN-style updates
- E(2)-equivariant message passing

Why it may help:

- better inductive bias for coordinate prediction
- less need to learn geometric symmetries from scratch

Concerns:

- more implementation complexity
- interaction with the repo's existing centering / scale normalization / canonicalization should be thought through carefully

### 6. Better normalization and residual design

Examples:

- layer norm or graph norm
- stronger residual mixing
- pre-norm attention blocks

Why it may help:

- could stabilize training and reduce outlier trajectories

Concerns:

- easy to add complexity without fixing the real missing inductive bias
- can improve optimization while still leaving sample quality bottlenecked by topology and features

## Debugging and Diagnostics

Use `notebooks/` for exploratory work. The repo also has automatic dataset and sample diagnostics for:

- regularity score
- edge, angle, and radius coefficient of variation
- compactness
- centering drift
- scale drift
- self-intersection rate

When looking at aligned diagnostic plots, remember that canonicalization intentionally anchors one vertex and rotates the polygon to a standard orientation. Those plots are useful for distribution comparison, but they are not raw geometry views.

Relevant code:

- [`polydiff/data/diagnostics.py`](polydiff/data/diagnostics.py) lines 21-41 define canonicalization used by many diagnostics.
