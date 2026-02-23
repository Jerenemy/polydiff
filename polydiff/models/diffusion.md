# `polydiff/models/diffusion.py` Reference

This document explains each class and function in `polydiff/models/diffusion.py`, including purpose, tensor shapes, and the DDPM math it implements.

## Notation

- `B`: batch size
- `D`: flattened data dimension (for polygons, typically `2 * n_vertices`)
- `t`: diffusion timestep index in `{0, 1, ..., T-1}`
- `T`: total number of diffusion steps (`n_steps`)
- `beta_t`: forward-process variance schedule
- `alpha_t = 1 - beta_t`
- `bar_alpha_t = prod_{s=0}^t alpha_s`

All vectors are assumed row-major batched tensors:
- `x0, x_t, noise`: shape `(B, D)`
- `t`: shape `(B,)`

## `class SinusoidalPosEmb(nn.Module)`

### Purpose
Convert scalar timesteps into continuous embeddings that let the denoiser condition on diffusion time.

### `__init__(dim: int)`
- Stores embedding output dimension `dim`.

### `forward(t: torch.Tensor) -> torch.Tensor`
Input:
- `t`: `(B,)`, integer or float timesteps.

Output:
- `(B, dim)` sinusoidal embedding.

Math:
1. Let `half = floor(dim / 2)`.
2. Construct frequencies
   `omega_i = exp(-log(10000) * i / (half - 1))`, for `i = 0 ... half-1`.
3. Form phase matrix `phi_{b,i} = t_b * omega_i`.
4. Return concatenation
   `[sin(phi), cos(phi)]` along feature axis.
5. If `dim` is odd, append one zero feature to keep exact width.

Interpretation:
- Lower `i` gives slower oscillations, higher `i` gives faster oscillations.
- This is the standard transformer-style positional/time embedding adapted for diffusion timesteps.

## `class DenoiseMLP(nn.Module)`

### Purpose
Predict additive Gaussian noise `epsilon` from noisy sample `x_t` and timestep `t`.

### `__init__(data_dim: int, hidden_dim: int, time_emb_dim: int, num_layers: int)`
Builds two parts:
1. `time_mlp`:
   - `SinusoidalPosEmb(time_emb_dim)`
   - `Linear(time_emb_dim -> time_emb_dim)`
   - `SiLU`
2. `net`:
   - Input width: `data_dim + time_emb_dim`
   - `num_layers` hidden blocks (`Linear -> SiLU`) with width `hidden_dim`
   - Final `Linear(hidden_dim -> data_dim)`

Notes:
- `max(1, num_layers)` guarantees at least one hidden layer.

### `forward(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor`
Input:
- `x`: `(B, data_dim)` noisy sample.
- `t`: `(B,)` timestep.

Output:
- `(B, data_dim)` predicted noise `epsilon_theta(x_t, t)`.

Computation:
1. `emb = time_mlp(t)` -> `(B, time_emb_dim)`
2. Concatenate `[x, emb]` -> `(B, data_dim + time_emb_dim)`
3. Feed through `net`.

## `@dataclass DiffusionConfig`

### Purpose
Container for scalar hyperparameters controlling the diffusion schedule.

Fields:
- `n_steps: int = 1000` -> total diffusion steps `T`
- `beta_start: float = 1e-4` -> `beta_0`
- `beta_end: float = 2e-2` -> `beta_{T-1}`

Schedule:
- `beta_t` is linearly interpolated from `beta_start` to `beta_end`.

## `class Diffusion`

### Purpose
Wrap DDPM forward process, reverse sampling, and training objective around a noise-predicting model.

### `__init__(model: nn.Module, config: DiffusionConfig, device: Optional[torch.device] = None)`
Initializes model and precomputes schedule tensors on `device`.

Precomputed quantities:
1. `betas = linspace(beta_start, beta_end, T)`
2. `alphas = 1 - betas`
3. `alphas_cumprod[t] = bar_alpha_t = prod_{s=0}^t alpha_s`
4. `alphas_cumprod_prev[t] = bar_alpha_{t-1}`, with `bar_alpha_{-1} := 1`
5. `sqrt_alphas_cumprod[t] = sqrt(bar_alpha_t)`
6. `sqrt_one_minus_alphas_cumprod[t] = sqrt(1 - bar_alpha_t)`
7. `sqrt_recip_alphas[t] = 1 / sqrt(alpha_t)`
8. Posterior variance
   `posterior_variance[t] = beta_t * (1 - bar_alpha_{t-1}) / (1 - bar_alpha_t)`

These are the standard DDPM closed-form coefficients.

### `q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor`
Purpose:
- Sample forward noised data `x_t` from clean data `x_0` at arbitrary timestep `t`.

Math:
- `q(x_t | x_0) = N(sqrt(bar_alpha_t) * x_0, (1 - bar_alpha_t) I)`
- Implemented as
  `x_t = sqrt(bar_alpha_t) * x_0 + sqrt(1 - bar_alpha_t) * noise`,
  where `noise ~ N(0, I)`.

### `predict_eps(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor`
Purpose:
- Thin wrapper calling the denoiser model:
  `epsilon_theta = model(x_t, t)`.

### `p_sample(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor`
Purpose:
- One reverse diffusion step `x_t -> x_{t-1}`.

DDPM reverse parameterization:
- `p_theta(x_{t-1} | x_t) = N(mu_theta(x_t, t), tilde_beta_t I)`
- With epsilon-prediction form:
  `mu_theta = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - bar_alpha_t)) * epsilon_theta(x_t, t))`
- `tilde_beta_t` is `posterior_variance[t]`.

Code mapping:
1. Gather timestep coefficients (`beta_t`, `1/sqrt(alpha_t)`, `sqrt(1-bar_alpha_t)`).
2. Compute `eps_pred = predict_eps(x_t, t)`.
3. Compute `model_mean` via the equation above.
4. If timestep is zero, return mean (deterministic final step).
5. Else sample
   `x_{t-1} = model_mean + sqrt(tilde_beta_t) * z`, `z ~ N(0, I)`.

Implementation detail:
- The `if t.min().item() == 0` branch assumes the whole batch shares the same timestep (true in `p_sample_loop`).

### `p_sample_loop(shape: tuple[int, int]) -> torch.Tensor`
Purpose:
- Generate fresh samples from pure Gaussian noise by iterating reverse steps from `T-1` to `0`.

Algorithm:
1. Start `x ~ N(0, I)` with given `shape`.
2. For `step = T-1, ..., 0`:
   - Create `t = [step, ..., step]` of shape `(B,)`.
   - Apply `x = p_sample(x, t)`.
3. Return final `x` as generated sample.

This implements ancestral DDPM sampling.

### `loss(x0: torch.Tensor) -> torch.Tensor`
Purpose:
- Training objective for epsilon-prediction DDPM.

Math:
1. Sample random timestep `t ~ Uniform({0,...,T-1})` independently per sample.
2. Sample `epsilon ~ N(0, I)`.
3. Construct
   `x_t = sqrt(bar_alpha_t) * x_0 + sqrt(1 - bar_alpha_t) * epsilon`.
4. Predict `epsilon_theta(x_t, t)`.
5. Optimize simple loss:
   `L_simple = E[ ||epsilon - epsilon_theta(x_t, t)||_2^2 ]`.

Code uses `nn.functional.mse_loss(eps_pred, noise)`, which is the standard simplified DDPM objective.

## Architecture Notes: GNN and Equivariance

### Does the current model use a GNN?
No. The denoiser is `DenoiseMLP`, a fully connected MLP over a flattened polygon vector `(B, D)`.
It does not perform message passing over polygon vertices/edges.

### What this implies
- Vertex structure is implicit (through fixed coordinate ordering), not explicit graph structure.
- The model is not permutation-equivariant to vertex reindexing.
- The model is not explicitly SE(2)/E(2)-equivariant (translations/rotations/reflections), although data preprocessing (centering, RMS scaling, CCW ordering) removes some nuisance variation.

### Should this be a GNN?
For fixed-`n`, consistently ordered polygons, this MLP baseline is valid and simple.
If sample quality, stability, or data efficiency becomes a bottleneck, a graph-based denoiser is usually a better inductive bias:
- Nodes: vertices
- Edges: polygon cycle (and optionally long-range edges)
- Outputs: per-vertex 2D noise vectors

### Should it be equivariant?
Usually yes for geometric generation tasks.
An E(2)-equivariant denoiser can improve generalization and reduce the burden of learning rotational/reflection symmetries from data alone.

Practical guidance:
1. Keep current MLP for baseline/debug speed.
2. Next upgrade: cycle-graph GNN denoiser.
3. If geometry fidelity matters, move to an E(2)-equivariant GNN/EGNN-style denoiser.

## End-to-End View

Training:
1. Corrupt clean samples with known Gaussian noise at random timestep.
2. Train model to recover that noise.

Sampling:
1. Start from Gaussian noise.
2. Apply learned reverse transition repeatedly to denoise.

Core assumption:
- Data distribution can be recovered by learning reverse-time dynamics of a fixed forward Gaussian diffusion process.
