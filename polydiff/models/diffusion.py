"""Minimal DDPM components for polygon diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from ..data.polygon_dataset import PolygonGraphBatch, build_polygon_graph_batch
from .gat import GAT
from .gcn import GraphConvolution


class SinusoidalPosEmb(nn.Module):
    """this module does not learn anything, it just passes t through a sort of encoder"""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) int or float tensor -> (B, dim) embedding."""
        half = self.dim // 2
        device = t.device
        emb = torch.arange(half, device=device, dtype=torch.float32) # [0, 1, ..., half]
        emb = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * emb / (half - 1)) # perform operation on each item of vector
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0) # basically make a mtx by mult-ing t and emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1) # puts the mtx's next to each other (doesnt stack them)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class DenoiseMLP(nn.Module):
    def __init__(self, data_dim: int, hidden_dim: int, time_emb_dim: int, num_layers: int) -> None:
        super().__init__()
        # build a small network that turns the timestep t into a learnable embedded vector 
        # (think of this as like a kernel param in a cnn. it needs to have t embedded in a way that makes it easiest to predict the noise added)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),                 # fixed Fourier features of t transformed to each have time_emb_dim features: (Batch,) -> (Batch, time_emb_dim)
            nn.Linear(time_emb_dim, time_emb_dim),          # learnable affine mix of those features
            nn.SiLU(),                                      # nonlinearity so time features can be transformed flexibly
        )

        layers = []
        in_dim = data_dim + time_emb_dim                    # input will be [x_t concat time_emb], so dims add
        for i in range(max(1, num_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))    # affine map: R^{in_dim} -> R^{hidden_dim} (eg input vects from R^2 output to R^8)
            layers.append(nn.SiLU())                        # nonlinearity
            in_dim = hidden_dim                             # next layer expects hidden_dim inputs
        layers.append(nn.Linear(hidden_dim, data_dim))      # output predicted noise eps_hat with same dim as x_t
        self.net = nn.Sequential(*layers)                   # bundle all layers into one module

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(t)                              # compute time embedding from t: (B,) -> (B, time_emb_dim)
        h = torch.cat([x, emb], dim=1)                      # concatenate features: (B, data_dim + time_emb_dim)
        return self.net(h)                                  # predict eps_hat: (B, data_dim)


def _graph_batch_from_dense_flattened(
    x: torch.Tensor,
    *,
    expected_data_dim: int,
    num_vertices: int,
) -> tuple[PolygonGraphBatch, int]:
    if x.ndim != 2 or x.shape[1] != expected_data_dim:
        raise ValueError(f"x must have shape (batch, {expected_data_dim}), got {tuple(x.shape)}")
    batch_size = x.shape[0]
    coords = x.reshape(batch_size * num_vertices, 2)
    graph_batch = build_polygon_graph_batch(
        torch.full((batch_size,), num_vertices, dtype=torch.long, device=x.device),
        coords=coords,
    )
    return graph_batch, batch_size


def _graph_batch_mean(node_features: torch.Tensor, batch: PolygonGraphBatch) -> torch.Tensor:
    sums = torch.zeros(
        (batch.batch_size, node_features.shape[1]),
        dtype=node_features.dtype,
        device=node_features.device,
    )
    sums.index_add_(0, batch.graph_index, node_features)
    counts = batch.num_vertices.to(node_features.dtype).unsqueeze(1).clamp_min(1.0)
    return sums / counts


def _cycle_positional_features(batch: PolygonGraphBatch, *, dtype: torch.dtype) -> torch.Tensor:
    num_vertices_per_node = batch.num_vertices.index_select(0, batch.graph_index).to(dtype)
    node_index = batch.node_index.to(dtype)
    phase = node_index * (2.0 * torch.pi / num_vertices_per_node)
    return torch.stack(
        [
            torch.sin(phase),
            torch.cos(phase),
            torch.sin(2.0 * phase),
            torch.cos(2.0 * phase),
        ],
        dim=1,
    )


def _cycle_edge_index(batch: PolygonGraphBatch) -> torch.Tensor:
    total_vertices = batch.total_vertices
    if total_vertices == 0:
        return torch.empty((2, 0), dtype=torch.long, device=batch.coords.device)
    src = torch.arange(total_vertices, device=batch.coords.device, dtype=torch.long)
    offsets = batch.ptr[:-1].index_select(0, batch.graph_index)
    counts = batch.num_vertices.index_select(0, batch.graph_index)
    next_nodes = offsets + torch.remainder(batch.node_index + 1, counts)
    prev_nodes = offsets + torch.remainder(batch.node_index - 1, counts)
    return torch.stack(
        [
            torch.cat([src, src, src], dim=0),
            torch.cat([next_nodes, prev_nodes, src], dim=0),
        ],
        dim=0,
    )


def _cycle_adj(batch: PolygonGraphBatch, *, device: torch.device) -> torch.Tensor:
    indices = _cycle_edge_index(batch)
    values = torch.full((indices.shape[1],), 1.0 / 3.0, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=(batch.total_vertices, batch.total_vertices),
        device=device,
    ).coalesce()


class DenoiseGAT(nn.Module):
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        num_layers: int,
        use_global_features: bool = False,
    ) -> None:
        super().__init__()
        if data_dim % 2 != 0:
            raise ValueError(f"data_dim must be even for flattened 2D polygon coordinates, got {data_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.data_dim = data_dim
        self.coord_dim = 2
        self.num_vertices = data_dim // self.coord_dim
        self.time_emb_dim = time_emb_dim
        self.pos_enc_dim = 4
        self.use_global_features = bool(use_global_features)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        node_in_dim = self.coord_dim + self.pos_enc_dim + time_emb_dim
        hidden_heads = 4 if hidden_dim >= 4 else 1
        hidden_per_head = max(1, hidden_dim // hidden_heads)
        num_heads_per_layer = [hidden_heads] * max(0, num_layers - 1) + [1]
        num_features_per_layer = [node_in_dim]
        if num_layers > 1:
            num_features_per_layer.extend([hidden_per_head] * (num_layers - 1))
        num_features_per_layer.append(hidden_dim)

        self.gat = GAT(
            num_of_layers=num_layers,
            num_heads_per_layer=num_heads_per_layer,
            num_features_per_layer=num_features_per_layer,
            dropout=0.0,
        )
        if self.use_global_features:
            self.global_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            self.global_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            node_head_in_dim = hidden_dim * 2
        else:
            self.global_head = None
            self.global_gate = None
            node_head_in_dim = hidden_dim
        self.node_head = nn.Sequential(
            nn.Linear(node_head_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.coord_dim),
        )
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        if batch is None:
            graph_batch, batch_size = _graph_batch_from_dense_flattened(
                x,
                expected_data_dim=self.data_dim,
                num_vertices=self.num_vertices,
            )
            return_dense = True
            coords = graph_batch.coords
        else:
            graph_batch = batch
            batch_size = graph_batch.batch_size
            return_dense = False
            if x.ndim != 2 or x.shape != (graph_batch.total_vertices, self.coord_dim):
                raise ValueError(
                    f"x must have shape ({graph_batch.total_vertices}, {self.coord_dim}), got {tuple(x.shape)}"
                )
            coords = x

        if t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError(f"t must have shape ({batch_size},), got {tuple(t.shape)}")

        time_emb = self.time_mlp(t).index_select(0, graph_batch.graph_index)
        pos_enc = _cycle_positional_features(graph_batch, dtype=coords.dtype)
        node_features = torch.cat([coords, pos_enc, time_emb], dim=1)
        edge_index = _cycle_edge_index(graph_batch)

        node_hidden, _ = self.gat((node_features, edge_index))
        if self.use_global_features:
            global_features = self.global_head(_graph_batch_mean(node_hidden, graph_batch))
            global_nodes = global_features.index_select(0, graph_batch.graph_index)
            global_gate = self.global_gate(torch.cat([node_hidden, global_nodes], dim=1))
            node_readout = torch.cat([node_hidden, global_gate * global_nodes], dim=1)
        else:
            node_readout = node_hidden

        node_noise = self.node_head(node_readout)
        if return_dense:
            return node_noise.reshape(batch_size, self.data_dim)
        return node_noise


class DenoiseGCN(nn.Module):
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        num_layers: int,
        use_global_features: bool = False,
    ) -> None:
        super().__init__()
        if data_dim % 2 != 0:
            raise ValueError(f"data_dim must be even for flattened 2D polygon coordinates, got {data_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.data_dim = data_dim
        self.coord_dim = 2
        self.num_vertices = data_dim // self.coord_dim
        self.use_global_features = bool(use_global_features)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        dims = [self.coord_dim + time_emb_dim]
        dims.extend([hidden_dim] * num_layers)
        self.layers = nn.ModuleList(
            GraphConvolution(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        )
        self.residual_projs = nn.ModuleList(
            nn.Identity() if dims[i] == dims[i + 1] else nn.Linear(dims[i], dims[i + 1], bias=False)
            for i in range(len(dims) - 1)
        )
        self.activation = nn.SiLU()
        if self.use_global_features:
            self.global_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            self.global_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            node_head_in_dim = hidden_dim * 2
        else:
            self.global_head = None
            self.global_gate = None
            node_head_in_dim = hidden_dim
        self.node_head = nn.Sequential(
            nn.Linear(node_head_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.coord_dim),
        )
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        if batch is None:
            graph_batch, batch_size = _graph_batch_from_dense_flattened(
                x,
                expected_data_dim=self.data_dim,
                num_vertices=self.num_vertices,
            )
            return_dense = True
            coords = graph_batch.coords
        else:
            graph_batch = batch
            batch_size = graph_batch.batch_size
            return_dense = False
            if x.ndim != 2 or x.shape != (graph_batch.total_vertices, self.coord_dim):
                raise ValueError(
                    f"x must have shape ({graph_batch.total_vertices}, {self.coord_dim}), got {tuple(x.shape)}"
                )
            coords = x

        if t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError(f"t must have shape ({batch_size},), got {tuple(t.shape)}")

        time_emb = self.time_mlp(t).index_select(0, graph_batch.graph_index)
        h = torch.cat([coords, time_emb], dim=1)
        adj = _cycle_adj(graph_batch, device=coords.device)

        for i, layer in enumerate(self.layers):
            residual = self.residual_projs[i](h)
            h = layer(h, adj) + residual
            h = self.activation(h)

        if self.use_global_features:
            global_features = self.global_head(_graph_batch_mean(h, graph_batch))
            global_nodes = global_features.index_select(0, graph_batch.graph_index)
            global_gate = self.global_gate(torch.cat([h, global_nodes], dim=1))
            node_readout = torch.cat([h, global_gate * global_nodes], dim=1)
        else:
            node_readout = h

        node_noise = self.node_head(node_readout)
        if return_dense:
            return node_noise.reshape(batch_size, self.data_dim)
        return node_noise


def build_denoiser(*, data_dim: int, model_cfg: Mapping[str, Any] | None = None) -> nn.Module:
    cfg = {} if model_cfg is None else dict(model_cfg)
    model_type = str(cfg.get("type", "gat")).lower()
    hidden_dim = int(cfg.get("hidden_dim", 256))
    time_emb_dim = int(cfg.get("time_emb_dim", 64))
    num_layers = int(cfg.get("num_layers", 3))
    use_global_features = bool(cfg.get("use_global_features", False))

    if model_type == "mlp":
        return DenoiseMLP(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            num_layers=num_layers,
        )
    if model_type == "gat":
        return DenoiseGAT(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            num_layers=num_layers,
            use_global_features=use_global_features,
        )
    if model_type == "gcn":
        return DenoiseGCN(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            num_layers=num_layers,
            use_global_features=use_global_features,
        )
    raise ValueError(f"Unsupported model.type {model_type!r}; expected one of: mlp, gat, gcn")



@dataclass
class DiffusionConfig:
    n_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


GuidanceGradFn = Callable[..., torch.Tensor]


class Diffusion:
    def __init__(self, model: nn.Module, config: DiffusionConfig, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)                          # move model parameters/tensors onto chosen device

        # betas: noise schedule (variance added each step), linearly spaced from beta_start to beta_end
        betas = torch.linspace(config.beta_start, config.beta_end, config.n_steps, device=self.device)
        
        # alphas: remaining signal fraction per step (alpha_t = 1 - beta_t)
        alphas = 1.0 - betas
        
        # alphas_cumprod: ᾱ_t = Π_{s=1..t} α_s (product over time, amount of signal left after t steps)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # ᾱ_{t-1} with ᾱ_{-1} defined as 1.0 (used in posterior formulas)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]], 
            dim=0
        )

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod                # ᾱ_t
        self.alphas_cumprod_prev = alphas_cumprod_prev      # ᾱ_{t-1}
        
        # sqrt(ᾱ_t) coefficient used in x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        
        # sqrt(1-ᾱ_t) coefficient used in same closed-form forward diffusion
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        
        # sqrt(1/α_t) used in reverse mean computation (undoing the per-step scaling)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        
        # posterior_variance: Var[q(x_{t-1} | x_t, x_0)] for DDPM reverse sampling
        # β̃_t = β_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def _resolve_n_steps(self, n_steps: Optional[int]) -> int:
        steps = self.config.n_steps if n_steps is None else int(n_steps)
        if steps < 1 or steps > self.config.n_steps:
            raise ValueError(f"n_steps must be in [1, {self.config.n_steps}], got {steps}")
        return steps

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        *,
        graph_batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        """Forward diffusion (adding noise). Create x_t in in one shot"""
        if graph_batch is None:
            sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(1)              # gather sqrt(ᾱ_t) per sample, shape (B,1)
            sqrt_omab = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)  # gather sqrt(1-ᾱ_t) per sample, shape (B,1)
        else:
            node_t = t.index_select(0, graph_batch.graph_index)
            sqrt_ab = self.sqrt_alphas_cumprod[node_t].unsqueeze(1)
            sqrt_omab = self.sqrt_one_minus_alphas_cumprod[node_t].unsqueeze(1)
        
        # x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
        return sqrt_ab * x0 + sqrt_omab * noise

    def predict_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        graph_batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        if graph_batch is None:
            return self.model(x_t, t)  # eps_hat = model(x_t, t)
        return self.model(x_t, t, batch=graph_batch)

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        guidance_grad: GuidanceGradFn | None = None,
        graph_batch: PolygonGraphBatch | None = None,
    ) -> torch.Tensor:
        """Reverse diffusion step: sample x_{t-1} from learned p(x_{t-1} | x_t)."""
        if graph_batch is None:
            effective_t = t
        else:
            effective_t = t.index_select(0, graph_batch.graph_index)

        betas_t = self.betas[effective_t].unsqueeze(1)                        # gather β_t per sample, shape (...,1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[effective_t].unsqueeze(1) # gather sqrt(1/α_t), shape (...,1)
        sqrt_omab_t = self.sqrt_one_minus_alphas_cumprod[effective_t].unsqueeze(1) # gather sqrt(1-ᾱ_t), shape (...,1)
        posterior_var_t = self.posterior_variance[effective_t].unsqueeze(1)   # gather β̃_t per sample, shape (...,1)

        eps_pred = self.predict_eps(x_t, t, graph_batch=graph_batch)                         # model predicts ε_hat(x_t, t)
        
        # DDPM mean:
        # μθ(x_t,t) = (1/sqrt(α_t)) * (x_t - (β_t / sqrt(1-ᾱ_t)) * ε_hat)
        model_mean = sqrt_recip_alpha_t * (x_t - betas_t * eps_pred / sqrt_omab_t)

        if guidance_grad is not None:
            if graph_batch is None:
                grad = guidance_grad(x_t, t)
            else:
                grad = guidance_grad(x_t, t, graph_batch=graph_batch)
            if grad.shape != x_t.shape:
                raise ValueError(
                    f"guidance_grad must return shape {tuple(x_t.shape)}, got {tuple(grad.shape)}"
                )
            # Classifier guidance shifts the reverse mean in x-space using ∇_x log p(y | x_t).
            model_mean = model_mean + posterior_var_t * grad

        # At t=0, return the mean (no noise added in the final step)
        if t.min().item() == 0:
            return model_mean

        noise = torch.randn_like(x_t)        
        
        # Sample:
        # x_{t-1} = μθ + sqrt(β̃_t) * z# fresh Gaussian noise z ~ N(0,I)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    def _sample_loop(
        self,
        shape: tuple[int, int],
        *,
        steps: int,
        trajectory_indices: Sequence[int] | None = None,
        guidance_grad: GuidanceGradFn | None = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        tracked_indices: list[int] | None = None
        if trajectory_indices is not None:
            tracked_indices = [int(idx) for idx in trajectory_indices]
            for idx in tracked_indices:
                if not 0 <= idx < shape[0]:
                    raise ValueError(f"trajectory_index must be in [0, {shape[0] - 1}], got {idx}")

        self.model.eval()
        with torch.no_grad():
            x = torch.randn(shape, device=self.device)      # initialize x_T as standard Gaussian noise
            trajectory = None
            if tracked_indices is not None:
                trajectory = [x[tracked_indices].detach().cpu().clone()]
            for step in reversed(range(steps)):             # iterate t = steps-1, ..., 0
                t = torch.full((shape[0],), step, device=self.device, dtype=torch.long) # batch of identical t
                x = self.p_sample(x, t, guidance_grad=guidance_grad) # sample x_{t-1} from x_t
                if trajectory is not None:
                    trajectory.append(x[tracked_indices].detach().cpu().clone())

        if trajectory is None:
            return x, None
        return x, torch.stack(trajectory, dim=0)

    def _sample_graph_loop(
        self,
        graph_batch: PolygonGraphBatch,
        *,
        steps: int,
        trajectory_indices: Sequence[int] | None = None,
        guidance_grad: GuidanceGradFn | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        tracked_indices: list[int] | None = None
        if trajectory_indices is not None:
            tracked_indices = [int(idx) for idx in trajectory_indices]
            for idx in tracked_indices:
                if not 0 <= idx < graph_batch.batch_size:
                    raise ValueError(
                        f"trajectory_index must be in [0, {graph_batch.batch_size - 1}], got {idx}"
                    )

        self.model.eval()
        with torch.no_grad():
            x = torch.randn((graph_batch.total_vertices, 2), device=self.device)
            trajectories = None
            if tracked_indices is not None:
                trajectories = [[] for _ in tracked_indices]
                for list_index, graph_index in enumerate(tracked_indices):
                    trajectories[list_index].append(x[graph_batch.graph_slice(graph_index)].detach().cpu().clone())

            for step in reversed(range(steps)):
                t = torch.full((graph_batch.batch_size,), step, device=self.device, dtype=torch.long)
                x = self.p_sample(x, t, guidance_grad=guidance_grad, graph_batch=graph_batch)
                if trajectories is not None:
                    for list_index, graph_index in enumerate(tracked_indices):
                        trajectories[list_index].append(x[graph_batch.graph_slice(graph_index)].detach().cpu().clone())

        if trajectories is None:
            return x, None
        return x, [torch.stack(trajectory, dim=0) for trajectory in trajectories]

    def p_sample_loop(
        self,
        shape: tuple[int, int],
        n_steps: Optional[int] = None,
        *,
        guidance_grad: GuidanceGradFn | None = None,
    ) -> torch.Tensor:
        """Generate samples starting from pure noise x_T ~ N(0,I)."""
        steps = self._resolve_n_steps(n_steps)
        x, _ = self._sample_loop(shape, steps=steps, guidance_grad=guidance_grad)
        return x                                            # final x is a generated sample (approx x_0)

    def p_sample_loop_graph(
        self,
        num_vertices: torch.Tensor | Sequence[int],
        *,
        n_steps: Optional[int] = None,
        guidance_grad: GuidanceGradFn | None = None,
    ) -> tuple[torch.Tensor, PolygonGraphBatch]:
        steps = self._resolve_n_steps(n_steps)
        graph_batch = build_polygon_graph_batch(num_vertices, device=self.device)
        x, _ = self._sample_graph_loop(graph_batch, steps=steps, guidance_grad=guidance_grad)
        return x, graph_batch

    def p_sample_loop_trajectory(
        self,
        shape: tuple[int, int],
        *,
        n_steps: Optional[int] = None,
        trajectory_index: int = 0,
        guidance_grad: GuidanceGradFn | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate samples and record one sample's denoising trajectory."""
        steps = self._resolve_n_steps(n_steps)
        x, trajectory = self._sample_loop(shape, steps=steps, trajectory_indices=[trajectory_index], guidance_grad=guidance_grad)
        if trajectory is None:
            raise RuntimeError("trajectory capture unexpectedly returned no trajectory")
        return x, trajectory[:, 0, :]

    def p_sample_loop_trajectories(
        self,
        shape: tuple[int, int],
        *,
        n_steps: Optional[int] = None,
        trajectory_indices: Sequence[int],
        guidance_grad: GuidanceGradFn | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate samples and record several samples' denoising trajectories."""
        steps = self._resolve_n_steps(n_steps)
        x, trajectories = self._sample_loop(
            shape,
            steps=steps,
            trajectory_indices=trajectory_indices,
            guidance_grad=guidance_grad,
        )
        if trajectories is None:
            raise RuntimeError("trajectory capture unexpectedly returned no trajectories")
        return x, trajectories

    def p_sample_loop_graph_trajectories(
        self,
        num_vertices: torch.Tensor | Sequence[int],
        *,
        n_steps: Optional[int] = None,
        trajectory_indices: Sequence[int],
        guidance_grad: GuidanceGradFn | None = None,
    ) -> tuple[torch.Tensor, PolygonGraphBatch, list[torch.Tensor]]:
        steps = self._resolve_n_steps(n_steps)
        graph_batch = build_polygon_graph_batch(num_vertices, device=self.device)
        x, trajectories = self._sample_graph_loop(
            graph_batch,
            steps=steps,
            trajectory_indices=trajectory_indices,
            guidance_grad=guidance_grad,
        )
        if trajectories is None:
            raise RuntimeError("trajectory capture unexpectedly returned no trajectories")
        return x, graph_batch, trajectories

    def loss(
        self,
        x0: torch.Tensor,
        *,
        graph_batch: PolygonGraphBatch | None = None,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Training loss: predict the noise used to create x_t from x_0."""
        b = graph_batch.batch_size if graph_batch is not None else x0.shape[0]     # batch size B
        
        # sample t ~ Uniform{0,...,T-1} independently for each batch element
        t = torch.randint(0, self.config.n_steps, (b,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x0)        # sample ε ~ N(0,I) same shape as x0
        x_t = self.q_sample(x0, t, noise, graph_batch=graph_batch)   # create noisy inputs x_t from x0 and ε
        eps_pred = self.predict_eps(x_t, t, graph_batch=graph_batch) # predict ε_hat(x_t, t)
        
        # minimize mean squared error: ||ε_hat - ε||^2
        loss = nn.functional.mse_loss(eps_pred, noise)
        if not return_stats:
            return loss

        if graph_batch is None:
            sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(1)
            sqrt_omab = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        else:
            node_t = t.index_select(0, graph_batch.graph_index)
            sqrt_ab = self.sqrt_alphas_cumprod[node_t].unsqueeze(1)
            sqrt_omab = self.sqrt_one_minus_alphas_cumprod[node_t].unsqueeze(1)
        x0_pred = (x_t - sqrt_omab * eps_pred) / sqrt_ab.clamp_min(1e-8)

        stats = {
            "loss": float(loss.detach().item()),
            "t_mean": float(t.float().mean().item()),
            "t_std": float(t.float().std(unbiased=False).item()),
            "x_t_std": float(x_t.detach().std(unbiased=False).item()),
            "noise_std": float(noise.detach().std(unbiased=False).item()),
            "eps_pred_std": float(eps_pred.detach().std(unbiased=False).item()),
            "x0_pred_mse": float(nn.functional.mse_loss(x0_pred, x0).detach().item()),
        }
        return loss, stats
