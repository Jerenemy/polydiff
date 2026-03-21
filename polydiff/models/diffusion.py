"""Minimal DDPM components for polygon diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

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

class DenoiseGAT(nn.Module):
    def __init__(self, data_dim: int, hidden_dim: int, time_emb_dim: int, num_layers: int) -> None:
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

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        self.register_buffer(
            "vertex_positional_features",
            self._build_cycle_positional_features(self.num_vertices),
            persistent=False,
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
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.global_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.node_head = nn.Sequential(
            # nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.coord_dim),
        )
        self.register_buffer("base_edge_index", self._build_cycle_edge_index(self.num_vertices), persistent=False)

    @staticmethod
    def _build_cycle_edge_index(num_vertices: int) -> torch.Tensor:
        nodes = torch.arange(num_vertices, dtype=torch.long)
        next_nodes = torch.roll(nodes, shifts=-1)
        prev_nodes = torch.roll(nodes, shifts=1)
        return torch.stack(
            [
                torch.cat([nodes, nodes, nodes], dim=0),
                torch.cat([next_nodes, prev_nodes, nodes], dim=0),
            ],
            dim=0,
        )

    @staticmethod
    def _build_cycle_positional_features(num_vertices: int) -> torch.Tensor:
        phase = torch.arange(num_vertices, dtype=torch.float32) * (2.0 * torch.pi / num_vertices)
        return torch.stack(
            [
                torch.sin(phase),
                torch.cos(phase),
                torch.sin(2.0 * phase),
                torch.cos(2.0 * phase),
            ],
            dim=1,
        )

    def _batched_edge_index(self, batch_size: int, device: torch.device) -> torch.Tensor:
        edge_index = self.base_edge_index.to(device)
        num_edges = edge_index.shape[1]
        offsets = (
            torch.arange(batch_size, device=device, dtype=edge_index.dtype)
            .repeat_interleave(num_edges)
            .mul(self.num_vertices)
        )
        return edge_index.repeat(1, batch_size) + offsets.unsqueeze(0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.data_dim:
            raise ValueError(f"x must have shape (batch, {self.data_dim}), got {tuple(x.shape)}")
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(f"t must have shape ({x.shape[0]},), got {tuple(t.shape)}")


        batch_size = x.shape[0]
        # shape: scalar int
        # meaning: number of polygons in the batch
        # call it B

        coords = x.reshape(batch_size, self.num_vertices, self.coord_dim)
        # shape: (B, V, 2)
        # meaning: turn each flattened polygon back into V vertices with 2 coordinates each
        # so coords[b, i] is the (x, y) pair for vertex i of polygon b

        time_emb = self.time_mlp(t).unsqueeze(1).expand(-1, self.num_vertices, -1)
        # self.time_mlp(t): (B, T)
        # unsqueeze(1):     (B, 1, T)
        # expand(...):      (B, V, T)
        # meaning: compute one timestep embedding per polygon, then copy it to every vertex
        # so all vertices in the same polygon get the same diffusion-time context

        pos_enc = self.vertex_positional_features.to(x.device).unsqueeze(0).expand(batch_size, -1, -1)
        # self.vertex_positional_features: (V, P) = (V, 4)
        # unsqueeze(0):                    (1, V, P)
        # expand(...):                     (B, V, P)
        # meaning: give each vertex a fixed "where am I on the ring?" encoding
        # this is the sin/cos cycle-position info, copied across the batch

        node_features = torch.cat([coords, pos_enc, time_emb], dim=-1).reshape(batch_size * self.num_vertices, -1)
        # before reshape:
        #   coords:      (B, V, 2)
        #   pos_enc:     (B, V, P)
        #   time_emb:    (B, V, T)
        # cat dim=-1 ->  (B, V, 2 + P + T)
        #
        # after reshape:
        #   (B * V, 2 + P + T)
        #
        # meaning: each row is now one vertex from one polygon
        # all batch polygons are flattened into one big list of nodes

        edge_index = self._batched_edge_index(batch_size, x.device)
        # shape: (2, E_total)
        # meaning: graph connectivity for the whole batch
        # this is not one connected giant graph in the semantic sense
        # it is a disjoint union of B separate cycle graphs, one per polygon

        node_hidden, _ = self.gat((node_features, edge_index))
        # input node_features: (B * V, 2 + P + T)
        # output node_hidden:  (B * V, H)
        # meaning: GAT computes one hidden embedding per vertex
        # each row now represents one vertex after message passing with neighbors

        node_hidden = node_hidden.reshape(batch_size, self.num_vertices, -1)
        # shape: (B, V, H)
        # meaning: restore the batch structure
        # node_hidden[b, i] is the hidden vector for vertex i in polygon b

        node_noise = self.node_head(node_hidden.reshape(batch_size * self.num_vertices, -1))
        # node_hidden.reshape(...): (B * V, H)
        # self.node_head output:    (B * V, 2)
        # meaning: predict a 2D noise vector for each vertex independently from its hidden embedding
        # so each row is the predicted (delta_x_noise, delta_y_noise) for one vertex

        return node_noise.reshape(batch_size, self.data_dim)
        # node_noise before reshape: (B * V, 2)
        # after reshape:             (B, 2 * V) = (B, data_dim)
        # meaning: flatten per-vertex predictions back into the same format as the input/output diffusion tensors

        ### global features: currently removed
        # # global_features = self.global_head(node_hidden.mean(dim=1))
        # # global_features = global_features.unsqueeze(1).expand(-1, self.num_vertices, -1)
        # # global_gate = self.global_gate(torch.cat([node_hidden, global_features], dim=-1))
        # # gated_global = global_gate * global_features
        # # node_noise = self.node_head(torch.cat([node_hidden, gated_global], dim=-1).reshape(batch_size * self.num_vertices, -1))



class DenoiseGCN(nn.Module):
    def __init__(self, data_dim: int, hidden_dim: int, time_emb_dim: int, num_layers: int) -> None:
        super().__init__()
        if data_dim % 2 != 0:
            raise ValueError(f"data_dim must be even for flattened 2D polygon coordinates, got {data_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.data_dim = data_dim
        self.coord_dim = 2
        self.num_vertices = data_dim // self.coord_dim

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
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.coord_dim),
        )
        self.register_buffer("base_adj_indices", self._build_cycle_adj_indices(self.num_vertices), persistent=False)
        self.register_buffer(
            "base_adj_values",
            torch.full((self.base_adj_indices.shape[1],), 1.0 / 3.0, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _build_cycle_adj_indices(num_vertices: int) -> torch.Tensor:
        nodes = torch.arange(num_vertices, dtype=torch.long)
        next_nodes = torch.roll(nodes, shifts=-1)
        prev_nodes = torch.roll(nodes, shifts=1)
        return torch.stack(
            [
                torch.cat([nodes, nodes, nodes], dim=0),
                torch.cat([nodes, next_nodes, prev_nodes], dim=0),
            ],
            dim=0,
        )

    def _batched_adj(self, batch_size: int, device: torch.device) -> torch.Tensor:
        base_indices = self.base_adj_indices.to(device)
        num_edges = base_indices.shape[1]
        offsets = (
            torch.arange(batch_size, device=device, dtype=base_indices.dtype)
            .repeat_interleave(num_edges)
            .mul(self.num_vertices)
        )
        indices = base_indices.repeat(1, batch_size) + offsets.unsqueeze(0)
        values = self.base_adj_values.to(device).repeat(batch_size)
        size = (batch_size * self.num_vertices, batch_size * self.num_vertices)
        return torch.sparse_coo_tensor(indices, values, size=size, device=device).coalesce()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.data_dim:
            raise ValueError(f"x must have shape (batch, {self.data_dim}), got {tuple(x.shape)}")
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(f"t must have shape ({x.shape[0]},), got {tuple(t.shape)}")

        batch_size = x.shape[0]
        coords = x.reshape(batch_size, self.num_vertices, self.coord_dim)
        time_emb = self.time_mlp(t).unsqueeze(1).expand(-1, self.num_vertices, -1)
        h = torch.cat([coords, time_emb], dim=-1).reshape(batch_size * self.num_vertices, -1)
        adj = self._batched_adj(batch_size, x.device)

        for i, layer in enumerate(self.layers):
            residual = self.residual_projs[i](h)
            h = layer(h, adj) + residual
            h = self.activation(h)

        h = self.node_head(h)
        return h.reshape(batch_size, self.data_dim)


def build_denoiser(*, data_dim: int, model_cfg: Mapping[str, Any] | None = None) -> nn.Module:
    cfg = {} if model_cfg is None else dict(model_cfg)
    model_type = str(cfg.get("type", "gat")).lower()
    hidden_dim = int(cfg.get("hidden_dim", 256))
    time_emb_dim = int(cfg.get("time_emb_dim", 64))
    num_layers = int(cfg.get("num_layers", 3))

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
        )
    if model_type == "gcn":
        return DenoiseGCN(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=time_emb_dim,
            num_layers=num_layers,
        )
    raise ValueError(f"Unsupported model.type {model_type!r}; expected one of: mlp, gat, gcn")



@dataclass
class DiffusionConfig:
    n_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


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

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion (adding noise). Create x_t in in one shot"""
        sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(1)              # gather sqrt(ᾱ_t) per sample, shape (B,1)
        sqrt_omab = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)  # gather sqrt(1-ᾱ_t) per sample, shape (B,1)
        
        # x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
        return sqrt_ab * x0 + sqrt_omab * noise

    def predict_eps(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x_t, t) # eps_hat = model(x_t, t)

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion step: sample x_{t-1} from learned p(x_{t-1} | x_t)."""
        betas_t = self.betas[t].unsqueeze(1)                        # gather β_t per sample, shape (B,1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].unsqueeze(1) # gather sqrt(1/α_t) per sample, shape (B,1)
        sqrt_omab_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1) # gather sqrt(1-ᾱ_t), shape (B,1)

        eps_pred = self.predict_eps(x_t, t)                         # model predicts ε_hat(x_t, t)
        
        # DDPM mean:
        # μθ(x_t,t) = (1/sqrt(α_t)) * (x_t - (β_t / sqrt(1-ᾱ_t)) * ε_hat)
        model_mean = sqrt_recip_alpha_t * (x_t - betas_t * eps_pred / sqrt_omab_t)

        # At t=0, return the mean (no noise added in the final step)
        if t.min().item() == 0:
            return model_mean

        posterior_var_t = self.posterior_variance[t].unsqueeze(1)   # gather β̃_t per sample, shape (B,1)
        noise = torch.randn_like(x_t)        
        
        # Sample:
        # x_{t-1} = μθ + sqrt(β̃_t) * z# fresh Gaussian noise z ~ N(0,I)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    def _sample_loop(
        self,
        shape: tuple[int, int],
        *,
        steps: int,
        trajectory_index: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if trajectory_index is not None and not 0 <= trajectory_index < shape[0]:
            raise ValueError(f"trajectory_index must be in [0, {shape[0] - 1}], got {trajectory_index}")

        self.model.eval()
        with torch.no_grad():
            x = torch.randn(shape, device=self.device)      # initialize x_T as standard Gaussian noise
            trajectory = None
            if trajectory_index is not None:
                trajectory = [x[trajectory_index].detach().cpu().clone()]
            for step in reversed(range(steps)):             # iterate t = steps-1, ..., 0
                t = torch.full((shape[0],), step, device=self.device, dtype=torch.long) # batch of identical t
                x = self.p_sample(x, t)                     # sample x_{t-1} from x_t
                if trajectory is not None:
                    trajectory.append(x[trajectory_index].detach().cpu().clone())

        if trajectory is None:
            return x, None
        return x, torch.stack(trajectory, dim=0)

    def p_sample_loop(self, shape: tuple[int, int], n_steps: Optional[int] = None) -> torch.Tensor:
        """Generate samples starting from pure noise x_T ~ N(0,I)."""
        steps = self._resolve_n_steps(n_steps)
        x, _ = self._sample_loop(shape, steps=steps)
        return x                                            # final x is a generated sample (approx x_0)

    def p_sample_loop_trajectory(
        self,
        shape: tuple[int, int],
        *,
        n_steps: Optional[int] = None,
        trajectory_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate samples and record one sample's denoising trajectory."""
        steps = self._resolve_n_steps(n_steps)
        x, trajectory = self._sample_loop(shape, steps=steps, trajectory_index=trajectory_index)
        if trajectory is None:
            raise RuntimeError("trajectory capture unexpectedly returned no trajectory")
        return x, trajectory

    def loss(self, x0: torch.Tensor, *, return_stats: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Training loss: predict the noise used to create x_t from x_0."""
        b = x0.shape[0]     # batch size B
        
        # sample t ~ Uniform{0,...,T-1} independently for each batch element
        t = torch.randint(0, self.config.n_steps, (b,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x0)        # sample ε ~ N(0,I) same shape as x0
        x_t = self.q_sample(x0, t, noise)   # create noisy inputs x_t from x0 and ε
        eps_pred = self.predict_eps(x_t, t) # predict ε_hat(x_t, t)
        
        # minimize mean squared error: ||ε_hat - ε||^2
        loss = nn.functional.mse_loss(eps_pred, noise)
        if not return_stats:
            return loss

        sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_omab = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
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
