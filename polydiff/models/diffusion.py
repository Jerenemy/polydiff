"""Minimal DDPM components for polygon diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) int or float tensor -> (B, dim) embedding."""
        half = self.dim // 2
        device = t.device
        emb = torch.arange(half, device=device, dtype=torch.float32)
        emb = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * emb / (half - 1))
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class DenoiseMLP(nn.Module):
    def __init__(self, data_dim: int, hidden_dim: int, time_emb_dim: int, num_layers: int) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        layers = []
        in_dim = data_dim + time_emb_dim
        for i in range(max(1, num_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(t)
        h = torch.cat([x, emb], dim=1)
        return self.net(h)


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
        self.model.to(self.device)

        betas = torch.linspace(config.beta_start, config.beta_end, config.n_steps, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion (adding noise)."""
        sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_omab = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        return sqrt_ab * x0 + sqrt_omab * noise

    def predict_eps(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x_t, t)

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion step."""
        betas_t = self.betas[t].unsqueeze(1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].unsqueeze(1)
        sqrt_omab_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)

        eps_pred = self.predict_eps(x_t, t)
        model_mean = sqrt_recip_alpha_t * (x_t - betas_t * eps_pred / sqrt_omab_t)

        if t.min().item() == 0:
            return model_mean

        posterior_var_t = self.posterior_variance[t].unsqueeze(1)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    def p_sample_loop(self, shape: tuple[int, int], n_steps: Optional[int] = None) -> torch.Tensor:
        """Generate samples starting from noise."""
        steps = self.config.n_steps if n_steps is None else int(n_steps)
        if steps < 1 or steps > self.config.n_steps:
            raise ValueError(f"n_steps must be in [1, {self.config.n_steps}], got {steps}")

        self.model.eval()
        with torch.no_grad():
            x = torch.randn(shape, device=self.device)
            for step in reversed(range(steps)):
                t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
                x = self.p_sample(x, t)
        return x

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Simple MSE loss between predicted and true noise."""
        b = x0.shape[0]
        t = torch.randint(0, self.config.n_steps, (b,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.predict_eps(x_t, t)
        return nn.functional.mse_loss(eps_pred, noise)
