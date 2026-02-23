"""Minimal DDPM components for polygon diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


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

    def p_sample_loop(self, shape: tuple[int, int], n_steps: Optional[int] = None) -> torch.Tensor:
        """Generate samples starting from pure noise x_T ~ N(0,I)."""
        steps = self.config.n_steps if n_steps is None else int(n_steps) # choose how many reverse steps to run
        if steps < 1 or steps > self.config.n_steps:
            raise ValueError(f"n_steps must be in [1, {self.config.n_steps}], got {steps}")

        self.model.eval()
        with torch.no_grad():
            x = torch.randn(shape, device=self.device)      # initialize x_T as standard Gaussian noise
            for step in reversed(range(steps)):             # iterate t = steps-1, ..., 0
                t = torch.full((shape[0],), step, device=self.device, dtype=torch.long) # batch of identical t
                x = self.p_sample(x, t)                     # sample x_{t-1} from x_t
        return x                                            # final x is a generated sample (approx x_0)

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Training loss: predict the noise used to create x_t from x_0."""
        b = x0.shape[0]     # batch size B
        
        # sample t ~ Uniform{0,...,T-1} independently for each batch element
        t = torch.randint(0, self.config.n_steps, (b,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x0)        # sample ε ~ N(0,I) same shape as x0
        x_t = self.q_sample(x0, t, noise)   # create noisy inputs x_t from x0 and ε
        eps_pred = self.predict_eps(x_t, t) # predict ε_hat(x_t, t)
        
        # minimize mean squared error: ||ε_hat - ε||^2
        return nn.functional.mse_loss(eps_pred, noise)
    