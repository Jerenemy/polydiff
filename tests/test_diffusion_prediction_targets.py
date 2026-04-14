from pathlib import Path

import torch
import torch.nn as nn

from polydiff.models.diffusion import Diffusion, DiffusionConfig, build_denoiser
from polydiff.sampling.runtime import load_diffusion_from_checkpoint


class ConstantModel(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("output", output.clone())

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del x, t
        return self.output.clone()


def test_diffusion_config_defaults_to_x0_and_normalizes_aliases():
    assert DiffusionConfig().prediction_target == "x0"
    assert DiffusionConfig(prediction_target="mean").prediction_target == "x0"
    assert DiffusionConfig(prediction_target="epsilon").prediction_target == "epsilon"


def test_predict_x0_and_eps_convert_between_targets():
    x0 = torch.tensor([[0.25, -0.50, 0.75, 1.25]], dtype=torch.float32)
    noise = torch.tensor([[0.10, -0.20, 0.30, -0.40]], dtype=torch.float32)
    t = torch.tensor([2], dtype=torch.long)

    epsilon_diffusion = Diffusion(
        model=ConstantModel(noise),
        config=DiffusionConfig(n_steps=5, prediction_target="epsilon"),
    )
    x_t = epsilon_diffusion.q_sample(x0, t, noise)
    x0_pred = epsilon_diffusion.predict_x0(x_t, t)
    assert torch.allclose(x0_pred, x0, atol=1e-6)

    x0_diffusion = Diffusion(
        model=ConstantModel(x0),
        config=DiffusionConfig(n_steps=5, prediction_target="x0"),
    )
    eps_pred = x0_diffusion.predict_eps(x_t, t)
    assert torch.allclose(eps_pred, noise, atol=1e-6)


def test_sampling_matches_between_epsilon_and_x0_parameterizations():
    x0 = torch.tensor([[0.40, -0.30, 1.10, 0.20]], dtype=torch.float32)
    noise = torch.tensor([[0.05, -0.15, 0.25, -0.35]], dtype=torch.float32)
    t = torch.tensor([3], dtype=torch.long)

    epsilon_diffusion = Diffusion(
        model=ConstantModel(noise),
        config=DiffusionConfig(n_steps=6, prediction_target="epsilon"),
    )
    x_t = epsilon_diffusion.q_sample(x0, t, noise)

    x0_diffusion = Diffusion(
        model=ConstantModel(x0),
        config=DiffusionConfig(
            n_steps=6,
            beta_start=epsilon_diffusion.config.beta_start,
            beta_end=epsilon_diffusion.config.beta_end,
            prediction_target="x0",
        ),
    )

    torch.manual_seed(11)
    sample_eps = epsilon_diffusion.p_sample(x_t, t)
    torch.manual_seed(11)
    sample_x0 = x0_diffusion.p_sample(x_t, t)

    assert torch.allclose(sample_eps, sample_x0, atol=1e-6)


def test_checkpoint_loading_preserves_x0_and_legacy_epsilon_targets(tmp_path: Path):
    model_cfg = {"type": "mlp", "hidden_dim": 8, "time_emb_dim": 4, "num_layers": 1}
    model = build_denoiser(data_dim=6, model_cfg=model_cfg)

    legacy_checkpoint_path = tmp_path / "legacy_diffusion.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "diffusion": {"n_steps": 4, "beta_start": 1e-4, "beta_end": 2e-2},
            "model_cfg": model_cfg,
            "n_vertices": 3,
            "max_vertices": 3,
        },
        legacy_checkpoint_path,
    )

    _, legacy_diffusion, legacy_config, legacy_max_vertices = load_diffusion_from_checkpoint(
        legacy_checkpoint_path,
        device=torch.device("cpu"),
    )

    assert legacy_diffusion.config.prediction_target == "epsilon"
    assert legacy_config.prediction_target == "epsilon"
    assert legacy_max_vertices == 3

    x0_checkpoint_path = tmp_path / "x0_diffusion.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "diffusion": {
                "n_steps": 4,
                "beta_start": 1e-4,
                "beta_end": 2e-2,
                "prediction_target": "x0",
            },
            "model_cfg": model_cfg,
            "n_vertices": 3,
            "max_vertices": 3,
        },
        x0_checkpoint_path,
    )

    _, x0_diffusion, x0_config, x0_max_vertices = load_diffusion_from_checkpoint(
        x0_checkpoint_path,
        device=torch.device("cpu"),
    )

    assert x0_diffusion.config.prediction_target == "x0"
    assert x0_config.prediction_target == "x0"
    assert x0_max_vertices == 3
