import numpy as np
import pytest
import torch
import torch.nn as nn

from polydiff.data.plot_polygons import select_animation_frames
from polydiff.models.diffusion import Diffusion, DiffusionConfig


class ZeroModel(nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_p_sample_loop_trajectory_matches_standard_sampling():
    diffusion = Diffusion(model=ZeroModel(), config=DiffusionConfig(n_steps=4))

    torch.manual_seed(7)
    expected = diffusion.p_sample_loop((3, 6), n_steps=4)

    torch.manual_seed(7)
    actual, trajectory = diffusion.p_sample_loop_trajectory((3, 6), n_steps=4, trajectory_index=1)

    assert torch.allclose(actual, expected)
    assert trajectory.shape == (5, 6)
    assert torch.allclose(trajectory[-1], actual[1].cpu())


def test_p_sample_loop_trajectory_rejects_out_of_range_index():
    diffusion = Diffusion(model=ZeroModel(), config=DiffusionConfig(n_steps=4))

    with pytest.raises(ValueError, match="trajectory_index"):
        diffusion.p_sample_loop_trajectory((2, 6), n_steps=4, trajectory_index=2)


def test_select_animation_frames_keeps_endpoints_when_downsampling():
    coords = np.arange(20 * 3 * 2, dtype=np.float32).reshape(20, 3, 2)

    frames, indices = select_animation_frames(coords, max_frames=6)

    assert len(indices) <= 6
    assert indices[0] == 0
    assert indices[-1] == coords.shape[0] - 1
    assert np.all(np.diff(indices) > 0)
    assert np.array_equal(frames[0], coords[0])
    assert np.array_equal(frames[-1], coords[-1])
