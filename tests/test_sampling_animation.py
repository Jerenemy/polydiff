import numpy as np
import pytest
import torch
import torch.nn as nn

from polydiff.data.plot_polygons import select_animation_frames
from polydiff.models.diffusion import Diffusion, DiffusionConfig


class ZeroModel(nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


class ZeroGraphModel(nn.Module):
    def forward(self, x, t, *, batch=None):
        del t, batch
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


def test_p_sample_loop_graph_trajectories_track_variable_sizes():
    diffusion = Diffusion(model=ZeroGraphModel(), config=DiffusionConfig(n_steps=4))
    num_vertices = [5, 7, 6]

    torch.manual_seed(7)
    expected, expected_batch = diffusion.p_sample_loop_graph(num_vertices, n_steps=4)

    torch.manual_seed(7)
    actual, actual_batch, trajectories = diffusion.p_sample_loop_graph_trajectories(
        num_vertices,
        n_steps=4,
        trajectory_indices=[1, 2],
    )

    assert torch.allclose(actual, expected)
    assert actual_batch.num_vertices.tolist() == expected_batch.num_vertices.tolist() == num_vertices
    assert len(trajectories) == 2
    assert trajectories[0].shape == (5, 7, 2)
    assert trajectories[1].shape == (5, 6, 2)
    assert torch.allclose(trajectories[0][-1], actual[actual_batch.graph_slice(1)].cpu())
    assert torch.allclose(trajectories[1][-1], actual[actual_batch.graph_slice(2)].cpu())
