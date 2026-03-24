import numpy as np
import pytest
import torch

from polydiff.data.polygon_dataset import build_polygon_graph_batch
from polydiff.restoration import (
    RestorationProxyConfig,
    RestorationTrajectoryRecorder,
    build_restoration_animation_overlay,
    restoration_scene_coords_numpy,
    restoration_state_numpy,
    restoration_state_torch_dense,
    restoration_state_torch_graph,
)
from polydiff.sampling.guidance import RestorationGuidance
from polydiff.sampling.runtime import resolve_sampling_request


def _restoration_config() -> RestorationProxyConfig:
    return RestorationProxyConfig(
        mutant_target_points=((-1.0, -0.8), (0.8, -0.7), (1.0, 0.2), (0.4, 0.9), (-0.8, 0.7)),
        wild_type_target_points=((-0.9, -0.72), (0.95, -0.62), (1.18, 0.28), (0.55, 1.02), (-0.72, 0.78)),
        ligand_binding_site=(1.0, 0.0),
        dna_unbound_position=(2.0, 0.0),
        dna_bound_position=(0.0, 0.0),
        activation_sigma=0.30,
        contact_beta=18.0,
        dna_binding_threshold=0.65,
        dna_binding_steepness=14.0,
        success_distance=0.25,
    )


def test_restoration_state_prefers_polygon_contact_near_binding_site():
    config = _restoration_config()
    near = np.asarray([[-1.0, -0.2], [-0.3, -0.8], [1.05, 0.0], [-0.2, 0.8]], dtype=np.float32)
    far = np.asarray([[-1.1, -0.2], [-0.5, -0.9], [-0.2, 0.0], [-0.5, 0.9]], dtype=np.float32)

    near_state = restoration_state_numpy(near, config)
    far_state = restoration_state_numpy(far, config)

    assert float(near_state.protein_restoration) > float(far_state.protein_restoration)
    assert float(near_state.dna_binding_activation) > float(far_state.dna_binding_activation)
    assert float(near_state.dna_distance) < float(far_state.dna_distance)


def test_restoration_graph_and_dense_paths_match_for_uniform_batches():
    config = _restoration_config()
    coords = torch.tensor(
        [
            [[-1.0, -0.2], [-0.3, -0.8], [1.05, 0.0], [-0.2, 0.8]],
            [[-1.1, -0.2], [-0.5, -0.9], [-0.2, 0.0], [-0.5, 0.9]],
        ],
        dtype=torch.float32,
    )

    dense_state = restoration_state_torch_dense(coords, config)
    graph_batch = build_polygon_graph_batch([4, 4], coords=coords.reshape(-1, 2))
    graph_state = restoration_state_torch_graph(coords.reshape(-1, 2), graph_batch, config)

    assert torch.allclose(dense_state.contact_point, graph_state.contact_point)
    assert torch.allclose(dense_state.protein_restoration, graph_state.protein_restoration)
    assert torch.allclose(dense_state.protein_points, graph_state.protein_points)
    assert torch.allclose(dense_state.dna_binding_activation, graph_state.dna_binding_activation)
    assert torch.allclose(dense_state.dna_distance, graph_state.dna_distance)


def test_restoration_animation_overlay_and_recorder_track_trajectory():
    config = _restoration_config()
    trajectory = np.asarray(
        [
            [[-1.1, -0.2], [-0.5, -0.9], [-0.2, 0.0], [-0.5, 0.9]],
            [[-1.0, -0.2], [-0.4, -0.8], [0.5, 0.0], [-0.4, 0.8]],
            [[-1.0, -0.2], [-0.3, -0.8], [1.05, 0.0], [-0.2, 0.8]],
        ],
        dtype=np.float32,
    )

    overlay = build_restoration_animation_overlay(trajectory, config)
    assert overlay.dna_positions.shape == (3, 2)
    assert overlay.protein_points.shape == (3, 5, 2)
    assert overlay.dna_distance[0] > overlay.dna_distance[-1]
    assert overlay.select_frames(np.asarray([0, 2], dtype=np.int32)).protein_restoration.shape == (2,)

    recorder = RestorationTrajectoryRecorder(config)
    recorder.observe(torch.tensor(trajectory[0:1].reshape(1, -1)), step=2)
    recorder.observe(torch.tensor(trajectory[-1:].reshape(1, -1)), step=0)
    payload = recorder.to_dict()
    assert payload["diffusion_step"] == [2, 0]
    assert payload["protein_restoration_mean"][0] < payload["protein_restoration_mean"][-1]
    assert payload["dna_binding_activation_mean"][0] < payload["dna_binding_activation_mean"][-1]
    assert payload["dna_distance_mean"][0] > payload["dna_distance_mean"][-1]


def test_restoration_state_is_translation_invariant_after_scene_anchoring():
    config = _restoration_config()
    polygon = np.asarray([[-1.0, -0.2], [-0.3, -0.8], [1.05, 0.0], [-0.2, 0.8]], dtype=np.float32)
    shifted = polygon + np.asarray([25.0, -17.5], dtype=np.float32)

    state_a = restoration_state_numpy(polygon, config)
    state_b = restoration_state_numpy(shifted, config)
    scene_a = restoration_scene_coords_numpy(polygon, config)
    scene_b = restoration_scene_coords_numpy(shifted, config)

    assert np.allclose(scene_a, scene_b, atol=1e-5)
    assert np.allclose(state_a.contact_point, state_b.contact_point, atol=1e-5)
    assert float(state_a.protein_restoration) == pytest.approx(float(state_b.protein_restoration), abs=1e-5)
    assert float(state_a.dna_distance) == pytest.approx(float(state_b.dna_distance), abs=5e-5)


def test_restoration_guidance_uses_timestep_schedule():
    config = _restoration_config()
    coords = torch.tensor(
        [[-1.1, -0.2, -0.5, -0.9, -0.2, 0.0, -0.5, 0.9]],
        dtype=torch.float32,
    )
    guidance = RestorationGuidance(
        n_vertices=4,
        scale=1.0,
        restoration=config,
        num_steps=10,
        min_timestep_weight=0.05,
        timestep_power=2.0,
    )

    grad_early = guidance(coords, torch.tensor([9], dtype=torch.long))
    grad_late = guidance(coords, torch.tensor([0], dtype=torch.long))

    assert torch.linalg.norm(grad_late) > torch.linalg.norm(grad_early)


def test_restoration_toggle_is_explicit_in_sampling_request(tmp_path):
    regularity_request = resolve_sampling_request(
        {
            "num_samples": 4,
            "guidance": {"enabled": True, "kind": "regularity", "scale": 2.0},
        },
        checkpoint_n_steps=8,
        default_out_path=tmp_path / "samples.npz",
        default_animation_out_path=tmp_path / "animations",
        enable_animation=False,
        animation_out_path=None,
        animation_count=None,
        animation_sample_index=None,
        animation_max_frames=None,
        animation_fps=None,
    )
    assert regularity_request.restoration is None
    assert regularity_request.guidance.enabled
    assert regularity_request.guidance.components[0].kind == "regularity"

    disabled_request = resolve_sampling_request(
        {
            "num_samples": 4,
            "restoration": {"enabled": False},
            "guidance": {"enabled": False},
        },
        checkpoint_n_steps=8,
        default_out_path=tmp_path / "samples_disabled.npz",
        default_animation_out_path=tmp_path / "animations_disabled",
        enable_animation=False,
        animation_out_path=None,
        animation_count=None,
        animation_sample_index=None,
        animation_max_frames=None,
        animation_fps=None,
    )
    assert disabled_request.restoration is None

    multi_request = resolve_sampling_request(
        {
            "num_samples": 4,
            "restoration": {
                "enabled": True,
                "mutant_target_points": [[-1.0, -0.8], [0.8, -0.7], [1.0, 0.2]],
                "wild_type_target_points": [[-0.9, -0.75], [0.9, -0.65], [1.2, 0.3]],
                "ligand_binding_site": [1.0, 0.0],
                "dna_unbound_position": [2.0, 0.0],
                "dna_bound_position": [0.0, 0.0],
                "dna_binding_threshold": 0.7,
                "dna_binding_steepness": 12.0,
            },
            "guidance": {
                "enabled": True,
                "components": [
                    {"kind": "regularity", "scale": 2.0},
                    {
                        "kind": "restoration",
                        "scale": 5.0,
                        "min_timestep_weight": 0.2,
                        "timestep_power": 1.5,
                    },
                ],
            },
        },
        checkpoint_n_steps=8,
        default_out_path=tmp_path / "samples_multi.npz",
        default_animation_out_path=tmp_path / "animations_multi",
        enable_animation=False,
        animation_out_path=None,
        animation_count=None,
        animation_sample_index=None,
        animation_max_frames=None,
        animation_fps=None,
    )
    assert multi_request.restoration is not None
    assert [component.kind for component in multi_request.guidance.components] == ["regularity", "restoration"]
    assert multi_request.restoration.ligand_binding_site == pytest.approx((1.0, 0.0))
    assert multi_request.restoration.dna_binding_threshold == pytest.approx(0.7)
    assert multi_request.guidance.components[1].min_timestep_weight == pytest.approx(0.2)
    assert multi_request.guidance.components[1].timestep_power == pytest.approx(1.5)

    legacy_alias_request = resolve_sampling_request(
        {
            "num_samples": 2,
            "restoration": {
                "enabled": True,
                "target_points": [[-1.0, -0.8], [0.8, -0.7], [1.0, 0.2]],
                "binding_site": [1.0, 0.0],
                "mutant_position": [2.0, 0.0],
                "wild_type_position": [0.0, 0.0],
            },
            "guidance": {"enabled": False},
        },
        checkpoint_n_steps=8,
        default_out_path=tmp_path / "samples_legacy.npz",
        default_animation_out_path=tmp_path / "animations_legacy",
        enable_animation=False,
        animation_out_path=None,
        animation_count=None,
        animation_sample_index=None,
        animation_max_frames=None,
        animation_fps=None,
    )
    assert legacy_alias_request.restoration is not None
    assert legacy_alias_request.restoration.ligand_binding_site == pytest.approx((1.0, 0.0))
    assert legacy_alias_request.restoration.dna_unbound_position == pytest.approx((2.0, 0.0))

    with pytest.raises(ValueError, match="restoration guidance requires sampling.restoration.enabled"):
        resolve_sampling_request(
            {
                "num_samples": 4,
                "guidance": {"enabled": True, "kind": "restoration", "scale": 3.0},
            },
            checkpoint_n_steps=8,
            default_out_path=tmp_path / "samples_error.npz",
            default_animation_out_path=tmp_path / "animations_error",
            enable_animation=False,
            animation_out_path=None,
            animation_count=None,
            animation_sample_index=None,
            animation_max_frames=None,
            animation_fps=None,
        )
