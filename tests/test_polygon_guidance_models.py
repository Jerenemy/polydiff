from pathlib import Path

import numpy as np
import torch
import yaml

from polydiff.data.gen_polygons import batch_variable_sizes
from polydiff.data.polygon_dataset import build_polygon_graph_batch
from polydiff.models.guidance_models import build_guidance_model
from polydiff.sampling.guidance import load_sampling_guidance
from polydiff.training.train_guidance_model import train_guidance_model_from_config


def test_graph_guidance_model_supports_variable_size_batches():
    model = build_guidance_model(
        task="regressor",
        data_dim=14,
        model_cfg={
            "type": "gat",
            "hidden_dim": 16,
            "time_emb_dim": 8,
            "num_layers": 2,
            "pooling": "sum",
            "timestep_conditioning": True,
        },
    )
    graph_batch = build_polygon_graph_batch(torch.tensor([5, 7], dtype=torch.long))
    x_t = torch.randn((graph_batch.total_vertices, 2), dtype=torch.float32)
    t = torch.tensor([3, 9], dtype=torch.long)

    pred = model(x_t, t, batch=graph_batch)

    assert pred.shape == (2,)


def test_train_guidance_model_supports_variable_size_graph_checkpoint(tmp_path):
    data_path = tmp_path / "mixed_polygons.npz"
    coords, num_vertices, score, _ = batch_variable_sizes(
        size_values=[5, 6, 7],
        size_probabilities=[0.3, 0.4, 0.3],
        num=24,
        seed=0,
        radial_sigma=0.18,
        angle_sigma=0.12,
        smooth_passes=3,
    )
    np.savez_compressed(
        data_path,
        coords=coords.astype(np.float32),
        num_vertices=num_vertices.astype(np.int32),
        score=score.astype(np.float32),
    )

    config_path = tmp_path / "train_graph_guidance.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "seed": 0,
                "device": "cpu",
                "experiment_name": "graph-guidance-test",
                "data": {
                    "path": str(data_path),
                    "batch_size": 8,
                    "shuffle": True,
                    "num_workers": 0,
                },
                "classifier": {
                    "type": "gat",
                    "hidden_dim": 16,
                    "time_emb_dim": 8,
                    "num_layers": 2,
                    "pooling": "max",
                    "timestep_conditioning": False,
                },
                "diffusion": {
                    "n_steps": 16,
                    "beta_start": 1e-4,
                    "beta_end": 1e-2,
                },
                "labels": {
                    "type": "score_regression",
                },
                "training": {
                    "epochs": 1,
                    "lr": 1e-3,
                    "log_every": 1,
                    "save_every": 0,
                    "save_dir": str(tmp_path / "models"),
                    "checkpoint_name": "graph_regressor_final.pt",
                },
            },
            f,
            sort_keys=False,
        )

    result = train_guidance_model_from_config(config_path)

    assert result.checkpoint_path.exists()

    checkpoint, guidance, guidance_n_vertices = load_sampling_guidance(
        result.checkpoint_path,
        device=torch.device("cpu"),
        kind="regressor",
        scale=1.0,
        num_steps=16,
    )

    assert guidance_n_vertices is None
    assert checkpoint["model_cfg"]["type"] == "gat"
    assert checkpoint["model_cfg"]["pooling"] == "max"
    assert checkpoint["model_cfg"]["timestep_conditioning"] is False

    graph_batch = build_polygon_graph_batch(torch.tensor([5, 7], dtype=torch.long))
    x_t = torch.randn((graph_batch.total_vertices, 2), dtype=torch.float32)
    t = torch.tensor([5, 11], dtype=torch.long)
    grad = guidance(x_t, t, graph_batch=graph_batch)

    assert grad.shape == x_t.shape


def test_train_guidance_model_creates_isolated_run_dirs(tmp_path):
    data_path = tmp_path / "mixed_polygons.npz"
    coords, num_vertices, score, _ = batch_variable_sizes(
        size_values=[5, 6, 7],
        size_probabilities=[0.3, 0.4, 0.3],
        num=12,
        seed=1,
        radial_sigma=0.18,
        angle_sigma=0.12,
        smooth_passes=3,
    )
    np.savez_compressed(
        data_path,
        coords=coords.astype(np.float32),
        num_vertices=num_vertices.astype(np.int32),
        score=score.astype(np.float32),
    )

    def write_config(path: Path, *, experiment_name: str, pooling: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "seed": 0,
                    "device": "cpu",
                    "experiment_name": experiment_name,
                    "data": {
                        "path": str(data_path),
                        "batch_size": 8,
                        "shuffle": True,
                        "num_workers": 0,
                    },
                    "classifier": {
                        "type": "gat",
                        "hidden_dim": 16,
                        "time_emb_dim": 8,
                        "num_layers": 2,
                        "pooling": pooling,
                        "timestep_conditioning": True,
                    },
                    "diffusion": {
                        "n_steps": 16,
                        "beta_start": 1e-4,
                        "beta_end": 1e-2,
                    },
                    "labels": {
                        "type": "score_regression",
                    },
                    "training": {
                        "epochs": 0,
                        "lr": 1e-3,
                        "log_every": 1,
                        "save_every": 0,
                        "save_dir": str(tmp_path / "models"),
                        "checkpoint_name": "regressor_final.pt",
                    },
                },
                f,
                sort_keys=False,
            )

    config_a = tmp_path / "train_guidance_a.yaml"
    config_b = tmp_path / "train_guidance_b.yaml"
    write_config(config_a, experiment_name="guidance-a", pooling="avg")
    write_config(config_b, experiment_name="guidance-b", pooling="max")

    result_a = train_guidance_model_from_config(config_a)
    result_b = train_guidance_model_from_config(config_b)

    assert result_a.run_name != result_b.run_name
    assert result_a.model_dir != result_b.model_dir
    assert result_a.checkpoint_path != result_b.checkpoint_path
    assert result_a.checkpoint_path.exists()
    assert result_b.checkpoint_path.exists()
    assert result_a.log_path.exists()
    assert result_b.metrics_path.exists()

    checkpoint_a = torch.load(result_a.checkpoint_path, map_location="cpu")
    checkpoint_b = torch.load(result_b.checkpoint_path, map_location="cpu")

    assert checkpoint_a["run_name"] == result_a.run_name
    assert checkpoint_b["run_name"] == result_b.run_name
    assert Path(checkpoint_a["config_path"]).name == config_a.name
    assert Path(checkpoint_b["config_path"]).name == config_b.name
    assert checkpoint_a["model_cfg"]["pooling"] == "avg"
    assert checkpoint_b["model_cfg"]["pooling"] == "max"
