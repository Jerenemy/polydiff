import json
from pathlib import Path

import torch
import yaml

from predict_binding_affinity.studies.run import run_surrogate_study_from_config
from predict_binding_affinity.surrogate_data import SurrogateGraph, collate_surrogate_graphs
from predict_binding_affinity.surrogate_models import LigandContextSurrogateModel
from predict_binding_affinity.train_surrogate_model import train_surrogate_model_from_config


def test_surrogate_pooling_is_permutation_invariant():
    ligand_pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [0.5, 0.9, 0.0],
        ],
        dtype=torch.float32,
    )
    pocket_pos = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [2.2, 0.3, 0.1],
        ],
        dtype=torch.float32,
    )
    pos = torch.cat([ligand_pos, pocket_pos], dim=0)
    z = torch.tensor([6, 7, 8, 6, 8], dtype=torch.long)
    node_type = torch.tensor([1, 1, 1, 0, 0], dtype=torch.long)
    graph = SurrogateGraph(
        pos=pos,
        z=z,
        node_type=node_type,
        y=torch.tensor(-7.5),
        ligand_size=3,
        metadata={},
    )
    permutation = torch.tensor([2, 0, 1, 3, 4], dtype=torch.long)
    graph_permuted = SurrogateGraph(
        pos=pos.index_select(0, permutation),
        z=z.index_select(0, permutation),
        node_type=node_type.index_select(0, permutation),
        y=torch.tensor(-7.5),
        ligand_size=3,
        metadata={},
    )

    for pooling in ("avg", "max", "sum"):
        batch_a = collate_surrogate_graphs([graph], r_ligand=4.0, r_cross=4.0)
        batch_b = collate_surrogate_graphs([graph_permuted], r_ligand=4.0, r_cross=4.0)
        batch_a.t = torch.tensor([12], dtype=torch.long)
        batch_b.t = torch.tensor([12], dtype=torch.long)

        model = LigandContextSurrogateModel(
            hidden_dim=16,
            num_layers=2,
            num_rbf=8,
            dropout=0.0,
            pooling=pooling,
            timestep_conditioning=True,
            time_emb_dim=8,
        )
        model.eval()
        pred_a = model(batch_a)
        pred_b = model(batch_b)
        assert torch.allclose(pred_a, pred_b, atol=1e-6)


def test_train_surrogate_model_from_config_writes_outputs(tmp_path):
    records_path, pocket_path = _write_synthetic_surrogate_inputs(tmp_path, num_records=18)
    config_path = tmp_path / "train_surrogate.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "seed": 0,
                "device": "cpu",
                "experiment_name": "synthetic-surrogate",
                "data": {
                    "pt_path": str(records_path),
                    "pdb_path": str(pocket_path),
                    "vina_mode": "score_only",
                    "val_fraction": 0.2,
                    "test_fraction": 0.2,
                    "r_ligand": 4.0,
                    "r_cross": 5.0,
                },
                "noise": {
                    "enabled": True,
                    "n_steps": 50,
                    "beta_start": 1e-4,
                    "beta_end": 1e-2,
                },
                "model": {
                    "hidden_dim": 16,
                    "num_layers": 2,
                    "num_rbf": 8,
                    "dropout": 0.0,
                    "pooling": "avg",
                    "timestep_conditioning": True,
                    "time_emb_dim": 8,
                },
                "training": {
                    "batch_size": 4,
                    "epochs": 2,
                    "lr": 1e-3,
                    "eval_repeats": 1,
                    "log_every": 1,
                    "save_dir": str(tmp_path / "models"),
                    "processed_dir": str(tmp_path / "processed"),
                },
            },
            f,
            sort_keys=False,
        )

    result = train_surrogate_model_from_config(config_path)

    assert result.checkpoint_path.exists()
    assert result.final_checkpoint_path.exists()
    assert result.history_path.exists()
    assert result.metrics_path.exists()

    metrics_payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert "test_metrics" in metrics_payload
    assert "mae" in metrics_payload["test_metrics"]


def test_run_surrogate_study_writes_summary_and_figures(tmp_path):
    records_path, pocket_path = _write_synthetic_surrogate_inputs(tmp_path, num_records=18)
    train_config_path = tmp_path / "train_surrogate.yaml"
    with open(train_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "seed": 0,
                "device": "cpu",
                "experiment_name": "study-surrogate",
                "data": {
                    "pt_path": str(records_path),
                    "pdb_path": str(pocket_path),
                    "vina_mode": "score_only",
                    "val_fraction": 0.2,
                    "test_fraction": 0.2,
                    "r_ligand": 4.0,
                    "r_cross": 5.0,
                },
                "noise": {
                    "enabled": True,
                    "n_steps": 50,
                    "beta_start": 1e-4,
                    "beta_end": 1e-2,
                },
                "model": {
                    "hidden_dim": 16,
                    "num_layers": 2,
                    "num_rbf": 8,
                    "dropout": 0.0,
                    "pooling": "avg",
                    "timestep_conditioning": True,
                    "time_emb_dim": 8,
                },
                "training": {
                    "batch_size": 4,
                    "epochs": 2,
                    "lr": 1e-3,
                    "eval_repeats": 1,
                    "log_every": 1,
                    "save_dir": str(tmp_path / "models"),
                    "processed_dir": str(tmp_path / "processed"),
                },
            },
            f,
            sort_keys=False,
        )

    study_config_path = tmp_path / "surrogate_study.yaml"
    with open(study_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "study": {
                    "name": "synthetic-surrogate-study",
                    "root_dir": str(tmp_path / "studies"),
                },
                "cases": [
                    {
                        "name": "avg-with-t",
                        "kind": "train_surrogate_model",
                        "config": str(train_config_path),
                        "overrides": {
                            "experiment_name": "avg-with-t",
                            "model.pooling": "avg",
                            "model.timestep_conditioning": True,
                        },
                        "tags": {
                            "pooling": "avg",
                            "timestep_conditioning": True,
                        },
                    },
                    {
                        "name": "avg-no-t",
                        "kind": "train_surrogate_model",
                        "config": str(train_config_path),
                        "overrides": {
                            "experiment_name": "avg-no-t",
                            "model.pooling": "avg",
                            "model.timestep_conditioning": False,
                        },
                        "tags": {
                            "pooling": "avg",
                            "timestep_conditioning": False,
                        },
                    },
                ],
            },
            f,
            sort_keys=False,
        )

    report = run_surrogate_study_from_config(study_config_path)

    assert Path(report["summary_csv_path"]).exists()
    assert Path(report["summary_json_path"]).exists()
    assert Path(report["interpretation_guide_path"]).exists()
    assert Path(report["figure_paths"]["surrogate_metric_panel"]).exists()
    assert Path(report["figure_paths"]["surrogate_validation_curves"]).exists()


def _write_synthetic_surrogate_inputs(tmp_path: Path, *, num_records: int) -> tuple[Path, Path]:
    pocket_path = tmp_path / "pocket.pdb"
    pocket_path.write_text(
        "\n".join(
            [
                "ATOM      1  C   GLY A   1       1.800   0.000   0.000  1.00  0.00           C",
                "ATOM      2  O   GLY A   1       2.000   0.600   0.100  1.00  0.00           O",
                "ATOM      3  N   GLY A   1       1.700  -0.700  -0.100  1.00  0.00           N",
                "",
            ]
        ),
        encoding="utf-8",
    )

    generator = torch.Generator().manual_seed(0)
    records = []
    target_center = torch.tensor([1.9, 0.0, 0.0], dtype=torch.float32)
    atomic_choices = torch.tensor([6, 7, 8, 16], dtype=torch.long)
    for index in range(num_records):
        ligand_size = 4 + int(index % 3)
        center = torch.randn((3,), generator=generator) * 0.25
        pos = center + torch.randn((ligand_size, 3), generator=generator) * 0.35
        z = atomic_choices[torch.randint(0, len(atomic_choices), (ligand_size,), generator=generator)]
        affinity = (
            -float((pos.mean(dim=0) - target_center).norm().item())
            - 0.025 * float(z.float().mean().item())
            - 0.010 * float(ligand_size)
        )
        records.append(
            {
                "ligand_pos": pos,
                "ligand_atomic_numbers": z,
                "vina": {"score_only": [{"affinity": affinity}]},
                "smiles": f"SYNTH-{index:03d}",
            }
        )

    records_path = tmp_path / "records.pt"
    torch.save({"all_results": records}, records_path)
    return records_path, pocket_path
