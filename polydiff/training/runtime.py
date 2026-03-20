"""Training-side logging, checkpointing, and diagnostics helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from ..data.diagnostics import (
    compare_polygon_summaries,
    format_polygon_delta_summary,
    format_polygon_summary,
    json_ready,
    summarize_polygon_dataset,
)
from ..models.diffusion import Diffusion, DiffusionConfig


@dataclass(frozen=True, slots=True)
class TrainingOptions:
    epochs: int
    lr: float
    log_every: int
    save_every: int
    save_dir: Path
    sample_diagnostics_every: int
    sample_diagnostics_num_samples: int
    sample_diagnostics_n_steps: int | None
    sample_diagnostics_seed: int | None


class _HandlerTargetFilter(logging.Filter):
    def __init__(self, *, attr_name: str, default: bool) -> None:
        super().__init__()
        self.attr_name = attr_name
        self.default = default

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return bool(getattr(record, self.attr_name, self.default))


def resolve_training_options(training_cfg: dict[str, Any], *, batch_size: int) -> TrainingOptions:
    sample_diag_n_steps = training_cfg.get("sample_diagnostics_n_steps")
    sample_diag_seed = training_cfg.get("sample_diagnostics_seed")
    return TrainingOptions(
        epochs=int(training_cfg.get("epochs", 10)),
        lr=float(training_cfg.get("lr", 1e-4)),
        log_every=int(training_cfg.get("log_every", 100)),
        save_every=int(training_cfg.get("save_every", 1000)),
        save_dir=Path(training_cfg.get("save_dir", "models")),
        sample_diagnostics_every=int(training_cfg.get("sample_diagnostics_every", 0)),
        sample_diagnostics_num_samples=int(training_cfg.get("sample_diagnostics_num_samples", batch_size)),
        sample_diagnostics_n_steps=None if sample_diag_n_steps is None else int(sample_diag_n_steps),
        sample_diagnostics_seed=None if sample_diag_seed is None else int(sample_diag_seed),
    )


def setup_train_logger(save_dir: Path) -> tuple[logging.Logger, Path, Path]:
    log_path = save_dir / "train.log"
    metrics_path = save_dir / "train_metrics.jsonl"

    logger_name = f"polydiff.training.{save_dir.resolve()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    file_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_formatter = logging.Formatter("%(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    stream_handler.addFilter(_HandlerTargetFilter(attr_name="to_console", default=False))
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(_HandlerTargetFilter(attr_name="to_file", default=True))
    logger.addHandler(file_handler)

    return logger, log_path, metrics_path


def log_file_info(logger: logging.Logger, message: str, *args: object) -> None:
    logger.info(message, *args, extra={"to_console": False, "to_file": True})


def log_console_info(logger: logging.Logger, message: str, *args: object) -> None:
    logger.info(message, *args, extra={"to_console": True, "to_file": False})


def log_both_info(logger: logging.Logger, message: str, *args: object) -> None:
    logger.info(message, *args, extra={"to_console": True, "to_file": True})


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        json.dump(json_ready(record), f, ensure_ascii=True, sort_keys=True)
        f.write("\n")


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().item())
    return total ** 0.5


def param_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        total += float(param.detach().pow(2).sum().item())
    return total ** 0.5


def run_sample_diagnostics(
    diffusion: Diffusion,
    *,
    num_samples: int,
    data_dim: int,
    n_vertices: int,
    n_steps: int | None,
    seed: int | None,
    reference_summary: dict[str, float | int],
) -> tuple[dict[str, float | int], dict[str, float]]:
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = diffusion.model.training

    try:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        samples = diffusion.p_sample_loop((num_samples, data_dim), n_steps=n_steps)
        coords = samples.detach().cpu().numpy().reshape(num_samples, n_vertices, 2).astype(np.float32)
        summary = summarize_polygon_dataset(coords)
        deltas = compare_polygon_summaries(reference_summary, summary)
        return summary, deltas
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        diffusion.model.train(was_training)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    diffusion_config: DiffusionConfig,
    model_cfg: dict[str, Any],
    n_vertices: int,
    global_step: int,
    training_data_path: Path,
    training_data_summary: dict[str, float | int],
    config_path: Path,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "diffusion": asdict(diffusion_config),
        "model_cfg": model_cfg,
        "n_vertices": n_vertices,
        "global_step": int(global_step),
        "training_data_path": str(training_data_path),
        "training_data_summary": json_ready(training_data_summary),
        "config_path": str(config_path),
    }
    torch.save(checkpoint, path)


def log_training_start(
    *,
    logger: logging.Logger,
    metrics_path: Path,
    config_path: Path,
    seed: int,
    device: torch.device,
    data_path: Path,
    coords_shape: tuple[int, ...],
    model_cfg: dict[str, Any],
    diffusion_config: DiffusionConfig,
    training_cfg: dict[str, Any],
    training_data_summary: dict[str, float | int],
    log_path: Path,
) -> None:
    log_file_info(logger, "config path: %s", config_path)
    log_file_info(logger, "device: %s | seed: %d", device, seed)
    log_file_info(logger, "data path: %s | coords shape: %s", data_path, coords_shape)
    log_file_info(logger, "training data summary: %s", format_polygon_summary(training_data_summary))
    log_file_info(logger, "logs: %s | metrics: %s", log_path, metrics_path)
    append_jsonl(
        metrics_path,
        {
            "event": "train_start",
            "config_path": str(config_path),
            "seed": seed,
            "device": str(device),
            "data_path": str(data_path),
            "coords_shape": list(coords_shape),
            "model_cfg": model_cfg,
            "diffusion_cfg": asdict(diffusion_config),
            "training_cfg": training_cfg,
            "training_data_summary": training_data_summary,
        },
    )


def log_training_step(
    *,
    logger: logging.Logger,
    metrics_path: Path,
    epoch: int,
    step: int,
    ema_loss: float,
    grad_norm_value: float,
    param_norm_value: float,
    lr: float,
    steps_per_sec: float,
    loss_stats: dict[str, float],
) -> None:
    record = {
        "event": "train_step",
        "epoch": epoch,
        "step": step,
        "ema_loss": float(ema_loss),
        "grad_norm": float(grad_norm_value),
        "param_norm": float(param_norm_value),
        "lr": float(lr),
        "steps_per_sec": float(steps_per_sec),
        **loss_stats,
    }
    log_both_info(
        logger,
        "[train] epoch=%d step=%d loss=%.6f ema_loss=%.6f",
        epoch,
        step,
        float(loss_stats["loss"]),
        float(ema_loss),
    )
    append_jsonl(metrics_path, record)


def log_sample_diagnostic(
    *,
    logger: logging.Logger,
    metrics_path: Path,
    epoch: int,
    step: int,
    num_samples: int,
    n_steps: int,
    sample_summary: dict[str, float | int],
    sample_deltas: dict[str, float],
) -> None:
    log_file_info(logger, "[diag] generated sample summary: %s", format_polygon_summary(sample_summary))
    log_file_info(logger, "[diag] generated vs training: %s", format_polygon_delta_summary(sample_deltas))
    append_jsonl(
        metrics_path,
        {
            "event": "sample_diagnostic",
            "epoch": epoch,
            "step": step,
            "num_samples": num_samples,
            "n_steps": n_steps,
            "sample_summary": sample_summary,
            "delta_vs_training": sample_deltas,
        },
    )


def log_training_end(
    *,
    logger: logging.Logger,
    metrics_path: Path,
    final_checkpoint: Path,
    final_step: int,
) -> None:
    log_both_info(logger, "[train] saved final checkpoint %s", final_checkpoint)
    append_jsonl(
        metrics_path,
        {
            "event": "train_end",
            "final_checkpoint": str(final_checkpoint),
            "final_step": final_step,
        },
    )
