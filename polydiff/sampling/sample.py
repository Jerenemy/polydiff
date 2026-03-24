"""Sampling entrypoint for polygon diffusion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import paths
from ..runs import (
    create_sampling_run_paths,
    infer_run_name_from_checkpoint_path,
    latest_model_run_dir,
    resolve_model_run_dir,
    slugify,
    write_sampling_run_files,
)
from ..utils.runtime import device_from_config, load_yaml_config, resolve_project_path, set_seed
from ..restoration import RestorationTrajectoryRecorder
from .guidance import CompositeGuidance, load_sampling_guidance
from .runtime import (
    DEFAULT_ANIMATION_DIR_NAME,
    DEFAULT_SAMPLES_OUT_NAME,
    GuidanceOptions,
    animation_sample_indices,
    load_diffusion_from_checkpoint,
    resolve_sampling_request,
    save_animations,
    write_sampling_diagnostics,
)


@dataclass(frozen=True, slots=True)
class SampleCliOverrides:
    run: str | None = None
    enable_animation: bool = False
    animation_out_path: str | None = None
    animation_count: int | None = None
    animation_sample_index: int | None = None
    animation_max_frames: int | None = None
    animation_fps: int | None = None


def build_sampling_run_label(cfg: dict[str, object], sampling_cfg: dict[str, object]) -> str:
    parts = [str(cfg.get("experiment_name", "polydiff-sample"))]
    guidance_cfg = sampling_cfg.get("guidance")
    if isinstance(guidance_cfg, dict) and bool(guidance_cfg.get("enabled", False)):
        components_cfg = guidance_cfg.get("components")
        if isinstance(components_cfg, list):
            kinds = [
                str(component_cfg.get("kind", "")).lower()
                for component_cfg in components_cfg
                if isinstance(component_cfg, dict) and bool(component_cfg.get("enabled", True))
            ]
            kinds = list(dict.fromkeys(kind for kind in kinds if kind))
            if kinds:
                parts.append("-".join(kinds))
        else:
            parts.append(str(guidance_cfg.get("kind", "classifier")).lower())
        parts.append("guided")
    else:
        parts.append("unguided")

    n_steps = sampling_cfg.get("n_steps")
    if n_steps is not None:
        parts.append(f"{int(n_steps)}steps")

    return "-".join(slugify(part) for part in parts if str(part).strip())


def resolve_checkpoint_from_run_selection(
    model_cfg: dict[str, object] | None,
    *,
    run_override: str | None,
) -> tuple[Path, str | None]:
    if model_cfg is None:
        model_cfg = {}
    if not isinstance(model_cfg, dict):
        raise ValueError("model config must be a mapping if provided")

    run_value = run_override if run_override is not None else model_cfg.get("run")
    checkpoint_value = model_cfg.get("checkpoint")

    if run_value is not None:
        run_dir = resolve_model_run_dir(str(run_value))
        checkpoint_path = run_dir / "diffusion_final.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No diffusion_final.pt found in {run_dir}")
        return checkpoint_path, run_dir.name

    if checkpoint_value is None or str(checkpoint_value) == "models/diffusion_final.pt":
        try:
            run_dir = latest_model_run_dir()
            checkpoint_path = run_dir / "diffusion_final.pt"
            if checkpoint_path.exists():
                return checkpoint_path, run_dir.name
        except FileNotFoundError:
            pass

    if checkpoint_value is None:
        checkpoint_path = resolve_project_path("models/diffusion_final.pt")
    else:
        checkpoint_path = resolve_project_path(str(checkpoint_value))
    return checkpoint_path, infer_run_name_from_checkpoint_path(checkpoint_path)


def _checkpoint_vertex_distribution(checkpoint: dict[str, object], *, fallback_size: int) -> tuple[np.ndarray, np.ndarray]:
    summary = checkpoint.get("training_data_summary")
    if isinstance(summary, dict):
        histogram = summary.get("vertex_count_histogram")
        if isinstance(histogram, dict) and histogram:
            sizes = np.asarray(sorted(int(key) for key in histogram.keys()), dtype=np.int32)
            counts = np.asarray([int(histogram[str(size)]) if str(size) in histogram else int(histogram[size]) for size in sizes], dtype=np.float64)
            probs = counts / counts.sum()
            return sizes, probs.astype(np.float64)

    n_vertices = checkpoint.get("n_vertices")
    if n_vertices is not None:
        return np.asarray([int(n_vertices)], dtype=np.int32), np.asarray([1.0], dtype=np.float64)
    return np.asarray([int(fallback_size)], dtype=np.int32), np.asarray([1.0], dtype=np.float64)


def _resolve_sample_num_vertices(
    *,
    checkpoint: dict[str, object],
    request_num_samples: int,
    model_type: str,
    fallback_size: int,
    requested_distribution: object,
    seed: int,
) -> np.ndarray:
    if requested_distribution is not None:
        values = np.asarray(requested_distribution.values, dtype=np.int32)
        probabilities = np.asarray(requested_distribution.probabilities, dtype=np.float64)
    else:
        values, probabilities = _checkpoint_vertex_distribution(checkpoint, fallback_size=fallback_size)

    if model_type == "mlp":
        if values.shape[0] != 1:
            raise ValueError("MLP checkpoints only support a single fixed polygon size during sampling")
        if int(values[0]) != int(fallback_size):
            raise ValueError(
                f"MLP checkpoint expects n_vertices={int(fallback_size)}, got requested size {int(values[0])}"
            )
        return np.full((request_num_samples,), int(values[0]), dtype=np.int32)

    rng = np.random.default_rng(seed)
    return rng.choice(values, size=request_num_samples, replace=True, p=probabilities).astype(np.int32)


def _resolved_guidance_config(guidance: GuidanceOptions) -> dict[str, object]:
    if not guidance.enabled:
        return {"enabled": False}
    if len(guidance.components) == 1:
        component = guidance.components[0]
        return {
            "enabled": True,
            **component.to_config_dict(),
        }
    return {
        "enabled": True,
        "components": [component.to_config_dict() for component in guidance.components],
    }


def sample_from_config(config_path: Path, *, cli_overrides: SampleCliOverrides | None = None) -> None:
    cli_overrides = cli_overrides or SampleCliOverrides()
    cfg = load_yaml_config(config_path)

    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    device = device_from_config(cfg)
    model_cfg = cfg.get("model") or {}
    sampling_cfg = cfg.get("sampling") or {}
    checkpoint_path, requested_run_name = resolve_checkpoint_from_run_selection(
        model_cfg,
        run_override=cli_overrides.run,
    )
    checkpoint, diffusion, diffusion_config, max_vertices = load_diffusion_from_checkpoint(
        checkpoint_path,
        device=device,
    )
    checkpoint_model_cfg = dict(checkpoint.get("model_cfg", {}))
    checkpoint_model_cfg.setdefault("type", "gat")
    model_type = str(checkpoint_model_cfg.get("type", "gat")).lower()
    run_name = str(checkpoint.get("run_name") or requested_run_name or infer_run_name_from_checkpoint_path(checkpoint_path) or "")
    sample_run_paths = None
    if run_name:
        sample_run_paths = create_sampling_run_paths(
            run_name=run_name,
            label=build_sampling_run_label(cfg, sampling_cfg),
        )
        default_out_path = sample_run_paths.processed_dir / DEFAULT_SAMPLES_OUT_NAME
        default_animation_out_path = sample_run_paths.media_dir / DEFAULT_ANIMATION_DIR_NAME
        print(
            f"[sample] using run {run_name} from checkpoint {checkpoint_path} "
            f"-> sampling subdir {sample_run_paths.sample_run_name}"
        )
    else:
        default_out_path = paths.PROCESSED_DATA_DIR / DEFAULT_SAMPLES_OUT_NAME
        default_animation_out_path = paths.PROCESSED_DATA_DIR / "media" / DEFAULT_ANIMATION_DIR_NAME
        print(f"[sample] using checkpoint {checkpoint_path}")

    request = resolve_sampling_request(
        sampling_cfg,
        checkpoint_n_steps=diffusion_config.n_steps,
        default_out_path=default_out_path,
        default_animation_out_path=default_animation_out_path,
        enable_animation=cli_overrides.enable_animation,
        animation_out_path=cli_overrides.animation_out_path,
        animation_count=cli_overrides.animation_count,
        animation_sample_index=cli_overrides.animation_sample_index,
        animation_max_frames=cli_overrides.animation_max_frames,
        animation_fps=cli_overrides.animation_fps,
    )
    restoration_recorder = None if request.restoration is None else RestorationTrajectoryRecorder(request.restoration)
    request.out_path.parent.mkdir(parents=True, exist_ok=True)
    if sample_run_paths is not None:
        resolved_cfg = {
            **cfg,
            "seed": seed,
            "device": str(device),
            "model": {
                "checkpoint": str(checkpoint_path),
                "run": run_name,
            },
            "sampling": {
                **sampling_cfg,
                "num_samples": request.num_samples,
                "n_steps": request.n_steps,
                "out_path": str(request.out_path),
                "guidance": _resolved_guidance_config(request.guidance),
                "diagnostics": {
                    **(sampling_cfg.get("diagnostics") or {}),
                    "enabled": request.diagnostics.enabled,
                    "out_path": None if request.diagnostics.out_path is None else str(request.diagnostics.out_path),
                    "reference_data_path": (
                        None
                        if request.diagnostics.reference_data_path is None
                        else str(request.diagnostics.reference_data_path)
                    ),
                },
            },
            "model_run_name": run_name,
            "sample_run_name": sample_run_paths.sample_run_name,
        }
        if request.size_distribution is not None:
            resolved_cfg["sampling"]["size_distribution"] = {
                "values": list(request.size_distribution.values),
                "probabilities": list(request.size_distribution.probabilities),
            }
        if request.animation is not None:
            resolved_cfg["sampling"]["animation"] = {
                **(sampling_cfg.get("animation") or {}),
                "out_path": str(request.animation.out_path),
                "sample_index": request.animation.sample_index,
                "count": request.animation.count,
                "max_frames": request.animation.max_frames,
                "fps": request.animation.fps,
            }
        if request.restoration is not None:
            resolved_cfg["sampling"]["restoration"] = {
                "enabled": True,
                **request.restoration.to_dict(),
            }
        write_sampling_run_files(
            sample_run_paths,
            config=resolved_cfg,
            config_path=config_path,
            extra_metadata={
                "checkpoint_path": str(checkpoint_path),
                "model_run_name": run_name,
                "sample_out_path": str(request.out_path),
            },
        )

    sample_num_vertices = _resolve_sample_num_vertices(
        checkpoint=checkpoint,
        request_num_samples=request.num_samples,
        model_type=model_type,
        fallback_size=max_vertices,
        requested_distribution=request.size_distribution,
        seed=seed,
    )
    uniform_sample_size = int(sample_num_vertices[0]) if np.all(sample_num_vertices == sample_num_vertices[0]) else None

    guidance_terms = []
    if request.guidance.enabled:
        descriptions: list[str] = []
        for component in request.guidance.components:
            if model_type != "mlp" and component.kind in {"classifier", "regressor"} and uniform_sample_size is None:
                raise ValueError(
                    "checkpoint-backed classifier/regressor guidance requires a uniform polygon size batch; "
                    "use sampling.size_distribution with one size or switch to analytic guidance"
                )
            _, guidance, guidance_n_vertices = load_sampling_guidance(
                component.checkpoint_path,
                device=device,
                kind=component.kind,
                scale=component.scale,
                num_steps=request.n_steps,
                n_vertices=uniform_sample_size,
                target_class=component.target_class,
                target_value=component.target_value,
                alpha=component.alpha,
                beta=component.beta,
                gamma=component.gamma,
                min_timestep_weight=component.min_timestep_weight,
                timestep_power=component.timestep_power,
                restoration=request.restoration,
            )
            if (
                model_type == "mlp"
                and component.kind in {"classifier", "regressor"}
                and guidance_n_vertices != int(sample_num_vertices[0])
            ):
                raise ValueError(
                    f"guidance model n_vertices={guidance_n_vertices} does not match "
                    f"diffusion checkpoint n_vertices={int(sample_num_vertices[0])}"
                )
            guidance_terms.append(guidance)

            description = f"{component.kind}(scale={component.scale}"
            if component.kind == "regularity":
                description += (
                    f", alpha={component.alpha}, beta={component.beta}, gamma={component.gamma}"
                )
            elif component.kind in {"classifier", "regressor"}:
                description += f", checkpoint={component.checkpoint_path}"
            elif component.kind == "restoration":
                description += (
                    ", objective=dna_distance_after_protein_restoration"
                    f", min_timestep_weight={component.min_timestep_weight}"
                    f", timestep_power={component.timestep_power}"
                )
            if component.kind == "classifier":
                description += f", target_class={component.target_class}"
            elif component.target_value is not None:
                description += f", target_value={component.target_value}"
            description += ")"
            descriptions.append(description)
        print(f"[sample] enabled guidance terms: {', '.join(descriptions)}")
    guidance_grad = None if not guidance_terms else CompositeGuidance(tuple(guidance_terms))

    if model_type == "mlp":
        fixed_n_vertices = int(sample_num_vertices[0])
        if request.animation is None:
            samples = diffusion.p_sample_loop(
                (request.num_samples, fixed_n_vertices * 2),
                n_steps=request.n_steps,
                guidance_grad=guidance_grad,
                observer=None if restoration_recorder is None else restoration_recorder.observe,
            )
            trajectories = None
        else:
            samples, trajectories = diffusion.p_sample_loop_trajectories(
                (request.num_samples, fixed_n_vertices * 2),
                n_steps=request.n_steps,
                trajectory_indices=animation_sample_indices(request.animation),
                guidance_grad=guidance_grad,
                observer=None if restoration_recorder is None else restoration_recorder.observe,
            )
        graph_batch = None
    else:
        if request.animation is None:
            samples, graph_batch = diffusion.p_sample_loop_graph(
                sample_num_vertices.tolist(),
                n_steps=request.n_steps,
                guidance_grad=guidance_grad,
                observer=None if restoration_recorder is None else restoration_recorder.observe,
            )
            trajectories = None
        else:
            samples, graph_batch, trajectories = diffusion.p_sample_loop_graph_trajectories(
                sample_num_vertices.tolist(),
                n_steps=request.n_steps,
                trajectory_indices=animation_sample_indices(request.animation),
                guidance_grad=guidance_grad,
                observer=None if restoration_recorder is None else restoration_recorder.observe,
            )

    meta = dict(
        checkpoint=str(checkpoint_path),
        run_name=run_name or None,
        sample_run_name=None if sample_run_paths is None else sample_run_paths.sample_run_name,
        num_samples=request.num_samples,
        n_steps=request.n_steps,
        checkpoint_n_steps=diffusion_config.n_steps,
    )
    if request.size_distribution is not None:
        meta["requested_size_distribution"] = {
            "values": list(request.size_distribution.values),
            "probabilities": list(request.size_distribution.probabilities),
        }

    if graph_batch is None:
        n_vertices = int(sample_num_vertices[0])
        samples_np = samples.detach().cpu().numpy().reshape(request.num_samples, n_vertices, 2).astype(np.float32)
        sample_num_vertices_np = np.full((request.num_samples,), n_vertices, dtype=np.int32)
        np.savez_compressed(
            request.out_path,
            coords=samples_np,
            n=np.int32(n_vertices),
            meta=meta,
        )
    else:
        sample_num_vertices_np = graph_batch.num_vertices.detach().cpu().numpy().astype(np.int32)
        samples_raw = samples.detach().cpu().numpy().astype(np.float32)
        if np.all(sample_num_vertices_np == sample_num_vertices_np[0]):
            n_vertices = int(sample_num_vertices_np[0])
            samples_np = samples_raw.reshape(request.num_samples, n_vertices, 2)
            np.savez_compressed(
                request.out_path,
                coords=samples_np,
                n=np.int32(n_vertices),
                meta=meta,
            )
        else:
            samples_np = samples_raw
            np.savez_compressed(
                request.out_path,
                coords=samples_np,
                num_vertices=sample_num_vertices_np,
                meta=meta,
            )

    print(
        f"[sample] saved {request.out_path} with {request.num_samples} samples "
        f"(sampling n_steps={request.n_steps}, checkpoint n_steps={diffusion_config.n_steps})"
    )
    if trajectories is not None and request.animation is not None:
        animation_out_paths = save_animations(
            trajectories,
            None if graph_batch is not None else n_vertices,
            request.animation,
            restoration=request.restoration,
        )
        print(
            f"[sample] saved {len(animation_out_paths)} animation(s) under {animation_out_paths[0].parent} "
            f"(start_index={request.animation.sample_index}, count={request.animation.count}, "
            f"max_frames={request.animation.max_frames}, fps={request.animation.fps})"
        )

    write_sampling_diagnostics(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        samples_out_path=request.out_path,
        coords=samples_np,
        num_vertices=None if np.all(sample_num_vertices_np == sample_num_vertices_np[0]) else sample_num_vertices_np,
        options=request.diagnostics,
        sampling_n_steps=request.n_steps,
        sample_run_name=None if sample_run_paths is None else sample_run_paths.sample_run_name,
        restoration=request.restoration,
        restoration_trajectory=None if restoration_recorder is None else restoration_recorder.to_dict(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(paths.CONFIG_DIR / "sample_diffusion.yaml"),
        help="path to sampling config",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="run directory name or run number to sample from; defaults to the most recent models/run_*",
    )
    parser.add_argument(
        "--save_animation",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="save reverse diffusion GIFs; optionally provide an output path or directory",
    )
    parser.add_argument(
        "--animation_count",
        type=int,
        default=None,
        help="number of sample trajectories to save as GIFs starting from --animation_index",
    )
    parser.add_argument(
        "--animation_index",
        type=int,
        default=None,
        help="starting sample index within the batch for animation export",
    )
    parser.add_argument(
        "--animation_max_frames",
        type=int,
        default=None,
        help="maximum number of frames to encode in the GIF",
    )
    parser.add_argument(
        "--animation_fps",
        type=int,
        default=None,
        help="GIF playback rate",
    )
    args = parser.parse_args()
    sample_from_config(
        resolve_project_path(args.config),
        cli_overrides=SampleCliOverrides(
            run=args.run,
            enable_animation=args.save_animation is not None,
            animation_out_path=args.save_animation or None,
            animation_count=args.animation_count,
            animation_sample_index=args.animation_index,
            animation_max_frames=args.animation_max_frames,
            animation_fps=args.animation_fps,
        ),
    )


if __name__ == "__main__":
    main()
