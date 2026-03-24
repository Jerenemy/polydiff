import torch

from polydiff.sampling.guidance import ScheduledGuidance, GuidanceSchedule, load_sampling_guidance


class ConstantGuidance:
    def __call__(self, x_t, t, *, graph_batch=None):
        del t, graph_batch
        return torch.ones_like(x_t)


def test_scheduled_guidance_masks_early_steps_for_late_schedule():
    guidance = ScheduledGuidance(
        guidance=ConstantGuidance(),
        schedule=GuidanceSchedule(kind="late", num_steps=10),
    )
    x_t = torch.ones((2, 8), dtype=torch.float32)

    early = guidance(x_t, torch.tensor([9, 9], dtype=torch.long))
    late = guidance(x_t, torch.tensor([0, 0], dtype=torch.long))

    assert torch.count_nonzero(early) == 0
    assert torch.allclose(late, torch.ones_like(late))


def test_load_sampling_guidance_wraps_regularity_with_requested_schedule():
    coords = torch.tensor(
        [[-1.0, -0.5, 0.9, -0.6, 1.0, 0.2, -0.7, 0.8]],
        dtype=torch.float32,
    )
    _, guidance, _ = load_sampling_guidance(
        None,
        device=torch.device("cpu"),
        kind="regularity",
        scale=1.0,
        num_steps=10,
        n_vertices=4,
        schedule="late",
    )

    grad_early = guidance(coords, torch.tensor([9], dtype=torch.long))
    grad_late = guidance(coords, torch.tensor([0], dtype=torch.long))

    assert torch.count_nonzero(grad_early) == 0
    assert torch.linalg.norm(grad_late) > 0.0
