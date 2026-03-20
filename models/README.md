# Model Artifacts

Store training outputs, checkpoints, and logs here.
Use subfolders per experiment if needed.

Current training writes:
- `train.log`: human-readable console/file log
- `train_metrics.jsonl`: structured metrics for plotting and debugging
- `diffusion_step_*.pt` / `diffusion_final.pt`: checkpoints with training-data summary metadata
