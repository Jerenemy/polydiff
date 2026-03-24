# Model Artifacts

Store training outputs, checkpoints, and logs here.
Use subfolders per experiment if needed.

Current training writes:
- `train.log`: human-readable console/file log
- `train_metrics.jsonl`: structured metrics for plotting and debugging
- `diffusion_step_*.pt` / `diffusion_final.pt`: checkpoints with training-data summary metadata

Checkpoint metadata now distinguishes fixed-size and variable-size runs:

- fixed-size runs store `n_vertices`
- GAT/GCN checkpoints also store `max_vertices` for model reconstruction
- training summaries include the empirical vertex-count histogram used as the default sampling prior for variable-size GAT/GCN sampling
