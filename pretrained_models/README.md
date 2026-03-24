# Pretrained Models

Place externally trained checkpoints here.

Recommended metadata expectations:

- fixed-size checkpoints should include `n_vertices`
- variable-size GAT/GCN checkpoints should include `max_vertices`
- if available, include `training_data_summary.vertex_count_histogram` so mixed-size sampling can reuse the original empirical size prior
