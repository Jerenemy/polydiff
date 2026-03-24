# Data

- `raw/`: generated polygon datasets
- `processed/`: sampling outputs, diagnostics, notebook figures, and other derived artifacts

Polygon dataset formats:

- fixed-size datasets: dense `coords` with shape `(num_polygons, n_vertices, 2)` plus scalar `n`
- variable-size datasets: ragged `coords` with shape `(total_vertices, 2)` plus per-polygon `num_vertices`

Current generator behavior:

- `python -m polydiff.data.gen_polygons --n ...` writes the fixed-size dense format
- `python -m polydiff.data.gen_polygons --size_values ... --size_probabilities ...` writes the ragged variable-size format

Current sampler behavior:

- fixed-size outputs stay dense when all sampled polygons have the same size
- mixed-size GAT/GCN outputs are written raggedly as `coords` + `num_vertices`

Diagnostics and plotting code in `polydiff.data` understands both formats.

Keep large files out of git unless using Git LFS.
