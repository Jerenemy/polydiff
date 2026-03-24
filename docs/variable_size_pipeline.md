# Variable-Size Polygon Pipeline

This document explains the current variable-size polygon support in `polydiff`.

## Scope

Variable-size polygons are supported for:

- `gat`
- `gcn`

They are not supported for:

- `mlp` diffusion denoisers
- learned guidance-model training (`train_guidance_model.py`), which is still fixed-size and MLP-only

The implementation does **not** use padded polygon storage. Instead, it uses ragged storage on disk and graph-native batches in memory.

## Storage Format

There are now two dataset/sample formats:

### Fixed-size

- `coords`: shape `(num_polygons, n_vertices, 2)`
- `n`: scalar integer

### Variable-size

- `coords`: shape `(total_vertices, 2)`
- `num_vertices`: shape `(num_polygons,)`

Interpretation:

- polygon `i` occupies a contiguous slice of `coords`
- the slice length is `num_vertices[i]`

Helpers for both formats live in `polydiff/data/polygon_dataset.py`.

## Data Generation

Fixed-size generation:

```bash
python -m polydiff.data.gen_polygons --n 6 --num 10000 --out hexagons.npz
```

Variable-size generation:

```bash
python -m polydiff.data.gen_polygons \
  --size_values 5,6,7,8 \
  --size_probabilities 0.1,0.2,0.4,0.3 \
  --num 10000 \
  --out mixed_polygons.npz
```

If `--size_probabilities` is omitted, the generator uses a uniform distribution over `--size_values`.

## Training

Training behavior depends on `model.type`:

- `mlp`: loads a fixed-size dense tensor and trains on flattened `(B, 2n)` inputs
- `gat` / `gcn`: loads polygons as ragged node sets and collates batches into concatenated graph batches

The graph batch stores:

- concatenated node coordinates
- `num_vertices` per polygon
- `ptr` offsets
- `graph_index` mapping each node to its polygon
- `node_index` within each polygon cycle

That representation is used throughout:

- the denoiser forward pass
- diffusion loss computation
- reverse diffusion sampling
- analytic guidance

## Diffusion and Sampling

For variable-size GAT/GCN sampling:

1. sample a polygon size for each requested sample
2. keep that size fixed for the entire reverse trajectory
3. build one ragged graph batch from all requested polygons
4. denoise all nodes in parallel with message passing over the cycle graph

The default size prior is the empirical vertex-count histogram stored in the checkpoint’s training summary.

You can override it with:

```yaml
sampling:
  size_distribution:
    values: [5, 6, 7, 8]
    probabilities: [0.1, 0.2, 0.4, 0.3]
```

If every sampled polygon has the same size, the saved sample file stays dense for compatibility.
If sizes differ, the saved sample file uses the ragged format.

## Guidance Compatibility

Analytic guidance:

- `regularity`: works for mixed-size GAT/GCN batches
- `area`: works for mixed-size GAT/GCN batches

Checkpoint-backed learned guidance:

- `classifier`: requires a uniform sampled size
- `regressor`: requires a uniform sampled size

Reason:

- learned guidance models are still MLPs over flattened `(B, 2n)` inputs

## Checkpoints

Checkpoint metadata now uses:

- `n_vertices` for fixed-size runs
- `max_vertices` for GAT/GCN reconstruction
- `training_data_summary.vertex_count_histogram` as the default sampling prior

## Diagnostics and Plotting

The following utilities now understand both dense and ragged polygon files:

- `polydiff.data.diagnostics`
- `polydiff.data.plot_polygons`

Summaries for variable-size datasets include:

- `min_vertices`
- `max_vertices`
- `mean_vertices`
- `vertex_count_histogram`

## Current Limitations

- the comparison notebooks still contain fixed-size assumptions and are not yet fully rewritten for mixed-size datasets
- learned guidance-model training is still fixed-size only
- `mlp` diffusion remains fixed-size only by design
