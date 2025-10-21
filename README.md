# image-clusterer

Image clustering utility using timm image encoders, UMAP for dimensionality reduction and KMeans or HDBSCAN for clustering.

This small tool finds images under one or more input directories, embeds them with a timm model, reduces features with UMAP and groups them into clusters. It can copy clustered images into output folders and write a CSV mapping.

## Features

- Pluggable timm model (default: `maxvit_base_tf_384`)
- Dimensionality reduction via UMAP
- Clustering with KMeans (default) or HDBSCAN (optional extra)
- CLI and Python API

## Requirements

- Python 3.8+
- torch
- torchvision
- timm
- umap-learn
- scikit-learn
- pandas
- pillow

Optional for HDBSCAN algorithm:

- hdbscan

Install with pip:

```powershell
git clone <url>
cd image_clusterer
# install manually:
pip install -e .
```


After installation a console script `image-clusterer` is available (see entry point `image_clusterer.cli:main`).

## CLI Usage

Basic example:

```powershell
image-clusterer --input_dirs C:\images\set1 C:\images\set2 --output_dir C:\out_clusters
```

Common options:

- `--input_dirs` (required): One or more input directories to search for images.
- `--output_dir` (required): Directory where cluster folders and CSV mapping will be written.
- `--model_name`: timm model name (default: `maxvit_base_tf_384`).
- `--no_pretrained`: don't load pretrained weights.
- `--img_size`: resize images to this square size before embedding (default 384).
- `--batch_size`: embedding batch size (default 32).
- `--algo`: `kmeans` (default) or `hdbscan`.
- `--n_clusters`: number of clusters for KMeans (default 10).
- `--hdb_min_cluster_size`, `--hdb_min_samples`, `--hdb_metric`: HDBSCAN params.
- `--no_normalize_feats`: disable feature normalization before UMAP.
- `--no_copy`: don't copy image files into cluster folders (CSV still written).
- `--folder_prefix`: prefix for generated cluster folders (default `cluster_`).
- `--device`: `cpu` or `cuda` (auto-chosen by default).
- `--quiet`: reduce output.

Example with HDBSCAN:

```powershell
image-clusterer --input_dirs C:\photos --output_dir C:\clusters --algo hdbscan --hdb_min_cluster_size 10
```

## Python API

You can also call the functionality from Python:

```python
from image_clusterer.core import ClusterConfig, cluster_images

cfg = ClusterConfig(
    input_dirs=["/path/to/images"],
    output_dir="/path/to/out",
    algo="kmeans",
    n_clusters=8,
)

df, cluster_to_id, csv_path = cluster_images(cfg)
```

Return values:

- `df`: pandas DataFrame describing each processed image (path, id, cluster)
- `cluster_to_id`: dict mapping cluster integer -> summarized id (or `noise`)
- `csv_path`: path to the written CSV mapping

## Tips & Troubleshooting

- If you get `No images found in input_dirs.`, verify paths and supported extensions (default: .jpg, .jpeg, .png, .bmp, .webp, .tif, .tiff).
- If using HDBSCAN and you see the runtime error about missing package, install with `pip install hdbscan` or use the `hdbscan` extra when packaging.
- For large datasets increase `batch_size` (if memory allows) and consider reducing `umap_n_components`.

## Development

Project entry points are defined in `image_clusterer.egg-info/entry_points.txt` during packaging. The CLI entry is `image-clusterer = image_clusterer.cli:main`.

Source files of interest:

- `image_clusterer/core.py` - main implementation and `ClusterConfig` dataclass
- `image_clusterer/cli.py` - command-line argument parsing and entry point

## License

This repository does not contain an explicit license file. Add a LICENSE if you intend to redistribute.
