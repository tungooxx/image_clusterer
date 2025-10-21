import sys, warnings
from .core import ClusterConfig, cluster_images
from loguru import logger
from typing import List, Tuple, Optional, Dict
import os
# optional: configure once (rotates at 50 MB, keep 3 files)
def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    logger.remove()
    logger.add(sys.stderr, level=level)
    if log_file:
        logger.add(log_file, level=level, rotation="50 MB", retention=3, enqueue=True)


def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(description="Image clustering with timm + UMAP + KMeans/HDBSCAN")
    p.add_argument("--input_dirs", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--id_regex", default=r"(M\d+)")
    p.add_argument("--img_exts", nargs="*", default=[".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"])
    p.add_argument("--model_name", default="maxvit_base_tf_384")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--umap_n_components", type=int, default=64)
    p.add_argument("--umap_n_neighbors", type=int, default=20)
    p.add_argument("--umap_min_dist", type=float, default=0.05)
    p.add_argument("--umap_metric", default="cosine")

    p.add_argument("--algo", choices=["kmeans","hdbscan"], default="kmeans")
    p.add_argument("--n_clusters", type=int, default=10)

    p.add_argument("--hdb_min_cluster_size", type=int, default=8)
    p.add_argument("--hdb_min_samples", type=int, default=1)
    p.add_argument("--hdb_metric", default="euclidean")

    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--no_normalize_feats", action="store_true")
    p.add_argument("--no_copy", action="store_true")
    p.add_argument("--folder_prefix", default="cluster_")
    p.add_argument("--device", choices=["cpu","cuda"], default=None)
    p.add_argument("--quiet", action="store_true")
    return p

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    ap = _build_argparser()
    a = ap.parse_args()
    cfg = ClusterConfig(
        input_dirs=a.input_dirs, output_dir=a.output_dir, id_regex=a.id_regex,
        img_exts=tuple(a.img_exts), model_name=a.model_name, pretrained=not a.no_pretrained,
        img_size=a.img_size, batch_size=a.batch_size,
        umap_n_components=a.umap_n_components, umap_n_neighbors=a.umap_n_neighbors,
        umap_min_dist=a.umap_min_dist, umap_metric=a.umap_metric,
        algo=a.algo, n_clusters=a.n_clusters,
        hdb_min_cluster_size=a.hdb_min_cluster_size, hdb_min_samples=a.hdb_min_samples,
        hdb_metric=a.hdb_metric,
        random_seed=a.random_seed, normalize_feats=not a.no_normalize_feats,
        copy_files=not a.no_copy, folder_prefix=a.folder_prefix,
        device=a.device, verbose=not a.quiet,
    )
    setup_logging(
        level="DEBUG" if not a.quiet else "WARNING",
        log_file=os.path.join(a.output_dir, "run.log"),
    )

    try:
        cluster_images(cfg)
    except Exception:
        logger.exception("Fatal error during clustering")
        sys.exit(1)
