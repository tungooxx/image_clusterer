import os, re, shutil, warnings
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch, timm
from torchvision import transforms
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from loguru import logger
import sys

# optional: configure once (rotates at 50 MB, keep 3 files)
# def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
#     logger.remove()
#     logger.add(sys.stderr, level=level)
#     if log_file:
#         logger.add(log_file, level=level, rotation="50 MB", retention=3, enqueue=True)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ---------------- CONFIG ----------------
@dataclass
class ClusterConfig:
    input_dirs: List[str]
    output_dir: str
    id_regex: str = r"(M\d+)"
    img_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    model_name: str = "maxvit_base_tf_384"
    pretrained: bool = True
    img_size: int = 384
    batch_size: int = 32
    umap_n_components: int = 64
    umap_n_neighbors: int = 20
    umap_min_dist: float = 0.05
    umap_metric: str = "cosine"

    # algo switch
    algo: str = "kmeans"  # "kmeans" | "hdbscan"

    # kmeans params
    n_clusters: int = 10

    # hdbscan params
    hdb_min_cluster_size: int = 8
    hdb_min_samples: Optional[int] = 1
    hdb_metric: str = "euclidean"

    random_seed: int = 42
    normalize_feats: bool = True
    copy_files: bool = True
    folder_prefix: str = "cluster_"
    device: Optional[str] = None
    verbose: bool = True

    def compile_pattern(self): return re.compile(self.id_regex, re.IGNORECASE)


# ---------------- CORE ----------------
def find_images(folders: List[str], img_exts: Tuple[str, ...]) -> List[str]:
    out = []
    for folder in folders:
        p = Path(folder)
        out.extend(str(f) for f in p.rglob("*") if f.suffix.lower() in img_exts)
    return out


def extract_id(fname: str, pat: re.Pattern) -> Optional[str]:
    m = pat.search(os.path.basename(fname))
    return m.group(1).upper() if m else None


def build_model(model_name: str, pretrained: bool, device: str):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0).eval().to(device)
    mean = model.default_cfg.get("mean", (0.485, 0.456, 0.406))
    std = model.default_cfg.get("std", (0.229, 0.224, 0.225))
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return model, tfm


def load_image_rgb(path: str, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None:
            im = im.resize((size, size), Image.BILINEAR)
        return np.array(im)


@torch.no_grad()
def embed_paths(paths: List[str], model, tfm, device: str, batch_size: int, size: int, verbose: bool):
    feats, valid_idx, batch = [], [], []
    iterator = tqdm(range(len(paths)), desc="Embedding", disable=not verbose)
    for i in iterator:
        p = paths[i]
        try:
            arr = load_image_rgb(p, size=size)
            x = tfm(arr)
            batch.append(x)
        except (UnidentifiedImageError, OSError, ValueError):
            continue
        if len(batch) == batch_size:
            t = torch.stack(batch).to(device)
            f = torch.nn.functional.normalize(model(t), dim=1).cpu().numpy()
            feats.append(f)
            valid_idx.extend(range(i - batch_size + 1, i + 1))
            batch = []
    if batch:
        t = torch.stack(batch).to(device)
        f = torch.nn.functional.normalize(model(t), dim=1).cpu().numpy()
        feats.append(f)
        last = len(batch)
        valid_idx.extend(range(len(paths) - last, len(paths)))
    return (np.vstack(feats) if feats else np.empty((0, 0), np.float32)), valid_idx


def ensure_dir(p: str): Path(p).mkdir(parents=True, exist_ok=True)


def _select_device(cfg: ClusterConfig) -> str:
    if cfg.device: return cfg.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def reduce_dim(feats: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    X = feats
    if cfg.normalize_feats:
        X = normalize(StandardScaler().fit_transform(X))
    nnei = max(2, min(cfg.umap_n_neighbors, len(X) - 1))
    reducer = umap.UMAP(
        n_components=cfg.umap_n_components,
        n_neighbors=nnei,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        init="random",
        random_state=cfg.random_seed,
    )
    return reducer.fit_transform(X)


def _cluster(Xr: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    if cfg.algo.lower() == "kmeans":
        k = int(max(1, min(cfg.n_clusters, len(Xr))))
        km = KMeans(n_clusters=k, n_init="auto", random_state=cfg.random_seed)
        return km.fit_predict(Xr)
    elif cfg.algo.lower() == "hdbscan":
        try:
            import hdbscan
        except ImportError:
            raise RuntimeError("hdbscan not installed. Install extra: pip install image-clusterer[hdbscan]")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg.hdb_min_cluster_size,
            min_samples=cfg.hdb_min_samples,
            metric=cfg.hdb_metric,
        )
        return clusterer.fit_predict(Xr)  # -1 = noise
    else:
        raise ValueError("algo must be 'kmeans' or 'hdbscan'")


def summarize_cluster_ids(df: pd.DataFrame) -> Dict[int, str]:
    mapping = {}
    for c in sorted(df.cluster.unique()):
        if c == -1:  # noise
            mapping[c] = "noise"
            continue
        ids = [i for i in df.loc[df.cluster == c, "id"].dropna().astype(str)]
        if not ids:
            mapping[c] = f"unknown_cluster_{c}";
            continue
        cnt = Counter([i.upper() for i in ids])
        top_id, top_n = cnt.most_common(1)[0]
        ties = [k for k, v in cnt.items() if v == top_n]
        mapping[c] = top_id if len(ties) == 1 else f"ambiguous_{c}"
    return mapping


def write_outputs(df: pd.DataFrame, cfg: ClusterConfig) -> str:
    ensure_dir(cfg.output_dir)
    labels = [c for c in sorted(df.cluster.unique()) if c != -1]
    cluster_to_folder = {c: f"{cfg.folder_prefix}{i + 1}" for i, c in enumerate(labels)}
    for _, r in df.iterrows():
        c = int(r.cluster)
        bucket = "noise" if c == -1 else cluster_to_folder[c]
        out_dir = os.path.join(cfg.output_dir, bucket)
        ensure_dir(out_dir)
        if cfg.copy_files:
            dst = os.path.join(out_dir, os.path.basename(r.path))
            try:
                shutil.copy2(r.path, dst)
            except Exception:
                pass
    rows = []
    for _, r in df.iterrows():
        c = int(r.cluster)
        bucket = "noise" if c == -1 else cluster_to_folder[c]
        rows.append({"src_path": r.path, "assigned_folder": bucket, "cluster": c, "known_id_in_name": r.id})
    map_csv = os.path.join(cfg.output_dir, "cluster_assignment.csv")
    pd.DataFrame(rows).to_csv(map_csv, index=False)
    return map_csv


def cluster_images(cfg: ClusterConfig):
    if cfg.verbose: logger.opt(lazy=True).info("Config: {}", lambda: cfg)
    device = _select_device(cfg)
    if cfg.verbose: logger.info("Device: {}", device)

    paths = find_images(cfg.input_dirs, cfg.img_exts)
    if not paths: raise RuntimeError("No images found in input_dirs.")
    pat = cfg.compile_pattern()
    df = pd.DataFrame([{"path": p, "id": extract_id(p, pat)} for p in paths])
    if cfg.verbose: logger.info("Total images discovered: {}", len(paths))

    if cfg.verbose: logger.info("Algorithm: {}", cfg.algo)

    model, tfm = build_model(cfg.model_name, cfg.pretrained, device)
    feats, valid_idx = embed_paths(df["path"].tolist(), model, tfm, device, cfg.batch_size, cfg.img_size, cfg.verbose)
    if feats.size == 0: raise RuntimeError("No embeddings produced.")
    df = df.iloc[valid_idx].reset_index(drop=True)
    if cfg.verbose: logger.info("Embedded images: {}", len(df))

    Xr = reduce_dim(feats, cfg)
    labels = _cluster(Xr, cfg)
    df["cluster"] = labels
    if cfg.verbose:
        counts = Counter(labels)
        logger.info("Cluster counts: {}", counts)

    cluster_to_id = summarize_cluster_ids(df)
    csv_path = write_outputs(df, cfg)
    if cfg.verbose: logger.success("Done. Output: {} | CSV: {}", cfg.output_dir, csv_path)
    return df, cluster_to_id, csv_path
