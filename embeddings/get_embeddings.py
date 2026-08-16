#!/usr/bin/env python3
"""
embed_from_patches.py
======================
Extrae embeddings de modelos fundacionales a partir de los HDF5 de patches
ya generados por patch_extractor.py — NO vuelve a abrir las WSIs (.mrxs/.czi/...),
solo lee los arrays "patches" ya decodificados. Mucho más rápido que re-correr
la extracción completa con save_embeddings: true.

Reutiliza MODEL_REGISTRY, load_embedding_model() y get_embedding_transform()
de patch_extractor.py, así que la normalización es idéntica a la que usarías
si hubieras extraído embeddings en la primera corrida.

Estructura esperada de entrada (la que produce patch_extractor.py):
    patches_dir/
        <dataset_name>/
            slide_001.h5      # datasets: "patches" (N,3,H,W) uint8, "coords" (N,2)
            slide_002.h5
            ...
        <dataset_name_2>/
            ...

Estructura de salida:
    output_dir/
        <dataset_name>/
            <model_name>/
                slide_001.h5   # datasets: "embeddings" (N,D) float32, "coords" (N,2)
                ...

Uso:
    python embed_from_patches.py embed_config.yaml
    python embed_from_patches.py embed_config.yaml --dry-run
    python embed_from_patches.py embed_config.yaml --models uni2 virchow2
    python embed_from_patches.py embed_config.yaml --datasets bcnb
"""

import os
import sys
import time
import yaml
import logging
import argparse
import traceback
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── Reutilizamos el módulo de extracción de patches ─────────────────────────
# (debe estar en el mismo directorio, o ajustar sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extraction.patch_extractor_opt import MODEL_REGISTRY, load_embedding_model  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Dataset que lee patches directamente del HDF5 (sin PIL, sin reabrir WSI)
# ─────────────────────────────────────────────

class H5PatchDataset(Dataset):
    """
    Lee patches uint8 (C,H,W) ya guardados y los normaliza para el modelo
    dado. No usa PIL/Resize: los patches ya están al patch_size correcto
    (fueron guardados así por patch_extractor.py), solo se castea a float
    y se aplica Normalize con las stats del modelo.
    """

    def __init__(self, h5_path: str, mean: list, std: list):
        self.h5_path = h5_path
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self._file = None
        with h5py.File(h5_path, "r") as f:
            self.n_patches = f["patches"].shape[0]
            self.coords = f["coords"][:]  # cargar coords completas a RAM (liviano: N*2*int32)

    def _get_file(self):
        # cada worker de DataLoader abre su propio handle (h5py no es fork-safe)
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        f = self._get_file()
        patch = f["patches"][idx]  # (3, H, W) uint8
        tensor = torch.from_numpy(patch).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        x, y = self.coords[idx]
        return tensor, int(x), int(y)


# ─────────────────────────────────────────────
# Embedding writer (solo embeddings + coords, sin duplicar patches)
# ─────────────────────────────────────────────

class EmbeddingH5Writer:
    def __init__(self, path: str, embedding_dim: int, compression: str = "lzf"):
        self.path = path
        self.compression = compression if compression != "None" else None
        self._file = h5py.File(path, "w")
        self._embeds_ds = self._file.create_dataset(
            "embeddings",
            shape=(0, embedding_dim),
            maxshape=(None, embedding_dim),
            dtype=np.float32,
            compression=self.compression,
            chunks=(64, embedding_dim),
        )
        self._coords_ds = self._file.create_dataset(
            "coords", shape=(0, 2), maxshape=(None, 2), dtype=np.int32
        )

    def write_batch(self, embeddings: np.ndarray, coords: np.ndarray):
        n = embeddings.shape[0]
        cur = self._embeds_ds.shape[0]
        self._embeds_ds.resize(cur + n, axis=0)
        self._embeds_ds[cur:cur + n] = embeddings
        self._coords_ds.resize(cur + n, axis=0)
        self._coords_ds[cur:cur + n] = coords

    def add_metadata(self, **kwargs):
        for k, v in kwargs.items():
            self._file.attrs[k] = str(v)

    def close(self):
        self._file.close()


# ─────────────────────────────────────────────
# Procesar un slide para un modelo dado
# ─────────────────────────────────────────────

def embed_slide(
    patches_h5_path: Path,
    out_h5_path: Path,
    model: nn.Module,
    embedding_dim: int,
    mean: list,
    std: list,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    compression: str,
    dry_run: bool = False,
) -> dict:
    slide_name = patches_h5_path.stem
    stats = {"slide": slide_name, "n_patches": 0, "errors": 0, "time_s": 0.0}
    t0 = time.time()

    try:
        with h5py.File(patches_h5_path, "r") as f:
            n_patches = f["patches"].shape[0]
        stats["n_patches"] = n_patches

        if dry_run:
            stats["time_s"] = time.time() - t0
            return stats

        dataset = H5PatchDataset(str(patches_h5_path), mean=mean, std=std)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

        writer = EmbeddingH5Writer(str(out_h5_path), embedding_dim, compression)
        writer.add_metadata(
            slide=slide_name,
            source_patches=str(patches_h5_path),
            n_patches=n_patches,
        )

        with torch.no_grad():
            for batch_tensors, batch_x, batch_y in loader:
                gpu_tensors = batch_tensors.to(device, non_blocking=True)
                embeds = model(gpu_tensors)
                if isinstance(embeds, (list, tuple)):
                    embeds = embeds[0]
                embeds_np = embeds.cpu().float().numpy()
                coords_np = np.stack(
                    [batch_x.numpy(), batch_y.numpy()], axis=1
                ).astype(np.int32)
                writer.write_batch(embeds_np, coords_np)

        writer.close()

    except Exception as e:
        logger.error(f"[{slide_name}] FAILED: {e}")
        logger.debug(traceback.format_exc())
        stats["errors"] += 1

    stats["time_s"] = time.time() - t0
    return stats


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────

def run_pipeline(config_path: str, dry_run: bool = False,
                  models_override=None, datasets_override=None):
    cfg = load_config(config_path)
    setup_logging(cfg.get("log_level", "INFO"), cfg.get("log_file"))

    patches_dir = Path(cfg["patches_dir"])
    output_dir = Path(cfg["output_dir"])
    models_dir = cfg.get("models_dir", "/home/DIINF/vmieres/tesis/models")
    batch_size = cfg.get("batch_size", 128)
    skip_existing = cfg.get("skip_existing", True)
    compression = cfg.get("hdf5_compression", "lzf")

    comp_cfg = cfg.get("compute", {})
    gpu_ids = comp_cfg.get("gpu_ids", [0])
    num_workers = comp_cfg.get("num_workers", 4)

    models = models_override or cfg.get("models", list(MODEL_REGISTRY.keys()))
    datasets = datasets_override or cfg.get("datasets", None)
    if datasets is None:
        datasets = sorted([d.name for d in patches_dir.iterdir() if d.is_dir()])

    if torch.cuda.is_available() and gpu_ids:
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU found, falling back to CPU")

    logger.info("=" * 70)
    logger.info("  EMBEDDING EXTRACTOR (desde patches ya guardados)")
    logger.info(f"  patches_dir : {patches_dir}")
    logger.info(f"  output_dir  : {output_dir}")
    logger.info(f"  modelos     : {models}")
    logger.info(f"  datasets    : {datasets}")
    if dry_run:
        logger.info("  *** DRY RUN MODE ***")
    logger.info("=" * 70)

    all_stats = []
    total_t0 = time.time()

    # Orden: modelo afuera → cargar cada modelo (pesado) una sola vez
    for model_name in models:
        logger.info(f"\n{'─'*60}")
        logger.info(f"Modelo: {model_name}")

        if dry_run:
            model = None
            embedding_dim = None
        else:
            model = load_embedding_model(model_name, models_dir, device)
            if len(gpu_ids) > 1:
                model = nn.DataParallel(model, device_ids=gpu_ids)
            model.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224).to(device)
                out = model(dummy)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                embedding_dim = out.shape[-1]
            logger.info(f"  Embedding dim: {embedding_dim}")

        reg = MODEL_REGISTRY[model_name.lower().replace("-", "").replace("_", "")]
        mean, std = reg["mean"], reg["std"]

        for ds_name in datasets:
            ds_patches_dir = patches_dir / ds_name
            if not ds_patches_dir.exists():
                logger.warning(f"  [{ds_name}] no existe en {patches_dir}, se omite")
                continue

            slide_files = sorted(ds_patches_dir.glob("*.h5"))
            slide_files = [f for f in slide_files if not f.stem.endswith("_embeddings")]
            if not slide_files:
                logger.warning(f"  [{ds_name}] sin archivos .h5 de patches")
                continue

            ds_out_dir = output_dir / ds_name / model_name
            if not dry_run:
                ds_out_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"  [{ds_name}] {len(slide_files)} slide(s)")

            ds_stats = []
            with tqdm(total=len(slide_files), desc=f"  [{ds_name}/{model_name}]",
                      unit="slide", dynamic_ncols=True) as bar:
                for patches_h5 in slide_files:
                    out_h5 = ds_out_dir / patches_h5.name

                    if skip_existing and out_h5.exists() and not dry_run:
                        bar.update(1)
                        continue

                    stats = embed_slide(
                        patches_h5_path=patches_h5,
                        out_h5_path=out_h5,
                        model=model,
                        embedding_dim=embedding_dim,
                        mean=mean,
                        std=std,
                        device=device,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        compression=compression,
                        dry_run=dry_run,
                    )
                    stats["dataset"] = ds_name
                    stats["model"] = model_name
                    ds_stats.append(stats)
                    all_stats.append(stats)
                    bar.update(1)

            total_patches = sum(s["n_patches"] for s in ds_stats)
            total_err = sum(s["errors"] for s in ds_stats)
            logger.info(
                f"  [{ds_name}/{model_name}] Resumen: {total_patches:,} patches | "
                f"Errores: {total_err}"
            )

        # liberar GPU antes de cargar el siguiente modelo
        if not dry_run:
            del model
            torch.cuda.empty_cache()

    elapsed = time.time() - total_t0
    logger.info(f"\n{'='*70}")
    logger.info("  RESUMEN GLOBAL")
    logger.info(f"  Tiempo total: {elapsed/60:.1f} min")
    logger.info(f"  Output: {output_dir.resolve()}")
    logger.info("=" * 70)

    if not dry_run and all_stats:
        import csv
        stats_path = output_dir / "embedding_stats.csv"
        with open(stats_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(all_stats[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(all_stats)
        logger.info(f"  Stats guardadas en: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extrae embeddings a partir de patches HDF5 ya generados"
    )
    parser.add_argument("config", type=str, help="Ruta al YAML de configuración")
    parser.add_argument("--dry-run", action="store_true",
                         help="Solo cuenta slides/patches, no calcula embeddings")
    parser.add_argument("--models", nargs="+", default=None,
                         help="Override: lista de modelos a correr (ej. uni2 virchow2)")
    parser.add_argument("--datasets", nargs="+", default=None,
                         help="Override: lista de datasets a procesar")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    run_pipeline(
        args.config,
        dry_run=args.dry_run,
        models_override=args.models,
        datasets_override=args.datasets,
    )


if __name__ == "__main__":
    main()