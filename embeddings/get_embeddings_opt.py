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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Desactiva file locking de HDF5: en storage compartido/NFS los locks pueden
# quedar "pegados" tras un crash o por contención entre procesos, causando
# "unable to lock file, errno=11". Solo leemos los .h5 de patches (no hay
# escritura concurrente sobre el mismo archivo), así que es seguro desactivarlo.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

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


def _h5_open_read(path):
    """
    Abre un .h5 en modo lectura sin pedir locks de filesystem al kernel/NFS.

    En mounts NFSv4 con local_lock=none, el locking se delega al lock
    manager del servidor; bajo I/O concurrente sostenido (muchas aperturas
    seguidas, como en este pipeline) eso produce intermitentemente
    "Resource temporarily unavailable" (errno 11), aunque el archivo esté
    sano. Como solo LEEMOS estos archivos y nunca escribimos sobre ellos
    desde múltiples procesos a la vez, es seguro desactivar el locking.

    locking=False requiere h5py >= 3.5. Si la versión instalada es más
    vieja, cae de vuelta a la apertura normal.
    """
    try:
        return h5py.File(path, "r", locking=False)
    except TypeError:
        return h5py.File(path, "r")


# ─────────────────────────────────────────────
# Dataset que lee patches directamente del HDF5 (sin PIL, sin reabrir WSI)
# ─────────────────────────────────────────────

class H5ChunkDataset(Dataset):
    """
    Lee patches uint8 (C,H,W) directo del HDF5 en CHUNKS CONTIGUOS en vez de
    índice por índice. h5py es mucho más rápido leyendo rangos contiguos que
    haciendo fancy-indexing aleatorio uno por uno (que es lo que hacía el
    DataLoader por defecto al pedir __getitem__(idx) disperso).

    No normaliza aquí: devuelve uint8 crudo. La normalización (a float,
    /255, mean/std) se hace vectorizada en GPU sobre el batch completo,
    que es mucho más barato que hacerlo en CPU item por item dentro de
    cada worker.
    """

    def __init__(self, h5_path: str, chunk_size: int):
        self.h5_path = h5_path
        self.chunk_size = chunk_size
        self._file = None
        with _h5_open_read(h5_path) as f:
            self.n_patches = f["patches"].shape[0]
            self.coords = f["coords"][:]  # liviano: N*2*int32, va completo a RAM
        self.n_chunks = (self.n_patches + chunk_size - 1) // chunk_size

    def _get_file(self):
        if self._file is None:
            self._file = _h5_open_read(self.h5_path)
        return self._file

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, chunk_idx):
        f = self._get_file()
        start = chunk_idx * self.chunk_size
        end = min(start + self.chunk_size, self.n_patches)
        # lectura contigua de un slice completo: una sola llamada h5py
        patches = f["patches"][start:end]  # (n, 3, H, W) uint8
        coords = self.coords[start:end]    # (n, 2)
        return torch.from_numpy(patches), torch.from_numpy(coords)


def _identity_collate(batch):
    # batch_size=1 en el DataLoader porque cada "item" ya es un chunk
    # de N patches; evitamos que default_collate intente apilar arrays
    # de distinto tamaño (el último chunk puede ser más chico).
    return batch[0]


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

def _embed_slide_once(
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
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Un solo intento de procesar el slide. Puede lanzar excepciones."""
    slide_name = patches_h5_path.stem
    stats = {"slide": slide_name, "n_patches": 0, "errors": 0, "time_s": 0.0}
    t0 = time.time()

    with _h5_open_read(patches_h5_path) as f:
        n_patches = f["patches"].shape[0]
    stats["n_patches"] = n_patches

    if dry_run:
        stats["time_s"] = time.time() - t0
        return stats

    # Lee en chunks contiguos del tamaño de batch_size: una sola llamada
    # h5py por batch (lectura secuencial) en vez de N llamadas dispersas.
    dataset = H5ChunkDataset(str(patches_h5_path), chunk_size=batch_size)
    loader = DataLoader(
        dataset,
        batch_size=1,                 # cada "item" ya es un chunk de N patches
        collate_fn=_identity_collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)

    writer = EmbeddingH5Writer(str(out_h5_path), embedding_dim, compression)
    writer.add_metadata(
        slide=slide_name,
        source_patches=str(patches_h5_path),
        n_patches=n_patches,
    )

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if (use_amp and device.type == "cuda")
        else torch.autocast(device_type="cpu", enabled=False)
    )

    with torch.no_grad(), autocast_ctx:
        for patches_u8, coords in loader:
            # patches_u8: (n, 3, H, W) uint8 en CPU; coords: (n, 2)
            gpu_u8 = patches_u8.to(device, non_blocking=True)
            # normalización vectorizada en GPU sobre todo el batch a la vez
            gpu_tensors = gpu_u8.float().div_(255.0).sub_(mean_t).div_(std_t)

            embeds = model(gpu_tensors)
            if isinstance(embeds, (list, tuple)):
                embeds = embeds[0]

            embeds_np = embeds.float().cpu().numpy()
            coords_np = coords.numpy().astype(np.int32)
            writer.write_batch(embeds_np, coords_np)
            # liberar tensores GPU explícitamente para evitar fragmentación
            del gpu_u8, gpu_tensors, embeds

    writer.close()

    # Verificación de integridad: si lo escrito no coincide con lo esperado,
    # tratarlo como fallo (evita dejar .h5 incompletos marcados como "listos")
    with h5py.File(out_h5_path, "r") as f:
        n_written = f["embeddings"].shape[0]
    if n_written != n_patches:
        raise RuntimeError(
            f"Embeddings incompletos: escritos {n_written}/{n_patches}"
        )

    return stats


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
    max_retries: int = 3,
    retry_wait_s: float = 10.0,
    use_amp: bool = True,
    failed_log_path: Optional[Path] = None,
) -> dict:
    """
    Wrapper con reintentos para errores transitorios (I/O de red, OOM puntual,
    etc). Si todos los intentos fallan, borra el .h5 de salida parcial para
    que no quede bloqueando skip_existing en la próxima corrida, y registra
    el slide en failed_log_path para revisión manual posterior.
    """
    slide_name = patches_h5_path.stem
    t0 = time.time()
    last_err = None
    effective_num_workers = num_workers

    for attempt in range(1, max_retries + 1):
        try:
            stats = _embed_slide_once(
                patches_h5_path, out_h5_path, model, embedding_dim,
                mean, std, device, batch_size, effective_num_workers,
                compression, dry_run=dry_run, use_amp=use_amp,
            )
            torch.cuda.empty_cache()
            stats["time_s"] = time.time() - t0
            stats["errors"] = 0
            return stats
        except Exception as e:
            last_err = e
            logger.warning(
                f"[{slide_name}] intento {attempt}/{max_retries} fallo: {e}"
            )
            logger.debug(traceback.format_exc())
            torch.cuda.empty_cache()
            # Borrar el .h5 parcial/corrupto antes de reintentar
            if out_h5_path.exists():
                try:
                    out_h5_path.unlink()
                except OSError:
                    pass
            # Errores de lock/I-O (errno 11, "unable to lock file") no son
            # multiprocessing-safe: si un worker muere a mitad de la lectura
            # puede dejar el contexto CUDA del proceso principal corrupto.
            # A partir del siguiente intento forzamos num_workers=0 (lectura
            # en el proceso principal, sin subprocesos) para evitar ese riesgo.
            is_lock_error = (
                isinstance(e, OSError) and getattr(e, "errno", None) == 11
            ) or "unable to lock file" in str(e).lower()
            if is_lock_error and effective_num_workers > 0:
                logger.warning(
                    f"[{slide_name}] error de lock detectado — "
                    f"reintentando con num_workers=0"
                )
                effective_num_workers = 0
            if attempt < max_retries:
                time.sleep(retry_wait_s)

    logger.error(f"[{slide_name}] FAILED tras {max_retries} intentos: {last_err}")

    # Registrar en archivo separado para poder revisar/reprocesar después
    # sin tener que grepear todo el log principal
    if failed_log_path is not None:
        try:
            with open(failed_log_path, "a") as f:
                f.write(f"{slide_name}\t{patches_h5_path}\t{last_err}\n")
        except OSError:
            pass

    stats = {"slide": slide_name, "n_patches": 0, "errors": 1,
             "time_s": time.time() - t0}
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
    failed_log_path = output_dir / "failed_slides.tsv"
    models_dir = cfg.get("models_dir", "/home/DIINF/vmieres/tesis/models")
    batch_size = cfg.get("batch_size", 128)
    skip_existing = cfg.get("skip_existing", True)
    compression = cfg.get("hdf5_compression", "lzf")
    use_amp = cfg.get("use_amp", True)  # bfloat16 autocast: ~1.5-3x más rápido

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
                        use_amp=use_amp,
                        failed_log_path=failed_log_path,
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