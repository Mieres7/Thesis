#!/usr/bin/env python3
"""
embed_from_patches_opt.py
==========================
Version optimizada de embed_from_patches.py para extraer embeddings desde
patches HDF5 ya generados por patch_extractor_opt.py.

Optimizaciones aplicadas respecto a la version original:
1. Multi-GPU real: cada GPU corre un proceso independiente (mp.Process) que
   consume slides de una cola compartida. Se elimina nn.DataParallel.
2. torch.autocast (bfloat16/float16) realmente activado y propagado desde
   el YAML (use_amp), igual que en patch_extractor_opt.py.
3. Normalizacion (cast a float + Normalize) movida a GPU en lugar de hacerse
   por item dentro de cada worker de DataLoader (menos overhead de CPU).
4. batch_size y num_workers configurables y con defaults mas altos.
5. torch.inference_mode() en vez de torch.no_grad() (menor overhead).

Uso:
    python embed_from_patches_opt.py embed_config.yaml
    python embed_from_patches_opt.py embed_config.yaml --dry-run
    python embed_from_patches_opt.py embed_config.yaml --models uni2 virchow2
    python embed_from_patches_opt.py embed_config.yaml --datasets HISTAI_Breast
"""

import os
import sys
import time
import yaml
import logging
import argparse
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


IO_MAX_RETRIES = 3
IO_RETRY_DELAY = 2.0


class IOErrorRetryable(Exception):
    """Señala que una operación de E/S falló pero puede reintentarse."""
    pass


def retry_io(func, *args, max_retries=IO_MAX_RETRIES, delay=IO_RETRY_DELAY, **kwargs):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except (OSError, IOError) as e:
            last_err = e
            if attempt < max_retries:
                wait = delay * attempt
                logger.warning(f"  I/O error (attempt {attempt}/{max_retries}): {e}. "
                               f"Reintentando en {wait:.1f}s...")
                time.sleep(wait)
    raise last_err


def validate_h5_patches(path: str) -> bool:
    """Verifica que un archivo H5 de patches se pueda leer correctamente."""
    try:
        def _check():
            with h5py.File(path, "r") as f:
                n = f["patches"].shape[0]
                if n == 0:
                    return False
                _ = f["patches"][0]
                _ = f["coords"][0]
            return True
        return retry_io(_check, max_retries=2, delay=1.0)
    except (OSError, IOError, KeyError) as e:
        logger.warning(f"  H5 validation failed for {path}: {e}")
        return False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extraction.patch_extractor_opt import MODEL_REGISTRY, load_embedding_model  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Dataset: SOLO lee uint8 crudo, sin normalizar en CPU.
# La normalizacion se hace en GPU dentro de embed_slide().
# ─────────────────────────────────────────────
class H5PatchDatasetRaw(Dataset):
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self._file = None
        with h5py.File(h5_path, "r") as f:
            self.n_patches = f["patches"].shape[0]
            self.coords = f["coords"][:]

    def _open_file(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
        self._file = h5py.File(self.h5_path, "r")
        return self._file

    def _get_file(self):
        if self._file is None:
            self._open_file()
        return self._file

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        try:
            f = self._get_file()
            patch = f["patches"][idx]  # (3,H,W) uint8
            x, y = self.coords[idx]
            return torch.from_numpy(patch), int(x), int(y)
        except (OSError, IOError) as e:
            logger.warning(f"  I/O error reading patch {idx}: {e}. Re-opening file...")
            self._open_file()
            f = self._file
            patch = retry_io(lambda: f["patches"][idx])
            x, y = self.coords[idx]
            return torch.from_numpy(patch), int(x), int(y)


class EmbeddingH5Writer:
    def __init__(self, path: str, embedding_dim: int, compression: str = "lzf"):
        self.path = path
        self.compression = compression if compression != "None" else None
        self._file = h5py.File(path, "w")
        self._embeds_ds = self._file.create_dataset(
            "embeddings", shape=(0, embedding_dim), maxshape=(None, embedding_dim),
            dtype=np.float32, compression=self.compression, chunks=(64, embedding_dim),
        )
        self._coords_ds = self._file.create_dataset(
            "coords", shape=(0, 2), maxshape=(None, 2), dtype=np.int32
        )

    def write_batch(self, embeddings: np.ndarray, coords: np.ndarray):
        def _write():
            n = embeddings.shape[0]
            cur = self._embeds_ds.shape[0]
            self._embeds_ds.resize(cur + n, axis=0)
            self._embeds_ds[cur:cur + n] = embeddings
            self._coords_ds.resize(cur + n, axis=0)
            self._coords_ds[cur:cur + n] = coords
        retry_io(_write)

    def add_metadata(self, **kwargs):
        for k, v in kwargs.items():
            self._file.attrs[k] = str(v)

    def close(self):
        try:
            if self._file:
                self._file.close()
        except (OSError, IOError) as e:
            logger.warning(f"  Error closing writer {self.path}: {e}")

    def abort(self):
        """Cierra el archivo y elimina el archivo parcial."""
        try:
            if self._file:
                self._file.close()
        except Exception:
            pass
        try:
            out = Path(self.path)
            if out.exists():
                out.unlink()
                logger.info(f"  Deleted incomplete output: {self.path}")
        except OSError:
            pass


def embed_slide(
    patches_h5_path: Path,
    out_h5_path: Path,
    model: nn.Module,
    embedding_dim: int,
    mean_gpu: torch.Tensor,
    std_gpu: torch.Tensor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    compression: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    dry_run: bool = False,
) -> dict:
    slide_name = patches_h5_path.stem
    stats = {"slide": slide_name, "n_patches": 0, "errors": 0, "time_s": 0.0}
    t0 = time.time()

    if not validate_h5_patches(str(patches_h5_path)):
        logger.error(f"[{slide_name}] SKIP: patches H5 validation failed (corrupt or inaccessible)")
        stats["errors"] = 1
        stats["time_s"] = time.time() - t0
        return stats

    writer = None
    try:
        with h5py.File(patches_h5_path, "r") as f:
            n_patches = f["patches"].shape[0]
        stats["n_patches"] = n_patches

        if dry_run:
            stats["time_s"] = time.time() - t0
            return stats

        dataset = H5PatchDatasetRaw(str(patches_h5_path))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )

        writer = EmbeddingH5Writer(str(out_h5_path), embedding_dim, compression)
        writer.add_metadata(slide=slide_name, source_patches=str(patches_h5_path), n_patches=n_patches)

        with torch.inference_mode():
            for batch_raw, batch_x, batch_y in loader:
                gpu_raw = batch_raw.to(device, non_blocking=True)
                gpu_tensors = gpu_raw.float().div_(255.0)
                gpu_tensors = (gpu_tensors - mean_gpu) / std_gpu

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
                    embeds = model(gpu_tensors)
                if isinstance(embeds, (list, tuple)):
                    embeds = embeds[0]

                embeds_np = embeds.float().cpu().numpy()
                coords_np = np.stack([batch_x.numpy(), batch_y.numpy()], axis=1).astype(np.int32)
                writer.write_batch(embeds_np, coords_np)

        writer.close()

    except (OSError, IOError) as e:
        logger.error(f"[{slide_name}] I/O FAILED: {e}")
        logger.debug(traceback.format_exc())
        stats["errors"] += 1
        if writer is not None:
            writer.abort()
        writer = None
    except Exception as e:
        logger.error(f"[{slide_name}] FAILED: {e}")
        logger.debug(traceback.format_exc())
        stats["errors"] += 1
        if writer is not None:
            writer.abort()
        writer = None

    stats["time_s"] = time.time() - t0
    return stats


# ─────────────────────────────────────────────
# Worker por GPU: carga cada modelo UNA vez, consume slides de la cola.
# ─────────────────────────────────────────────
def gpu_worker(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    models_dir: str,
    batch_size: int,
    num_workers: int,
    compression: str,
    use_amp: bool,
    skip_existing: bool,
    log_level: str,
    log_file: Optional[str],
):
    setup_logging(log_level, log_file)
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    props = torch.cuda.get_device_properties(device)
    amp_dtype = torch.bfloat16 if props.major >= 8 else torch.float16

    loaded_model_name = None
    model = None
    embedding_dim = None
    mean_gpu = std_gpu = None
    failed_slides = []

    while True:
        item = task_queue.get()
        if item is None:
            break

        model_name, patches_h5, out_h5 = item

        if skip_existing and out_h5.exists():
            result_queue.put((model_name, {"slide": patches_h5.stem, "n_patches": 0, "errors": 0, "time_s": 0.0, "skipped": True}))
            continue

        if model_name != loaded_model_name:
            if model is not None:
                del model
                torch.cuda.empty_cache()
            logger.info(f"[GPU {gpu_id}] Cargando modelo: {model_name}")
            model = load_embedding_model(model_name, models_dir, device)
            model.eval()

            reg = MODEL_REGISTRY[model_name.lower().replace("-", "").replace("_", "")]
            mean_gpu = torch.tensor(reg["mean"], device=device).view(1, 3, 1, 1)
            std_gpu = torch.tensor(reg["std"], device=device).view(1, 3, 1, 1)

            with torch.inference_mode():
                dummy = torch.zeros(1, 3, 224, 224, device=device)
                out = model(dummy)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                embedding_dim = out.shape[-1]
            logger.info(f"[GPU {gpu_id}] {model_name} embedding_dim={embedding_dim}")
            loaded_model_name = model_name

        out_h5.parent.mkdir(parents=True, exist_ok=True)
        stats = embed_slide(
            patches_h5_path=patches_h5,
            out_h5_path=out_h5,
            model=model,
            embedding_dim=embedding_dim,
            mean_gpu=mean_gpu,
            std_gpu=std_gpu,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            compression=compression,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        if stats["errors"] > 0:
            failed_slides.append(patches_h5.name)
            if len(failed_slides) <= 3:
                logger.info(f"[GPU {gpu_id}] Reintentando {patches_h5.name} en 5s...")
                time.sleep(5)
                stats2 = embed_slide(
                    patches_h5_path=patches_h5,
                    out_h5_path=out_h5,
                    model=model,
                    embedding_dim=embedding_dim,
                    mean_gpu=mean_gpu,
                    std_gpu=std_gpu,
                    device=device,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    compression=compression,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                if stats2["errors"] == 0:
                    stats = stats2
                    failed_slides.pop()
                    logger.info(f"[GPU {gpu_id}] {patches_h5.name} OK en reintento")

        result_queue.put((model_name, stats))

    if failed_slides:
        logger.warning(f"[GPU {gpu_id}] {len(failed_slides)} slides fallaron definitivamente: "
                       f"{failed_slides[:10]}{'...' if len(failed_slides) > 10 else ''}")
    result_queue.put(None)  # sentinel: este worker terminó


def run_pipeline(config_path: str, dry_run: bool = False,
                  models_override=None, datasets_override=None):
    cfg = load_config(config_path)
    setup_logging(cfg.get("log_level", "INFO"), cfg.get("log_file"))

    patches_dir = Path(cfg["patches_dir"])
    output_dir = Path(cfg["output_dir"])
    models_dir = cfg.get("models_dir", "/home/DIINF/vmieres/tesis/models")
    batch_size = cfg.get("batch_size", 256)
    skip_existing = cfg.get("skip_existing", True)
    compression = cfg.get("hdf5_compression", "lzf")
    use_amp = cfg.get("use_amp", True)

    comp_cfg = cfg.get("compute", {})
    gpu_ids = comp_cfg.get("gpu_ids", [0])
    num_workers = comp_cfg.get("num_workers", 4)

    models = models_override or cfg.get("models", list(MODEL_REGISTRY.keys()))
    datasets = datasets_override or cfg.get("datasets", None)
    if datasets is None:
        datasets = sorted([d.name for d in patches_dir.iterdir() if d.is_dir()])

    if not torch.cuda.is_available() or not gpu_ids:
        logger.error("No hay GPU disponible. Este pipeline requiere CUDA.")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info(" EMBEDDING EXTRACTOR OPT (multi-GPU, AMP real)")
    logger.info(f" patches_dir : {patches_dir}")
    logger.info(f" output_dir  : {output_dir}")
    logger.info(f" modelos     : {models}")
    logger.info(f" datasets    : {datasets}")
    logger.info(f" gpu_ids     : {gpu_ids}  | num_workers/GPU: {num_workers} | batch_size: {batch_size} | use_amp: {use_amp}")
    if dry_run:
        logger.info(" *** DRY RUN MODE ***")
    logger.info("=" * 70)

    # Construir lista de tareas (model_name, patches_h5, out_h5)
    work_items = []
    for model_name in models:
        for ds_name in datasets:
            ds_patches_dir = patches_dir / ds_name
            if not ds_patches_dir.exists():
                logger.warning(f" [{ds_name}] no existe en {patches_dir}, se omite")
                continue
            slide_files = sorted(ds_patches_dir.glob("*.h5"))
            slide_files = [f for f in slide_files if not f.stem.endswith("_embeddings")]
            if not slide_files:
                logger.warning(f" [{ds_name}] sin archivos .h5 de patches")
                continue
            ds_out_dir = output_dir / ds_name / model_name
            for patches_h5 in slide_files:
                out_h5 = ds_out_dir / patches_h5.name
                work_items.append((model_name, patches_h5, out_h5))

    if not work_items:
        logger.error("No hay slides para procesar.")
        sys.exit(1)

    logger.info(f" Total de tareas (slide x modelo): {len(work_items)}")

    if dry_run:
        total_patches = 0
        unreadable = 0
        for _, patches_h5, _ in work_items:
            try:
                def _count():
                    with h5py.File(patches_h5, "r") as f:
                        return f["patches"].shape[0]
                total_patches += retry_io(_count, max_retries=2, delay=0.5)
            except (OSError, IOError) as e:
                unreadable += 1
                logger.warning(f"  [DRY-RUN] No se pudo leer {patches_h5.name}: {e}")
        logger.info(f" DRY-RUN: {len(work_items)} tareas, {total_patches:,} patches candidatos"
                    f"{f' ({unreadable} archivos ilegibles)' if unreadable else ''}")
        return

    mp.set_start_method("spawn", force=True)
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    for item in work_items:
        task_queue.put(item)
    for _ in gpu_ids:
        task_queue.put(None)

    log_level = cfg.get("log_level", "INFO")
    log_file = cfg.get("log_file")

    workers = []
    for gid in gpu_ids:
        p = mp.Process(
            target=gpu_worker,
            args=(gid, task_queue, result_queue, models_dir, batch_size,
                  num_workers, compression, use_amp, skip_existing,
                  log_level, log_file),
            daemon=False,  # NO daemon: cada worker abre su propio DataLoader
                            # con num_workers>0, que crea procesos hijos.
                            # Python prohibe que procesos daemon tengan hijos.
        )
        p.start()
        workers.append(p)

    all_stats = []
    done_workers = 0
    total_t0 = time.time()
    with tqdm(total=len(work_items), desc="Embeddings (multi-GPU)", unit="slide", dynamic_ncols=True) as bar:
        while done_workers < len(gpu_ids):
            item = result_queue.get()
            if item is None:
                done_workers += 1
                continue
            model_name, stats = item
            stats["model"] = model_name
            all_stats.append(stats)
            bar.update(1)

    for p in workers:
        p.join()

    elapsed = time.time() - total_t0
    output_dir.mkdir(parents=True, exist_ok=True)
    n_failed = sum(1 for s in all_stats if s.get("errors", 0) > 0)
    n_skipped = sum(1 for s in all_stats if s.get("skipped", False))
    logger.info(f"\n{'='*70}")
    logger.info(" RESUMEN GLOBAL")
    logger.info(f" Tiempo total: {elapsed/60:.1f} min")
    logger.info(f" Output: {output_dir.resolve()}")
    logger.info(f" Procesados: {len(all_stats) - n_failed - n_skipped} | "
                f"Saltados (ya existían): {n_skipped} | Fallidos: {n_failed}")
    if n_failed > 0:
        failed = [s["slide"] for s in all_stats if s.get("errors", 0) > 0]
        logger.warning(f" Slides fallidos ({len(failed)}): "
                       f"{failed[:20]}{'...' if len(failed) > 20 else ''}")
    logger.info("=" * 70)

    if all_stats:
        import csv
        stats_path = output_dir / "embedding_stats.csv"
        with open(stats_path, "w", newline="") as f:
            fieldnames = sorted(set(k for s in all_stats for k in s.keys()))
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(all_stats)
        logger.info(f" Stats guardadas en: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Extrae embeddings (version optimizada multi-GPU + AMP)")
    parser.add_argument("config", type=str, help="Ruta al YAML de configuracion")
    parser.add_argument("--dry-run", action="store_true", help="Solo cuenta slides/patches")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    run_pipeline(args.config, dry_run=args.dry_run,
                 models_override=args.models, datasets_override=args.datasets)


if __name__ == "__main__":
    main()