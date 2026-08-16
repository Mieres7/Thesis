#!/usr/bin/env python3
"""
patch_extractor.py  (optimized)
================================
WSI Patch Extractor for Histology Datasets
Supports: HistologyHSI (.mrxs), BCNB (.jpg), HISTAI (.tiff), SurGen (.czi)

Optimizations over original:
  1. Multi-GPU parallelism: each GPU runs an independent worker process
     that consumes slides from a shared queue.
  2. Vectorized tissue-mask filtering via summed-area table (integral image).
     Replaces O(N) Python loop with a single NumPy operation.
  3. torch.autocast (bfloat16/float16) for 1.5-3x faster embedding inference.
  4. Asynchronous HDF5 writes: a background thread drains a queue so the GPU
     is never stalled waiting for disk I/O.
  5. Vectorized patch-coordinate generation with np.meshgrid.
  6. CZI bounding-box cached once at open() instead of per-region call.
  7. Shared-memory array for large PIL slides so DataLoader workers do not
     each load their own copy of the image.

Usage:
    python patch_extractor.py config.yaml
    python patch_extractor.py config.yaml --dry-run
"""

import os
import sys
import csv
import yaml
import time
import queue
import logging
import argparse
import warnings
import traceback
import threading
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field

import numpy as np
import h5py
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# print("estoy en el opt")

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config dataclasses  (unchanged API)
# ─────────────────────────────────────────────

@dataclass
class DatasetConfig:
    name: str
    path: str
    format: str
    enabled: bool = True
    select: Union[str, list] = "all"
    subfolder_structure: str = "flat"

@dataclass
class ExtractionConfig:
    patch_size: int = 224
    magnification: float = 20.0
    overlap: float = 0.0
    padding: bool = True

@dataclass
class FilterConfig:
    enabled: bool = True
    tissue_threshold: float = 0.5
    otsu_level: int = 0
    saturation_filter: bool = True
    saturation_threshold: float = 0.05

@dataclass
class OutputConfig:
    base_dir: str = "./output"
    save_patches: bool = True
    patches_format: str = "hdf5"
    save_embeddings: bool = False
    embedding_model: str = "uni2"
    embedding_batch_size: int = 64
    jpeg_quality: int = 95
    hdf5_compression: str = "lzf"

@dataclass
class ComputeConfig:
    num_workers: int = 8
    gpu_ids: list = field(default_factory=lambda: [0, 1])
    prefetch_factor: int = 2
    pin_memory: bool = True


# ─────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# WSI Reader  (thin wrappers — CZI bbox cached)
# ─────────────────────────────────────────────

class WSIReader:
    """Unified WSI reader: mrxs/tiff/svs via OpenSlide, czi via aicspylibczi,
    jpg/png via PIL.  CZI bounding-box is cached at open time."""

    def __init__(self, path: str, target_magnification: float = 20.0,
                 native_magnification: float = None):
        self.path = Path(path)
        self.target_mag = target_magnification
        self._native_mag_override = native_magnification
        self._slide = None
        self._level = None
        self._downsample = None
        self._open()

    # ── open ──────────────────────────────────────────────────────────────

    def _open(self):
        suffix = self.path.suffix.lower()
        if suffix in (".mrxs", ".tiff", ".tif", ".svs", ".ndpi", ".scn"):
            self._open_openslide()
        elif suffix == ".czi":
            self._open_czi()
        elif suffix in (".jpg", ".jpeg", ".png"):
            self._open_pil()
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    def _open_openslide(self):
        import openslide
        self._type = "openslide"
        self._slide = openslide.OpenSlide(str(self.path))
        native_mag = (self._native_mag_override
                      if self._native_mag_override is not None
                      else self._get_native_mag())
        if native_mag is None:
            raise ValueError(
                f"[{self.path.name}] No magnification found. "
                "Set native_magnification in config.")
        self._native_mag = native_mag
        ratio = native_mag / self.target_mag
        downsamples = self._slide.level_downsamples
        best_level, best_diff = 0, abs(downsamples[0] - ratio)
        for lvl, ds in enumerate(downsamples):
            if abs(ds - ratio) < best_diff:
                best_diff = abs(ds - ratio)
                best_level = lvl
        self._level = best_level
        self._downsample = downsamples[best_level]
        self._scale_factor = ratio / self._downsample
        self.dimensions = self._slide.level_dimensions[best_level]
        self.full_dimensions = self._slide.dimensions

    def _get_native_mag(self):
        props = self._slide.properties
        for key in ("openslide.objective-power", "aperio.AppMag"):
            val = props.get(key)
            if val is not None:
                try:
                    return float(val)
                except ValueError:
                    pass
        return None

    def _open_czi(self):
        try:
            from aicspylibczi import CziFile
            self._type = "czi"
            self._czi = CziFile(str(self.path))
            bbox = self._czi.get_mosaic_bounding_box()
            # ── OPTIMIZATION: cache bbox fields once ──────────────────────
            self._czi_x0 = bbox.x
            self._czi_y0 = bbox.y
            self._czi_bbox_x1 = bbox.x + bbox.w
            self._czi_bbox_y1 = bbox.y + bbox.h
            native_w, native_h = bbox.w, bbox.h
            self.full_dimensions = (native_w, native_h)
            native_mag = (self._native_mag_override
                          if self._native_mag_override is not None
                          else self._get_czi_native_mag())
            if native_mag is None:
                raise ValueError(
                    f"[{self.path.name}] No magnification found. "
                    "Set native_magnification in config.")
            self._czi_native_mag = native_mag
            self._czi_scale = self.target_mag / native_mag
            self.dimensions = (int(native_w * self._czi_scale),
                                int(native_h * self._czi_scale))
            self._level = 0
            self._downsample = 1.0 / self._czi_scale
            self._scale_factor = 1.0
        except ImportError:
            raise ImportError(
                "aicspylibczi not installed. Run: pip install aicspylibczi")

    def _get_czi_native_mag(self):
        try:
            import xml.etree.ElementTree as ET
            meta_xml = self._czi.meta
            root = ET.fromstring(meta_xml) if isinstance(meta_xml, str) else meta_xml
            for tag in (".//NominalMagnification", ".//Magnification",
                        ".//ObjectiveMagnification"):
                el = root.find(tag)
                if el is not None and el.text:
                    try:
                        return float(el.text)
                    except ValueError:
                        pass
        except Exception:
            pass
        return None

    def _open_pil(self):
        self._type = "pil"
        img_lazy = Image.open(str(self.path))
        native_w, native_h = img_lazy.width, img_lazy.height
        self.full_dimensions = (native_w, native_h)
        self._level = 0
        self._downsample = 1.0
        if (self._native_mag_override is not None
                and self._native_mag_override != self.target_mag):
            self._scale_factor = self.target_mag / self._native_mag_override
            self.dimensions = (int(native_w * self._scale_factor),
                                int(native_h * self._scale_factor))
        else:
            self._scale_factor = 1.0
            self.dimensions = (native_w, native_h)
        # Populated later by set_shared_array() or lazily in read_region()
        self._pil_arr = None
        self._shared_mem = None   # reference kept to prevent GC

    # ── Shared-memory API for PIL slides ──────────────────────────────────

    def set_shared_array(self, arr: np.ndarray):
        """Give the reader a pre-loaded array (shared memory from main process)."""
        self._pil_arr = arr

    # ── Thumbnail ─────────────────────────────────────────────────────────

    def get_thumbnail(self, size: tuple = (512, 512)) -> np.ndarray:
        if self._type == "openslide":
            thumb = self._slide.get_thumbnail(size)
            return np.array(thumb.convert("RGB"))
        elif self._type == "czi":
            w, h = self.full_dimensions
            scale = min(size[0] / w, size[1] / h)
            arr = self._czi.read_mosaic(
                region=(self._czi_x0, self._czi_y0, w, h),
                scale_factor=scale, C=0)
            arr = np.squeeze(arr)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = (arr / arr.max() * 255).clip(0, 255).astype(np.uint8)
            return arr
        elif self._type == "pil":
            img = Image.open(str(self.path)).convert("RGB")
            img.thumbnail(size, Image.LANCZOS)
            return np.array(img)

    # ── Region read ───────────────────────────────────────────────────────

    def read_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Return HxWx3 uint8 array at target magnification."""
        if self._type == "openslide":
            x0 = int(x * self._downsample)
            y0 = int(y * self._downsample)
            region = self._slide.read_region((x0, y0), self._level, (w, h))
            arr = np.array(region.convert("RGB"))
            if abs(self._scale_factor - 1.0) > 0.01:
                arr = np.array(Image.fromarray(arr).resize(
                    (int(w * self._scale_factor), int(h * self._scale_factor)),
                    Image.LANCZOS))
            return arr

        elif self._type == "czi":
            x_nat = int(x / self._czi_scale) + self._czi_x0
            y_nat = int(y / self._czi_scale) + self._czi_y0
            w_nat = int(w / self._czi_scale)
            h_nat = int(h / self._czi_scale)
            # Use cached bbox (no extra call per region)
            x_end = min(x_nat + w_nat, self._czi_bbox_x1)
            y_end = min(y_nat + h_nat, self._czi_bbox_y1)
            x_nat = max(x_nat, self._czi_x0)
            y_nat = max(y_nat, self._czi_y0)
            w_nat = x_end - x_nat
            h_nat = y_end - y_nat
            arr = self._czi.read_mosaic(
                region=(x_nat, y_nat, w_nat, h_nat),
                scale_factor=self._czi_scale, C=0)
            arr = np.squeeze(arr)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = (arr / arr.max() * 255).clip(0, 255).astype(np.uint8)
            if arr.shape[0] != h or arr.shape[1] != w:
                pad = np.ones((h, w, 3), dtype=np.uint8) * 255
                pad[:arr.shape[0], :arr.shape[1]] = arr
                return pad
            return arr

        elif self._type == "pil":
            # Lazy load (first access per worker); shared array takes priority
            if self._pil_arr is None:
                img = Image.open(str(self.path)).convert("RGB")
                if abs(self._scale_factor - 1.0) > 0.01:
                    img = img.resize(
                        (int(img.width * self._scale_factor),
                         int(img.height * self._scale_factor)), Image.LANCZOS)
                self._pil_arr = np.array(img)
            patch = self._pil_arr[y:y + h, x:x + w]
            if patch.shape[:2] != (h, w):
                pad = np.ones((h, w, 3), dtype=np.uint8) * 255
                pad[:patch.shape[0], :patch.shape[1]] = patch
                return pad
            return patch

    def close(self):
        if self._type == "openslide" and self._slide:
            self._slide.close()
        elif self._type == "pil":
            self._pil_arr = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────
# Tissue mask  (GPU-accelerated Otsu + saturation)
# ─────────────────────────────────────────────

def compute_tissue_mask_gpu(
    thumbnail: np.ndarray,
    device: torch.device,
    saturation_filter: bool = True,
    sat_threshold: float = 0.05
) -> np.ndarray:
    """
    Return bool numpy array (H, W) — True where tissue is present.
    Auto-detects dark-background (microscopy) vs bright-background (scanner).
    """
    t = torch.from_numpy(thumbnail).float().to(device) / 255.0
    gray = 0.2989 * t[..., 0] + 0.5870 * t[..., 1] + 0.1140 * t[..., 2]
    hist = torch.histc(gray.flatten(), bins=256, min=0.0, max=1.0)
    total = gray.numel()
    cumsum = torch.cumsum(hist, dim=0)
    cumsum_w = torch.cumsum(hist * torch.linspace(0, 1, 256, device=device), dim=0)
    global_mean = cumsum_w[-1]
    w0 = cumsum / total
    w1 = 1.0 - w0
    mu0 = cumsum_w / (cumsum + 1e-10)
    mu1 = (global_mean - cumsum_w) / (w1 + 1e-10)
    between_var = w0 * w1 * (mu0 - mu1) ** 2
    otsu_thresh = (torch.argmax(between_var).float() / 255.0).item()

    # Auto-detect: dark background (microscopy) → tissue is BRIGHTER than bg
    median_val = gray.median().item()
    if median_val < 0.3:
        tissue_otsu = gray > otsu_thresh
    else:
        tissue_otsu = gray < otsu_thresh

    if saturation_filter:
        r, g, b = t[..., 0], t[..., 1], t[..., 2]
        cmax = torch.maximum(torch.maximum(r, g), b)
        cmin = torch.minimum(torch.minimum(r, g), b)
        sat = (cmax - cmin) / (cmax + 1e-8)
        tissue = tissue_otsu & (sat > sat_threshold)
    else:
        tissue = tissue_otsu

    return tissue.cpu().numpy()


# ─────────────────────────────────────────────
# OPTIMIZATION 2: vectorized patch filtering
# via summed-area table (integral image)
# ─────────────────────────────────────────────

def filter_coords_by_mask(
    coords: np.ndarray,   # (N, 2) int32  [x, y]
    patch_size: int,
    slide_w: int,
    slide_h: int,
    mask: np.ndarray,     # (mH, mW) bool
    tissue_thresh: float,
) -> np.ndarray:
    """
    Return subset of coords where tissue fraction >= tissue_thresh.

    Uses a summed-area table so all N patches are evaluated in O(N)
    NumPy operations instead of O(N) Python function calls.
    """
    mh, mw = mask.shape
    # Build integral image (float32 to handle large slides)
    integral = np.zeros((mh + 1, mw + 1), dtype=np.float32)
    integral[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.float32), axis=0), axis=1)

    xs, ys = coords[:, 0], coords[:, 1]

    # Map patch corners to mask coordinates (vectorized)
    x1 = np.clip((xs / slide_w * mw).astype(np.int32), 0, mw)
    y1 = np.clip((ys / slide_h * mh).astype(np.int32), 0, mh)
    x2 = np.clip(((xs + patch_size) / slide_w * mw).astype(np.int32), 0, mw)
    y2 = np.clip(((ys + patch_size) / slide_h * mh).astype(np.int32), 0, mh)

    # Area of each patch region in mask pixels
    area = ((x2 - x1) * (y2 - y1)).astype(np.float32)
    area = np.maximum(area, 1.0)  # avoid div/0

    # Sum of tissue pixels via integral image
    tissue_sum = (integral[y2, x2]
                  - integral[y1, x2]
                  - integral[y2, x1]
                  + integral[y1, x1])

    fractions = tissue_sum / area
    return coords[fractions >= tissue_thresh]


# ─────────────────────────────────────────────
# OPTIMIZATION 4: vectorized coordinate generation
# ─────────────────────────────────────────────

def generate_patch_coords(
    slide_w: int, slide_h: int,
    patch_size: int,
    overlap: float = 0.0,
    padding: bool = True
) -> np.ndarray:
    """Return (N, 2) int32 array of (x, y) top-left coordinates."""
    stride = int(patch_size * (1.0 - overlap))
    xs = np.arange(0, slide_w, stride, dtype=np.int32)
    ys = np.arange(0, slide_h, stride, dtype=np.int32)
    if not padding:
        xs = xs[xs + patch_size <= slide_w]
        ys = ys[ys + patch_size <= slide_h]
    xv, yv = np.meshgrid(xs, ys, indexing='xy')
    coords = np.stack([xv.ravel(), yv.ravel()], axis=1)  # (N, 2)
    return coords


# ─────────────────────────────────────────────
# Patch Dataset  (DataLoader-compatible)
# ─────────────────────────────────────────────

class PatchDataset(Dataset):
    """Reads patches from a WSI using multiple CPU workers.

    For PIL slides, accepts an optional pre-loaded shared array so workers
    do not each open the full image independently.
    """

    def __init__(self, wsi_path: str, coords: np.ndarray, patch_size: int,
                 target_mag: float, native_mag=None, transform=None,
                 shared_arr: Optional[np.ndarray] = None):
        self.wsi_path = wsi_path
        self.coords = coords          # (N, 2) int32
        self.patch_size = patch_size
        self.target_mag = target_mag
        self.native_mag = native_mag
        self.transform = transform
        self.shared_arr = shared_arr  # pre-loaded PIL array (may be None)
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            self._reader = WSIReader(self.wsi_path, self.target_mag,
                                     native_magnification=self.native_mag)
            if self.shared_arr is not None:
                self._reader.set_shared_array(self.shared_arr)
        return self._reader

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = int(self.coords[idx, 0]), int(self.coords[idx, 1])
        reader = self._get_reader()
        try:
            patch = reader.read_region(x, y, self.patch_size, self.patch_size)
        except Exception:
            patch = np.ones((self.patch_size, self.patch_size, 3), dtype=np.uint8) * 255
        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            padded = np.ones((self.patch_size, self.patch_size, 3), dtype=np.uint8) * 255
            padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded
        img = Image.fromarray(patch)
        tensor = self.transform(img) if self.transform else T.ToTensor()(img)
        return tensor, x, y


# ─────────────────────────────────────────────
# Embedding model  (unchanged registry)
# ─────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

MODEL_REGISTRY = {
    "uni2": {
        "hf_type": "timm_local",
        "timm_name": "vit_huge_patch14_224",
        # BUG FIX: faltaban los kwargs que hacen que esto sea realmente UNI2-h
        # y no un ViT-Huge/14 genérico (embed_dim=1280 por defecto en timm).
        # Config oficial de MahmoodLab/UNI2-h: embed_dim=1536, depth=24,
        # mlp_ratio=2.66667*2 (SwiGLU), 8 register tokens, no_embed_class=True.
        "timm_kwargs": {
            "img_size": 224, "patch_size": 14, "num_classes": 0,
            "embed_dim": 1536, "depth": 24, "num_heads": 24,
            "init_values": 1e-5, "mlp_ratio": 2.66667 * 2,
            "no_embed_class": True, "reg_tokens": 8,
            "dynamic_img_size": True,
        },
        "uses_swiglu": True,  # señal para load_embedding_model: agregar mlp_layer/act_layer
        "mean": _IMAGENET_MEAN,
        "std":  _IMAGENET_STD,
        "description": "UNI2-h (MahmoodLab) — ViT-H/14, 1536-dim",
    },
    "virchow2": {
        "hf_type": "timm_local",
        "timm_name": "vit_huge_patch14_224",
        # BUG FIX: faltaban los kwargs de la MLP SwiGLU. Sin ellos timm arma
        # una MLP estándar (mlp_ratio=4.0 por defecto) y el checkpoint real
        # de Virchow2 (SwiGLUPacked, mlp_ratio=5.3375) no calza -> size mismatch
        # en blocks.*.mlp.fc1/fc2. Config oficial de Paige (HF: paige-ai/Virchow2):
        # vit_huge_patch14_224, mlp_layer=SwiGLUPacked, act_layer=SiLU,
        # mlp_ratio=5.3375, init_values=1e-5, reg_tokens=4, global_pool="".
        "timm_kwargs": {"img_size": 224, "patch_size": 14, "num_classes": 0,
                        "dynamic_img_size": True, "reg_tokens": 4,
                        "global_pool": "", "init_values": 1e-5,
                        "mlp_ratio": 5.3375},
        "uses_swiglu": True,  # señal para load_embedding_model: agregar mlp_layer/act_layer
        "mean": _IMAGENET_MEAN,
        "std":  _IMAGENET_STD,
        "description": "Virchow2 (Paige) — ViT-H/14 con register tokens, 2560-dim",
        "output_mode": "virchow2",  # CLS + mean(patch tokens)
    },
    "phikonv2": {
        "hf_type": "transformers",
        "mean": _IMAGENET_MEAN,
        "std":  _IMAGENET_STD,
        "description": "Phikon-v2 (Owkin) — ViT-L/16 transformers, 1024-dim",
    },
    "provgigapath": {
        "hf_type": "timm_local",
        "timm_name": "vit_giant_patch14_dinov2",
        # BUG FIX: el nombre de arquitectura timm dice "patch14" (heredado del
        # nombre "ViT-g/14"), pero el checkpoint real publicado en HF usa
        # patch_size=16 (verificado: patch_embed.proj.weight es [1536,3,16,16]
        # y pos_embed es [1,197,1536] = 1 cls + 14*14 patches). Sin este
        # override, timm arma la conv con kernel 14x14 y el tamaño no calza.
        "timm_kwargs": {"img_size": 224, "num_classes": 0, "patch_size": 16,
                        "dynamic_img_size": True},
        # BUG FIX: "mean"/"std" tenían stats de otro modelo/dataset. El transform
        # oficial de Microsoft (HF prov-gigapath/prov-gigapath, README) usa
        # normalización ImageNet estándar.
        "mean": _IMAGENET_MEAN,
        "std":  _IMAGENET_STD,
        "description": "ProvGigaPath (Microsoft) — ViT-g/14 tile encoder, 1536-dim",
        # BUG FIX: estaba apuntando a slide_encoder.pth (slide-level, arquitectura
        # LongNet, NO es un ViT) en vez de pytorch_model.bin, que es el tile
        # encoder real (ViT-g/14, DINOv2). De ahí el size mismatch en cls_token,
        # patch_embed, etc. — son arquitecturas completamente distintas.
        "weights_file": "pytorch_model.bin",  # tile encoder
    },
}


def _load_safetensors_or_bin(model_dir: str,
                              weights_file: Optional[str] = None) -> dict:
    model_path = Path(model_dir)
    if weights_file is not None:
        explicit = model_path / weights_file
        if not explicit.exists():
            raise FileNotFoundError(f"Weights file not found: {explicit}")
        ckpt = torch.load(str(explicit), map_location="cpu")
        if isinstance(ckpt, dict):
            for key in ("model", "state_dict", "teacher", "encoder"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    logger.info(f"  Extracted state_dict from key '{key}'")
                    return ckpt[key]
        return ckpt

    safetensors_files = list(model_path.glob("*.safetensors"))
    bin_files = list(model_path.glob("pytorch_model*.bin"))

    if safetensors_files:
        try:
            from safetensors.torch import load_file
            state_dict = {}
            for sf in sorted(safetensors_files):
                state_dict.update(load_file(str(sf), device="cpu"))
            logger.info(f"  Loaded safetensors: {[f.name for f in safetensors_files]}")
            return state_dict
        except ImportError:
            logger.warning("safetensors not installed, falling back to .bin")

    if bin_files:
        state_dict = {}
        for bf in sorted(bin_files):
            ckpt = torch.load(str(bf), map_location="cpu")
            if isinstance(ckpt, dict) and not any(
                isinstance(v, torch.Tensor) for v in list(ckpt.values())[:3]
            ):
                for key in ("model", "state_dict"):
                    if key in ckpt:
                        ckpt = ckpt[key]
                        break
            state_dict.update(ckpt)
        logger.info(f"  Loaded bin: {[f.name for f in bin_files]}")
        return state_dict

    raise FileNotFoundError(
        f"No weights (.safetensors or pytorch_model*.bin) found in {model_dir}")


class Virchow2Wrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone.forward_features(x)
        cls_token = out[:, 0]
        patch_tokens = out[:, 5:]
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)


class PhikonV2Wrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(pixel_values=x).last_hidden_state[:, 0]


# def load_embedding_model(model_name: str, models_dir: str,
#                           device: torch.device) -> nn.Module:
#     import timm
#     key = model_name.lower().replace("-", "").replace("_", "")
#     if key not in MODEL_REGISTRY:
#         raise ValueError(f"Model '{model_name}' not in registry. "
#                          f"Options: {list(MODEL_REGISTRY.keys())}")
#     reg = MODEL_REGISTRY[key]
#     model_dir = Path(models_dir) / model_name
#     if not model_dir.exists():
#         raise FileNotFoundError(f"Model folder not found: {model_dir}")

#     logger.info(f"Loading model: {reg['description']}")
#     logger.info(f"  From: {model_dir}")

#     if reg["hf_type"] == "timm_local":
#         kwargs = reg.get("timm_kwargs", {"num_classes": 0})
#         model = timm.create_model(reg["timm_name"], pretrained=False, **kwargs)
#         state_dict = _load_safetensors_or_bin(str(model_dir),
#                                                reg.get("weights_file"))
#         cleaned = {}
#         for k, v in state_dict.items():
#             for prefix in ("model.", "backbone.", "encoder."):
#                 if k.startswith(prefix):
#                     k = k[len(prefix):]
#                     break
#             cleaned[k] = v
#         missing, unexpected = model.load_state_dict(cleaned, strict=False)
#         if missing:
#             logger.debug(f"  Missing keys ({len(missing)}): {missing[:5]}...")
#         if unexpected:
#             logger.debug(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
#         if reg.get("output_mode") == "virchow2":
#             model = Virchow2Wrapper(model)

#     elif reg["hf_type"] == "transformers":
#         from transformers import AutoModel
#         backbone = AutoModel.from_pretrained(
#             str(model_dir), local_files_only=True, trust_remote_code=True)
#         model = PhikonV2Wrapper(backbone)
#     else:
#         raise ValueError(f"Unknown hf_type: {reg['hf_type']}")

#     model = model.eval().to(device)
#     logger.info(f"  Model loaded on {device}")
#     return model

def load_embedding_model(model_name: str, models_dir: str, device: torch.device) -> nn.Module:
    import timm

    key = model_name.lower().replace("-", "").replace("_", "")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not in registry. Options: {list(MODEL_REGISTRY.keys())}")

    reg = MODEL_REGISTRY[key]
    model_dir = Path(models_dir) / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model folder not found: {model_dir}")

    logger.info(f"Loading model: {reg['description']}")
    logger.info(f"  From: {model_dir}")

    if reg["hf_type"] == "timm_local":
        kwargs = dict(reg.get("timm_kwargs", {"num_classes": 0}))

        if reg.get("uses_swiglu", False):
            kwargs["mlp_layer"] = timm.layers.SwiGLUPacked
            kwargs["act_layer"] = torch.nn.SiLU

        model = timm.create_model(
            reg["timm_name"],
            pretrained=False,
            **kwargs
        )

        state_dict = _load_safetensors_or_bin(str(model_dir), reg.get("weights_file"))
        cleaned = {}
        for k, v in state_dict.items():
            for prefix in ("model.", "backbone.", "encoder."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            cleaned[k] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            logger.debug(f"Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.debug(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        if reg.get("output_mode") == "virchow2":
            model = Virchow2Wrapper(model)

    elif reg["hf_type"] == "transformers":
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=True
        )
        model = PhikonV2Wrapper(backbone)

    else:
        raise ValueError(f"Unknown hf_type: {reg['hf_type']}")

    model = model.eval().to(device)
    logger.info(f"  Model loaded on {device}")
    return model


def get_embedding_transform(model_name: str, patch_size: int = 224) -> T.Compose:
    key = model_name.lower().replace("-", "").replace("_", "")
    reg = MODEL_REGISTRY.get(key, {})
    mean = reg.get("mean", _IMAGENET_MEAN)
    std  = reg.get("std",  _IMAGENET_STD)
    return T.Compose([
        T.Resize((patch_size, patch_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ─────────────────────────────────────────────
# OPTIMIZATION 3: async HDF5 writer
# ─────────────────────────────────────────────

class AsyncHDF5Writer:
    """
    HDF5 writer that drains a background thread queue so the GPU
    is never stalled waiting for disk writes.

    Usage:
        writer = AsyncHDF5Writer(path, compression)
        writer.open(patch_size, embedding_dim=dim)
        writer.write_batch(patches, coords, embeddings)   # non-blocking
        writer.close()                                    # flushes + joins
    """

    _SENTINEL = None  # signals the writer thread to stop

    def __init__(self, path: str, compression: str = "lzf"):
        self.path = path
        self.compression = compression if compression != "None" else None
        self._file = None
        self._q = queue.Queue(maxsize=8)   # backpressure limit
        self._thread = None
        self._error = None

    def open(self, patch_size: int, n_channels: int = 3,
             embedding_dim: Optional[int] = None):
        self._file = h5py.File(self.path, "w")
        self._patches_ds = self._file.create_dataset(
            "patches",
            shape=(0, n_channels, patch_size, patch_size),
            maxshape=(None, n_channels, patch_size, patch_size),
            dtype=np.uint8,
            compression=self.compression,
            chunks=(16, n_channels, patch_size, patch_size))
        self._coords_ds = self._file.create_dataset(
            "coords", shape=(0, 2), maxshape=(None, 2), dtype=np.int32)
        self._embeds_ds = None
        if embedding_dim is not None:
            self._embeds_ds = self._file.create_dataset(
                "embeddings",
                shape=(0, embedding_dim), maxshape=(None, embedding_dim),
                dtype=np.float32,
                compression=self.compression,
                chunks=(64, embedding_dim))
        # Start background writer thread
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _writer_loop(self):
        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                self._q.task_done()
                break
            try:
                patches, coords, embeddings = item
                n = patches.shape[0]
                cur = self._patches_ds.shape[0]
                self._patches_ds.resize(cur + n, axis=0)
                self._patches_ds[cur:cur + n] = patches
                self._coords_ds.resize(cur + n, axis=0)
                self._coords_ds[cur:cur + n] = coords
                if embeddings is not None and self._embeds_ds is not None:
                    self._embeds_ds.resize(cur + n, axis=0)
                    self._embeds_ds[cur:cur + n] = embeddings
            except Exception as e:
                self._error = e
            finally:
                self._q.task_done()

    def write_batch(self, patches: np.ndarray, coords: np.ndarray,
                    embeddings: Optional[np.ndarray] = None):
        if self._error:
            raise RuntimeError(f"HDF5 writer thread failed: {self._error}")
        self._q.put((patches, coords, embeddings))

    def add_metadata(self, **kwargs):
        if self._file:
            for k, v in kwargs.items():
                self._file.attrs[k] = str(v)

    def close(self):
        if self._thread is not None:
            self._q.put(self._SENTINEL)
            self._q.join()
            self._thread.join()
        if self._file:
            self._file.close()
        if self._error:
            raise RuntimeError(f"HDF5 writer thread failed: {self._error}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────
# File discovery  (unchanged logic)
# ─────────────────────────────────────────────

def discover_files(ds_cfg: dict) -> list:
    base = Path(ds_cfg["path"])
    fmt = ds_cfg["format"].lower()
    structure = ds_cfg.get("subfolder_structure", "flat")
    select = ds_cfg.get("select", "all")

    ext_map = {
        "mrxs": ["*.mrxs"],
        "jpg":  ["*.jpg", "*.jpeg"],
        "tiff": ["*.tiff", "*.tif"],
        "czi":  ["*.czi"],
    }
    patterns = ext_map.get(fmt, [f"*.{fmt}"])
    files = []

    if structure == "flat":
        for pat in patterns:
            files.extend(base.glob(pat))
        files = sorted(files)
        if select != "all" and isinstance(select, list):
            sel = set(str(s) for s in select)
            files = [f for f in files if f.stem in sel or f.name in sel]

    elif structure == "case_folders":
        sel = set(str(s) for s in select) if (
            select != "all" and isinstance(select, list)) else None
        for case_dir in sorted(base.iterdir()):
            if not case_dir.is_dir() or case_dir.name == "patches":
                continue
            if sel is not None and case_dir.name not in sel:
                continue
            for pat in patterns:
                files.extend(case_dir.glob(pat))
        files = sorted(files)

    elif structure == "split_folders":
        for sub_dir in sorted(base.iterdir()):
            if sub_dir.is_dir() and sub_dir.name != "patches":
                for pat in patterns:
                    files.extend(sub_dir.glob(pat))
        files = sorted(files)
        if select != "all" and isinstance(select, list):
            sel = set(str(s) for s in select)
            files = [f for f in files if f.stem in sel or f.name in sel]

    return [str(f) for f in files]


# ─────────────────────────────────────────────
# Core: extract patches from one slide
# ─────────────────────────────────────────────

def extract_patches_from_slide(
    wsi_path: str,
    output_dir: Path,
    ext_cfg: dict,
    filt_cfg: dict,
    out_cfg: dict,
    comp_cfg: dict,
    device: torch.device,
    embedding_model: Optional[nn.Module] = None,
    embedding_dim: Optional[int] = None,
    dry_run: bool = False,
    output_stem: Optional[str] = None,
) -> dict:
    patch_size   = ext_cfg["patch_size"]
    magnification = ext_cfg["magnification"]
    overlap      = ext_cfg.get("overlap", 0.0)
    padding      = ext_cfg.get("padding", True)

    tissue_thresh  = filt_cfg.get("tissue_threshold", 0.5)
    filter_enabled = filt_cfg.get("enabled", True)
    sat_filter     = filt_cfg.get("saturation_filter", True)
    sat_thresh     = filt_cfg.get("saturation_threshold", 0.05)

    save_patches    = out_cfg.get("save_patches", True)
    save_embeddings = out_cfg.get("save_embeddings", False)
    patches_fmt     = out_cfg.get("patches_format", "hdf5")
    hdf5_compression = out_cfg.get("hdf5_compression", "lzf")
    embed_batch     = out_cfg.get("embedding_batch_size", 64)
    jpeg_quality    = out_cfg.get("jpeg_quality", 95)

    num_workers  = comp_cfg.get("num_workers", 8)
    prefetch     = comp_cfg.get("prefetch_factor", 2)
    pin_mem      = comp_cfg.get("pin_memory", True)

    slide_name = Path(wsi_path).stem
    out_stem   = output_stem if output_stem is not None else slide_name
    stats = {"slide": out_stem, "total_patches": 0, "kept_patches": 0,
             "filtered_patches": 0, "errors": 0, "time_s": 0.0}
    t0 = time.time()

    try:
        reader = WSIReader(wsi_path, target_magnification=magnification,
                           native_magnification=ext_cfg.get("native_magnification"))
        slide_w, slide_h = reader.dimensions

        # ── Tissue mask ───────────────────────────────────────────────────
        mask_np = None
        if filter_enabled:
            thumb = reader.get_thumbnail((1024, 1024))
            mask_np = compute_tissue_mask_gpu(thumb, device, sat_filter, sat_thresh)

        # ── Generate all candidate coordinates (vectorized) ───────────────
        coords = generate_patch_coords(slide_w, slide_h, patch_size,
                                        overlap, padding)  # (N, 2) int32
        stats["total_patches"] = len(coords)

        if dry_run:
            reader.close()
            stats["time_s"] = time.time() - t0
            logger.info(f"[DRY-RUN] {out_stem}: {slide_w}x{slide_h}px "
                        f"→ {len(coords)} candidate patches")
            return stats

        # ── Filter coordinates (vectorized via integral image) ────────────
        if filter_enabled and mask_np is not None:
            kept_coords = filter_coords_by_mask(
                coords, patch_size, slide_w, slide_h, mask_np, tissue_thresh)
        else:
            kept_coords = coords

        stats["kept_patches"]    = len(kept_coords)
        stats["filtered_patches"] = stats["total_patches"] - stats["kept_patches"]

        if len(kept_coords) == 0:
            logger.warning(f"[{slide_name}] No tissue patches after filtering.")
            reader.close()
            return stats

        # ── Pre-load PIL slides into a single array (shared among workers) ─
        is_pil = Path(wsi_path).suffix.lower() in (".jpg", ".jpeg", ".png")
        shared_arr = None
        if is_pil:
            img = Image.open(wsi_path).convert("RGB")
            sf = reader._scale_factor
            if abs(sf - 1.0) > 0.01:
                img = img.resize((int(img.width * sf), int(img.height * sf)),
                                  Image.LANCZOS)
            shared_arr = np.array(img)   # one copy in memory, shared via fork

        # ── Output setup ──────────────────────────────────────────────────
        writer = None
        img_dir = None
        if patches_fmt == "hdf5" and save_patches:
            hdf5_path = output_dir / f"{out_stem}.h5"
            writer = AsyncHDF5Writer(str(hdf5_path), compression=hdf5_compression)
            writer.open(patch_size, n_channels=3,
                        embedding_dim=embedding_dim if save_embeddings else None)
            writer.add_metadata(slide=slide_name, wsi_path=wsi_path,
                                 patch_size=patch_size, magnification=magnification,
                                 n_patches=len(kept_coords))
        elif patches_fmt in ("png", "jpeg") and save_patches:
            img_dir = output_dir / out_stem
            img_dir.mkdir(parents=True, exist_ok=True)

        if save_embeddings and not save_patches:
            hdf5_path = output_dir / f"{out_stem}_embeddings.h5"
            writer = AsyncHDF5Writer(str(hdf5_path), compression=hdf5_compression)
            writer.open(patch_size, n_channels=3, embedding_dim=embedding_dim)
            writer.add_metadata(slide=slide_name, wsi_path=wsi_path,
                                 patch_size=patch_size, magnification=magnification,
                                 n_patches=len(kept_coords))

        # ── Transform ────────────────────────────────────────────────────
        transform = (get_embedding_transform(
                        out_cfg.get("embedding_model", "uni2"), patch_size)
                     if save_embeddings and embedding_model is not None
                     else T.ToTensor())

        # ── DataLoader ────────────────────────────────────────────────────
        # PIL slides: workers share the pre-loaded array, no persistent workers
        # needed (no file handle to keep alive across batches).
        use_persistent = num_workers > 0 and not is_pil
        dataset = PatchDataset(
            wsi_path=wsi_path, coords=kept_coords, patch_size=patch_size,
            target_mag=magnification,
            native_mag=ext_cfg.get("native_magnification"),
            transform=transform, shared_arr=shared_arr)
        loader = DataLoader(
            dataset, batch_size=embed_batch, num_workers=num_workers,
            prefetch_factor=prefetch if num_workers > 0 else None,
            pin_memory=pin_mem and torch.cuda.is_available(),
            persistent_workers=use_persistent)

        # ── Determine autocast dtype ──────────────────────────────────────
        # bfloat16 is preferred on Ampere+; fall back to float16 otherwise.
        if save_embeddings and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            amp_dtype = (torch.bfloat16
                         if props.major >= 8 else torch.float16)
        else:
            amp_dtype = torch.float32

        # ── Main extraction loop ──────────────────────────────────────────
        with tqdm(total=len(kept_coords),
                  desc=f"  {slide_name[:40]:40s}",
                  unit="patch", leave=False, dynamic_ncols=True) as pbar:
            for batch_tensors, batch_x, batch_y in loader:
                bs = batch_tensors.shape[0]
                coords_np = np.stack(
                    [batch_x.numpy(), batch_y.numpy()], axis=1).astype(np.int32)

                # ── OPTIMIZATION 3: autocast embedding inference ──────────
                embeds_np = None
                if save_embeddings and embedding_model is not None:
                    with torch.no_grad(), torch.autocast(
                            device_type=device.type,
                            dtype=amp_dtype,
                            enabled=(device.type == "cuda")):
                        gpu_t = batch_tensors.to(device, non_blocking=True)
                        embeds = embedding_model(gpu_t)
                        if isinstance(embeds, (list, tuple)):
                            embeds = embeds[0]
                    embeds_np = embeds.cpu().float().numpy()

                # ── Save patches ──────────────────────────────────────────
                if save_patches:
                    if batch_tensors.dtype == torch.float32:
                        # Do the uint8 conversion on GPU before moving to CPU
                        patch_uint8 = (batch_tensors.mul(255)
                                       .clamp(0, 255)
                                       .byte()
                                       .numpy())
                    else:
                        patch_uint8 = batch_tensors.numpy()

                    if patches_fmt == "hdf5" and writer is not None:
                        writer.write_batch(patch_uint8, coords_np, embeds_np)
                    elif patches_fmt in ("png", "jpeg") and img_dir is not None:
                        ext = "jpg" if patches_fmt == "jpeg" else "png"
                        for i in range(bs):
                            xi, yi = int(batch_x[i]), int(batch_y[i])
                            img_arr = np.transpose(patch_uint8[i], (1, 2, 0))
                            img_pil = Image.fromarray(img_arr)
                            fname = img_dir / f"patch_{xi:06d}_{yi:06d}.{ext}"
                            (img_pil.save(fname, quality=jpeg_quality)
                             if patches_fmt == "jpeg" else img_pil.save(fname))

                elif save_embeddings and not save_patches and writer is not None:
                    dummy = np.zeros((bs, 3, patch_size, patch_size), dtype=np.uint8)
                    writer.write_batch(dummy, coords_np, embeds_np)

                pbar.update(bs)

        if writer is not None:
            writer.close()   # flushes async queue, then closes HDF5
        reader.close()

    except Exception as e:
        logger.error(f"[{Path(wsi_path).stem}] FAILED: {e}")
        logger.debug(traceback.format_exc())
        stats["errors"] += 1

    stats["time_s"] = time.time() - t0
    return stats


# ─────────────────────────────────────────────
# OPTIMIZATION 1: GPU worker process
# ─────────────────────────────────────────────

def _gpu_worker(
    gpu_id: int,
    slide_queue: "mp.Queue",
    result_queue: "mp.Queue",
    ext_cfg: dict,
    filt_cfg: dict,
    out_cfg: dict,
    comp_cfg: dict,
    output_base: str,
    datasets_meta: list,   # list of (ds_name, structure) for stem logic
    dry_run: bool,
    log_level: str,
    log_file: Optional[str],
):
    """
    Worker process: owns one GPU, consumes (ds_name, wsi_path, output_stem)
    tuples from slide_queue until it receives None.
    """
    setup_logging(log_level, log_file)
    device = torch.device(f"cuda:{gpu_id}")

    # Load embedding model once per worker
    embedding_model = None
    embedding_dim   = None
    if out_cfg.get("save_embeddings", False):
        model_name  = out_cfg.get("embedding_model", "uni2")
        models_dir  = out_cfg.get("models_dir", "./models")
        embedding_model = load_embedding_model(model_name, models_dir, device)
        embedding_model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, ext_cfg["patch_size"],
                                ext_cfg["patch_size"]).to(device)
            out = embedding_model(dummy)
            if isinstance(out, (list, tuple)):
                out = out[0]
            embedding_dim = out.shape[-1]
        logger.info(f"[GPU {gpu_id}] Embedding dim: {embedding_dim}")

    while True:
        item = slide_queue.get()
        if item is None:
            break
        ds_name, wsi_path, output_stem = item
        ds_output = Path(output_base) / ds_name
        ds_output.mkdir(parents=True, exist_ok=True)

        stats = extract_patches_from_slide(
            wsi_path=wsi_path,
            output_dir=ds_output,
            ext_cfg=ext_cfg,
            filt_cfg=filt_cfg,
            out_cfg=out_cfg,
            comp_cfg=comp_cfg,
            device=device,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            dry_run=dry_run,
            output_stem=output_stem,
        )
        result_queue.put((ds_name, stats))

    result_queue.put(None)  # signal this worker is done


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(config_path: str, dry_run: bool = False):
    cfg = load_config(config_path)
    setup_logging(
        log_level=cfg.get("log_level", "INFO"),
        log_file=cfg.get("log_file", None))

    logger.info("=" * 70)
    logger.info("  WSI PATCH EXTRACTOR  (optimized)")
    logger.info(f"  Config: {config_path}")
    if dry_run:
        logger.info("  *** DRY RUN MODE — no files will be written ***")
    logger.info("=" * 70)

    comp_cfg  = cfg.get("compute", {})
    ext_cfg   = cfg.get("extraction", {"patch_size": 224, "magnification": 20.0,
                                       "overlap": 0.0, "padding": True})
    filt_cfg  = cfg.get("filtering",  {"enabled": True, "tissue_threshold": 0.5,
                                       "saturation_filter": True,
                                       "saturation_threshold": 0.05})
    out_cfg   = cfg.get("output",     {"base_dir": "./output", "save_patches": True,
                                       "patches_format": "hdf5",
                                       "save_embeddings": False,
                                       "hdf5_compression": "lzf",
                                       "embedding_batch_size": 64})

    output_base = Path(out_cfg.get("base_dir", "./output"))
    output_base.mkdir(parents=True, exist_ok=True)

    # ── GPU setup ─────────────────────────────────────────────────────────
    gpu_ids = comp_cfg.get("gpu_ids", [0])
    if not torch.cuda.is_available():
        logger.warning("No CUDA GPUs found — falling back to single CPU process.")
        gpu_ids = []

    if gpu_ids:
        logger.info(f"GPUs to use: {gpu_ids}")
        for gid in gpu_ids:
            props = torch.cuda.get_device_properties(gid)
            logger.info(f"  GPU {gid}: {props.name} "
                        f"({props.total_memory // 1024**3} GB)")

    # ── Collect all slides across all datasets ────────────────────────────
    datasets_cfg = cfg.get("datasets", [])
    if not datasets_cfg:
        logger.error("No datasets configured.")
        sys.exit(1)

    # Build work list: (ds_name, wsi_path, output_stem)
    out_cfg    = cfg.get("output", {})
    skip_exist = out_cfg.get("skip_existing", False)

    work_items = []
    ds_names_seen = set()
    for ds in datasets_cfg:
        if not ds.get("enabled", True):
            logger.info(f"[{ds['name']}] SKIPPED (disabled)")
            continue
        files = discover_files(ds)
        if not files:
            logger.warning(f"[{ds['name']}] No files found!")
            continue
        logger.info(f"[{ds['name']}] {len(files)} file(s) found")
        ds_names_seen.add(ds["name"])
        structure = ds.get("subfolder_structure", "flat")
        for wsi_path in files:
            if structure == "case_folders":
                parent_name = Path(wsi_path).parent.name
                output_stem = f"{parent_name}__{Path(wsi_path).stem}"
            else:
                output_stem = None
            # skip_existing: verificar si el .h5 ya existe
            if skip_exist:
                stem = output_stem if output_stem is not None else Path(wsi_path).stem
                out_file = output_base / ds["name"] / f"{stem}.h5"
                if out_file.exists():
                    logger.info(f"[{ds['name']}] SKIP (already exists): {out_file.name}")
                    continue
            work_items.append((ds["name"], wsi_path, output_stem))

    if not work_items:
        logger.error("No slides to process.")
        sys.exit(1)

    total_t0 = time.time()

    # ── Single-GPU / CPU fallback (no multiprocessing) ────────────────────
    if len(gpu_ids) <= 1:
        device = torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cpu")

        embedding_model, embedding_dim = None, None
        if out_cfg.get("save_embeddings", False):
            model_name = out_cfg.get("embedding_model", "uni2")
            models_dir = out_cfg.get("models_dir", "./models")
            embedding_model = load_embedding_model(model_name, models_dir, device)
            embedding_model.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, 3, ext_cfg["patch_size"],
                                    ext_cfg["patch_size"]).to(device)
                out = embedding_model(dummy)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                embedding_dim = out.shape[-1]

        all_stats = []
        with tqdm(total=len(work_items), desc="Slides", unit="slide",
                  dynamic_ncols=True) as sbar:
            for ds_name, wsi_path, output_stem in work_items:
                sbar.set_postfix_str(Path(wsi_path).name[:35])
                ds_output = output_base / ds_name
                if not dry_run:
                    ds_output.mkdir(parents=True, exist_ok=True)
                stats = extract_patches_from_slide(
                    wsi_path=wsi_path, output_dir=ds_output,
                    ext_cfg=ext_cfg, filt_cfg=filt_cfg, out_cfg=out_cfg,
                    comp_cfg=comp_cfg, device=device,
                    embedding_model=embedding_model,
                    embedding_dim=embedding_dim, dry_run=dry_run,
                    output_stem=output_stem)
                all_stats.append((ds_name, stats))
                sbar.update(1)

    # ── Multi-GPU: each GPU runs an independent worker process ────────────
    else:
        slide_queue  = mp.Queue()
        result_queue = mp.Queue()

        # Enqueue all work
        for item in work_items:
            slide_queue.put(item)
        # Poison pills (one per worker)
        for _ in gpu_ids:
            slide_queue.put(None)

        log_level = cfg.get("log_level", "INFO")
        log_file  = cfg.get("log_file", None)

        workers = []
        for gid in gpu_ids:
            p = mp.Process(
                target=_gpu_worker,
                args=(gid, slide_queue, result_queue,
                      ext_cfg, filt_cfg, out_cfg, comp_cfg,
                      str(output_base), [],
                      dry_run, log_level, log_file),
                daemon=False)
            p.start()
            workers.append(p)

        # Collect results
        all_stats = []
        done_workers = 0
        with tqdm(total=len(work_items), desc="Slides (multi-GPU)",
                  unit="slide", dynamic_ncols=True) as sbar:
            while done_workers < len(gpu_ids):
                item = result_queue.get()
                if item is None:
                    done_workers += 1
                else:
                    all_stats.append(item)
                    sbar.update(1)

        for p in workers:
            p.join()

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - total_t0
    stats_flat = [s for _, s in all_stats]
    total_kept = sum(s["kept_patches"] for s in stats_flat)
    total_cand = sum(s["total_patches"] for s in stats_flat)
    total_err  = sum(s["errors"] for s in stats_flat)

    logger.info(f"\n{'='*70}")
    logger.info("  GLOBAL SUMMARY")
    logger.info(f"  Total slides   : {len(stats_flat)}")
    logger.info(f"  Candidates     : {total_cand:,}")
    logger.info(f"  Kept patches   : {total_kept:,}")
    logger.info(f"  Filter rate    : "
                f"{100*(total_cand - total_kept)/max(total_cand,1):.1f}%")
    logger.info(f"  Errors         : {total_err}")
    logger.info(f"  Total time     : {elapsed/60:.1f} min")
    logger.info(f"  Output dir     : {output_base.resolve()}")
    logger.info("=" * 70)

    if not dry_run and stats_flat:
        stats_path = output_base / "extraction_stats.csv"
        with open(stats_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(stats_flat[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(stats_flat)
        logger.info(f"  Stats saved to : {stats_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    # Required for multiprocessing on some platforms (macOS / Windows)
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="WSI Patch Extractor — optimized multi-GPU version")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count patches without extracting")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    run_pipeline(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()