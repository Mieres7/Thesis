#!/usr/bin/env python3
"""
patch_extractor.py
==================
WSI Patch Extractor for Histology Datasets
Supports: HistologyHSI (.mrxs), BCNB (.jpg), HISTAI (.tiff), SurGen (.czi)

Usage:
    python patch_extractor.py config.yaml
    python patch_extractor.py config.yaml --dry-run
"""

import os
import sys
import yaml
import time
import logging
import argparse
import warnings
import traceback
from pathlib import Path
from typing import Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import h5py
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # WSIs como JPG pueden ser >178MP, deshabilitar límite Pillow
from tqdm import tqdm

# Suppress minor warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─────────────────────────────────────────────
# Logging setup
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
# Config dataclass
# ─────────────────────────────────────────────

@dataclass
class DatasetConfig:
    name: str
    path: str
    format: str                          # mrxs | jpg | tiff | czi
    enabled: bool = True
    select: Union[str, list] = "all"     # "all" or list of filenames/stems
    subfolder_structure: str = "flat"    # "flat" | "case_folders" | "split_folders"

@dataclass
class ExtractionConfig:
    patch_size: int = 224
    magnification: float = 20.0
    overlap: float = 0.0                 # 0.0 = no overlap, 0.5 = 50%
    padding: bool = True

@dataclass
class FilterConfig:
    enabled: bool = True
    tissue_threshold: float = 0.5        # min fraction of tissue per patch
    otsu_level: int = 0                  # WSI level for mask generation (0=full, 1=2x down, etc.)
    saturation_filter: bool = True       # additional HSV saturation filter
    saturation_threshold: float = 0.05

@dataclass
class OutputConfig:
    base_dir: str = "./output"
    save_patches: bool = True
    patches_format: str = "hdf5"         # hdf5 | png | jpeg
    save_embeddings: bool = False
    embedding_model: str = "hf_hub:MahmoodLab/uni"  # timm-compatible string or HF model id
    embedding_batch_size: int = 64
    jpeg_quality: int = 95
    hdf5_compression: str = "lzf"        # lzf (fast) | gzip | None

@dataclass
class ComputeConfig:
    num_workers: int = 8                 # CPU workers for I/O
    gpu_ids: list = field(default_factory=lambda: [0, 1])
    prefetch_factor: int = 2
    pin_memory: bool = True


# ─────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─────────────────────────────────────────────
# WSI readers (thin wrappers)
# ─────────────────────────────────────────────

class WSIReader:
    """Unified WSI reader supporting mrxs, tiff, czi via OpenSlide / czifile."""

    def __init__(self, path: str, target_magnification: float = 20.0, native_magnification: float = None):
        self.path = Path(path)
        self.target_mag = target_magnification
        self._native_mag_override = native_magnification
        self._slide = None
        self._level = None
        self._downsample = None
        self._open()

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
        # Determine best level for target magnification
        # BUG FIX: override tiene prioridad; si no hay override, leer del slide una sola vez
        if self._native_mag_override is not None:
            native_mag = self._native_mag_override
        else:
            native_mag = self._get_native_mag()
            if native_mag is None:
                raise ValueError(
                    f"[{self.path.name}] No se pudo leer magnificación y no se "
                    f"especificó native_magnification en el config. "
                    f"Agrégala explícitamente para este dataset."
                )
        self._native_mag = native_mag
        ratio = native_mag / self.target_mag
        # Find closest level
        downsamples = self._slide.level_downsamples
        best_level = 0
        best_diff = abs(downsamples[0] - ratio)
        for lvl, ds in enumerate(downsamples):
            if abs(ds - ratio) < best_diff:
                best_diff = abs(ds - ratio)
                best_level = lvl
        self._level = best_level
        self._downsample = downsamples[best_level]
        self._scale_factor = ratio / self._downsample  # fine-tune rescale if needed
        self.dimensions = self._slide.level_dimensions[best_level]
        self.full_dimensions = self._slide.dimensions  # level 0

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

            # Dimensiones nativas del CZI (resolución completa del microscopio)
            bbox = self._czi.get_mosaic_bounding_box()
            self._czi_x0 = bbox.x
            self._czi_y0 = bbox.y
            native_w = bbox.w
            native_h = bbox.h
            self.full_dimensions = (native_w, native_h)

            # Intentar leer magnificación nativa desde metadatos
            # BUG FIX: override tiene prioridad; si no hay override, leer del CZI una sola vez
            if self._native_mag_override is not None:
                native_mag = self._native_mag_override
            else:
                native_mag = self._get_czi_native_mag()
                if native_mag is None:
                    raise ValueError(
                        f"[{self.path.name}] No se pudo leer magnificación y no se "
                        f"especificó native_magnification en el config."
                    )
            self._czi_native_mag = native_mag

            # Factor de downscale para alcanzar magnificación objetivo
            # e.g. nativa=40x, objetivo=20x → scale_factor=0.5
            self._czi_scale = self.target_mag / native_mag

            # Dimensiones a la magnificación objetivo
            self.dimensions = (
                int(native_w * self._czi_scale),
                int(native_h * self._czi_scale)
            )
            self._level = 0
            self._downsample = 1.0 / self._czi_scale
            self._scale_factor = 1.0

        except ImportError:
            raise ImportError(
                "aicspylibczi no instalado. Ejecuta: pip install aicspylibczi\n"
                "No uses czifile — carga el archivo completo en RAM (inviable para WSI)."
            )

    def _get_czi_native_mag(self):
        """Intenta leer la magnificación objetiva desde los metadatos XML del CZI."""
        try:
            import xml.etree.ElementTree as ET
            meta_xml = self._czi.meta
            if isinstance(meta_xml, str):
                root = ET.fromstring(meta_xml)
            else:
                root = meta_xml
            # Ruta típica en metadatos Zeiss CZI
            for tag in [
                ".//NominalMagnification",
                ".//Magnification",
                ".//ObjectiveMagnification",
            ]:
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
        # Usar lazy load: solo leer cabecera para dimensiones, no descomprimir
        img_lazy = Image.open(str(self.path))
        native_w, native_h = img_lazy.width, img_lazy.height
        self.full_dimensions = (native_w, native_h)
        self._level = 0
        self._downsample = 1.0
        # BUG FIX: si se conoce la magnificación nativa, escalar dimensiones al target
        if self._native_mag_override is not None and self._native_mag_override != self.target_mag:
            self._scale_factor = self.target_mag / self._native_mag_override
            self.dimensions = (
                int(native_w * self._scale_factor),
                int(native_h * self._scale_factor)
            )
        else:
            self._scale_factor = 1.0
            self.dimensions = (native_w, native_h)
        self._pil_img = None  # cargado bajo demanda en read_region(), liberado en close()

    def get_thumbnail(self, size: tuple = (512, 512)) -> np.ndarray:
        """Return low-res thumbnail as numpy RGB array."""
        if self._type == "openslide":
            thumb = self._slide.get_thumbnail(size)
            return np.array(thumb.convert("RGB"))
        elif self._type == "czi":
            # Leer región completa a baja resolución para thumbnail
            # aicspylibczi no tiene thumbnail nativo, leemos submuestra
            w, h = self.full_dimensions
            # Calcular escala para que quepa en size
            scale = min(size[0] / w, size[1] / h)
            # read_mosaic v3.x retorna ndarray (1, H, W, 3) directamente
            arr = self._czi.read_mosaic(
                region=(self._czi_x0, self._czi_y0, w, h),
                scale_factor=scale,
                C=0
            )
            arr = np.squeeze(arr)          # (H, W, 3) o (H, W)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = (arr / arr.max() * 255).clip(0, 255).astype(np.uint8)
            return arr
        elif self._type == "pil":
            img = Image.open(str(self.path)).convert("RGB")
            img.thumbnail(size, Image.LANCZOS)
            return np.array(img)

    def read_region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Read a region at the target magnification level.
        x, y are coordinates at the target level (not level 0).
        Returns HxWx3 uint8 numpy array.
        """
        if self._type == "openslide":
            # OpenSlide read_region takes level-0 coordinates
            x0 = int(x * self._downsample)
            y0 = int(y * self._downsample)
            region = self._slide.read_region((x0, y0), self._level, (w, h))
            arr = np.array(region.convert("RGB"))
            # Rescale if needed
            if abs(self._scale_factor - 1.0) > 0.01:
                new_w = int(w * self._scale_factor)
                new_h = int(h * self._scale_factor)
                arr = np.array(Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS))
            return arr
        elif self._type == "czi":
            # Convertir coordenadas del nivel objetivo a coordenadas nativas
            x_native = int(x / self._czi_scale) + self._czi_x0
            y_native = int(y / self._czi_scale) + self._czi_y0
            w_native = int(w / self._czi_scale)
            h_native = int(h / self._czi_scale)

            # --- FIX: clip al bounding box del CZI ---  Modificacion para ver si funciona
            bbox = self._czi.get_mosaic_bounding_box()
            x_end = min(x_native + w_native, bbox.x + bbox.w)
            y_end = min(y_native + h_native, bbox.y + bbox.h)
            x_native = max(x_native, bbox.x)
            y_native = max(y_native, bbox.y)
            w_native = x_end - x_native
            h_native = y_end - y_native
            # -----------------------------------------

            # read_mosaic v3.x: retorna ndarray (1, H, W, 3) directamente
            arr = self._czi.read_mosaic(
                region=(x_native, y_native, w_native, h_native),
                scale_factor=self._czi_scale,
                C=0
            )
            arr = np.squeeze(arr)          # (H, W, 3) o (H, W)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = (arr / arr.max() * 255).clip(0, 255).astype(np.uint8)

            # Pad si el parche es más pequeño (bordes del slide)
            if arr.shape[0] != h or arr.shape[1] != w:
                pad = np.ones((h, w, 3), dtype=np.uint8) * 255
                pad[:arr.shape[0], :arr.shape[1]] = arr
                return pad
            return arr
        elif self._type == "pil":
            # Carga completa bajo demanda (primera vez por worker).
            # La imagen se libera en close() al terminar cada slide.
            if self._pil_img is None:
                img_full = Image.open(str(self.path)).convert("RGB")
                # BUG FIX: escalar la imagen si native_mag != target_mag
                if abs(self._scale_factor - 1.0) > 0.01:
                    new_w = int(img_full.width * self._scale_factor)
                    new_h = int(img_full.height * self._scale_factor)
                    img_full = img_full.resize((new_w, new_h), Image.LANCZOS)
                self._pil_img = np.array(img_full)
            patch = self._pil_img[y:y+h, x:x+w]
            if patch.shape[:2] != (h, w):
                pad = np.ones((h, w, 3), dtype=np.uint8) * 255
                pad[:patch.shape[0], :patch.shape[1]] = patch
                return pad
            return patch

    def close(self):
        if self._type == "openslide" and self._slide:
            self._slide.close()
        elif self._type == "pil" and self._pil_img is not None:
            # Liberar la imagen de RAM explícitamente para que el GC la recoja
            # antes de pasar a la siguiente slide.
            del self._pil_img
            self._pil_img = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────
# Tissue mask (GPU-accelerated Otsu + saturation)
# ─────────────────────────────────────────────

def compute_tissue_mask_gpu(
    thumbnail: np.ndarray,
    device: torch.device,
    saturation_filter: bool = True,
    sat_threshold: float = 0.05
) -> torch.Tensor:
    """
    Compute binary tissue mask on GPU.
    Auto-detects dark-background (microscopy) vs bright-background (scanner).
    Returns bool tensor of shape (H, W) on CPU.
    """
    # thumbnail: HxWx3 uint8
    t = torch.from_numpy(thumbnail).float().to(device) / 255.0  # (H, W, 3)

    # ── Grayscale + Otsu ──────────────────────────────────────────────────
    gray = 0.2989 * t[..., 0] + 0.5870 * t[..., 1] + 0.1140 * t[..., 2]
    # Compute Otsu threshold on GPU via histogram
    hist = torch.histc(gray.flatten(), bins=256, min=0.0, max=1.0)
    total = gray.numel()
    cumsum = torch.cumsum(hist, dim=0)
    cumsum_weighted = torch.cumsum(
        hist * torch.linspace(0, 1, 256, device=device), dim=0
    )
    global_mean = cumsum_weighted[-1]
    w0 = cumsum / total
    w1 = 1.0 - w0
    mu0 = cumsum_weighted / (cumsum + 1e-10)
    mu1 = (global_mean - cumsum_weighted) / (w1 + 1e-10)
    between_var = w0 * w1 * (mu0 - mu1) ** 2
    otsu_thresh = (torch.argmax(between_var).float() / 255.0).item()

    # Auto-detect: if median intensity is low → dark background (microscopy)
    # → tissue is BRIGHTER than background. Otherwise → bright background (scanner)
    median_val = gray.median().item()
    dark_background = median_val < 0.3
    if dark_background:
        tissue_mask_otsu = gray > otsu_thresh
    else:
        tissue_mask_otsu = gray < otsu_thresh

    # ── Saturation filter (HSV) ───────────────────────────────────────────
    if saturation_filter:
        r, g, b = t[..., 0], t[..., 1], t[..., 2]
        cmax = torch.maximum(torch.maximum(r, g), b)
        cmin = torch.minimum(torch.minimum(r, g), b)
        sat = (cmax - cmin) / (cmax + 1e-8)
        sat_mask = sat > sat_threshold
        tissue_mask = tissue_mask_otsu & sat_mask
    else:
        tissue_mask = tissue_mask_otsu

    return tissue_mask.cpu()


def map_patch_to_mask(
    x: int, y: int, pw: int, ph: int,
    slide_w: int, slide_h: int,
    mask: np.ndarray
) -> float:
    """Compute fraction of patch area covered by tissue mask."""
    mh, mw = mask.shape
    # Map patch coordinates to mask coordinates
    x1 = int(x / slide_w * mw)
    y1 = int(y / slide_h * mh)
    x2 = int((x + pw) / slide_w * mw)
    y2 = int((y + ph) / slide_h * mh)
    x1, x2 = max(0, x1), min(mw, x2)
    y1, y2 = max(0, y1), min(mh, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = mask[y1:y2, x1:x2]
    return float(region.sum()) / region.size


# ─────────────────────────────────────────────
# Patch coordinate generator
# ─────────────────────────────────────────────

def generate_patch_coords(
    slide_w: int, slide_h: int,
    patch_size: int,
    overlap: float = 0.0,
    padding: bool = True
) -> list:
    """Generate (x, y) top-left coordinates for all patches."""
    stride = int(patch_size * (1.0 - overlap))
    coords = []
    y = 0
    while y < slide_h:
        x = 0
        while x < slide_w:
            if padding:
                coords.append((x, y))
            else:
                # Only include if full patch fits
                if x + patch_size <= slide_w and y + patch_size <= slide_h:
                    coords.append((x, y))
            x += stride
        y += stride
    return coords


# ─────────────────────────────────────────────
# Patch dataset for DataLoader
# ─────────────────────────────────────────────

class PatchDataset(Dataset):
    """Reads patches from a WSI using multiple CPU workers."""

    def __init__(
        self,
        wsi_path: str,
        coords: list,
        patch_size: int,
        target_mag: float,
        native_mag=None,
        transform=None
    ):
        self.wsi_path = wsi_path
        self.coords = coords
        self.patch_size = patch_size
        self.target_mag = target_mag
        self.native_mag = native_mag
        self.transform = transform
        # Each worker opens its own slide handle
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            self._reader = WSIReader(
                self.wsi_path,
                self.target_mag,
                native_magnification=self.native_mag)
        return self._reader

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        reader = self._get_reader()
        patch = reader.read_region(x, y, self.patch_size, self.patch_size)
        # Pad if smaller than expected (border patches)
        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            padded = np.ones((self.patch_size, self.patch_size, 3), dtype=np.uint8) * 255
            padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded
        img = Image.fromarray(patch)
        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = T.ToTensor()(img)
        return tensor, x, y


# ─────────────────────────────────────────────
# Embedding model loader (local HuggingFace folders)
# ─────────────────────────────────────────────

# Normalización estándar ImageNet (usada por todos estos modelos)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Registro de modelos soportados:
#   key       → nombre corto usado en config.yaml
#   timm_name → arquitectura timm exacta (para modelos basados en timm)
#   hf_type   → "timm_local" | "transformers"
#   mean/std  → normalización específica del modelo si difiere de ImageNet

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
        "description": "Virchow2 (Paige) — ViT-H/14 con register tokens, 1024-dim",
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


def _load_safetensors_or_bin(model_dir: str, weights_file: Optional[str] = None) -> dict:
    """
    Load weights from a specific file, .safetensors, or pytorch_model.bin.

    Args:
        model_dir    : carpeta del modelo
        weights_file : nombre de archivo explícito (e.g. "slide_encoder.pth").
                       Si es None, busca automáticamente safetensors o .bin.
    """
    model_path = Path(model_dir)

    # ── Archivo explícito (e.g. slide_encoder.pth para GigaPath) ──────────
    if weights_file is not None:
        explicit = model_path / weights_file
        if not explicit.exists():
            raise FileNotFoundError(f"Archivo de pesos no encontrado: {explicit}")
        ckpt = torch.load(str(explicit), map_location="cpu")
        # Los .pth pueden ser state_dict directos o dicts con clave "model"/"state_dict"
        if isinstance(ckpt, dict):
            for key in ("model", "state_dict", "teacher", "encoder"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    logger.info(f"  Extraído state_dict desde clave '{key}' en {weights_file}")
                    return ckpt[key]
        logger.info(f"  Cargado: {weights_file}")
        return ckpt

    # ── Auto-detección: safetensors primero ───────────────────────────────
    safetensors_files = list(model_path.glob("*.safetensors"))
    bin_files = list(model_path.glob("pytorch_model*.bin"))

    if safetensors_files:
        try:
            from safetensors.torch import load_file
            state_dict = {}
            for sf in sorted(safetensors_files):
                state_dict.update(load_file(str(sf), device="cpu"))
            logger.info(f"  Cargado safetensors: {[f.name for f in safetensors_files]}")
            return state_dict
        except ImportError:
            logger.warning("safetensors no instalado, usando .bin")

    if bin_files:
        state_dict = {}
        for bf in sorted(bin_files):
            ckpt = torch.load(str(bf), map_location="cpu")
            if isinstance(ckpt, dict) and not any(
                isinstance(v, torch.Tensor) for v in list(ckpt.values())[:3]
            ):
                # Es un checkpoint con claves anidadas, no state_dict directo
                for key in ("model", "state_dict"):
                    if key in ckpt:
                        ckpt = ckpt[key]
                        break
            state_dict.update(ckpt)
        logger.info(f"  Cargado bin: {[f.name for f in bin_files]}")
        return state_dict

    raise FileNotFoundError(
        f"No se encontraron pesos (.safetensors o pytorch_model*.bin) en {model_dir}"
    )


class Virchow2Wrapper(nn.Module):
    """
    Virchow2 produce tokens [CLS, REG..., PATCH...].
    El embedding final es: concat(CLS, mean(PATCH_TOKENS)) → 2048-dim
    según el paper oficial de Paige.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone.forward_features(x)
        # out shape: (B, 1 + num_reg + num_patches, D)
        cls_token    = out[:, 0]           # (B, D)
        patch_tokens = out[:, 5:]          # skip CLS + 4 register tokens
        patch_mean   = patch_tokens.mean(dim=1)  # (B, D)
        return torch.cat([cls_token, patch_mean], dim=-1)  # (B, 2D)


class PhikonV2Wrapper(nn.Module):
    """Phikon-v2 via transformers — extrae CLS token."""
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        return out.last_hidden_state[:, 0]  # CLS token


def load_embedding_model(
    model_name: str,
    models_dir: str,
    device: torch.device
) -> nn.Module:
    """
    Carga un modelo fundacional desde disco local (carpeta HuggingFace).

    Args:
        model_name : clave del MODEL_REGISTRY (uni2 | virchow2 | phikonv2 | provgigapath)
        models_dir : ruta base donde están todas las carpetas de modelos
        device     : dispositivo torch destino
    """
    import timm

    key = model_name.lower().replace("-", "").replace("_", "")
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{model_name}' no reconocido. "
            f"Opciones: {list(MODEL_REGISTRY.keys())}"
        )

    reg = MODEL_REGISTRY[key]
    model_dir = Path(models_dir) / model_name  # e.g. /home/.../models/virchow2
    if not model_dir.exists():
        raise FileNotFoundError(f"Carpeta del modelo no encontrada: {model_dir}")

    logger.info(f"Cargando modelo: {reg['description']}")
    logger.info(f"  Desde: {model_dir}")

    # ── timm con pesos locales ─────────────────────────────────────────────
    if reg["hf_type"] == "timm_local":
        kwargs = dict(reg.get("timm_kwargs", {"num_classes": 0}))
        if reg.get("uses_swiglu"):
            # Requiere imports en runtime: timm.layers.SwiGLUPacked y nn.SiLU
            from timm.layers import SwiGLUPacked
            kwargs["mlp_layer"] = SwiGLUPacked
            kwargs["act_layer"] = nn.SiLU
        model = timm.create_model(
            reg["timm_name"],
            pretrained=False,
            **kwargs
        )
        weights_file = reg.get("weights_file", None)  # e.g. "slide_encoder.pth"
        state_dict = _load_safetensors_or_bin(str(model_dir), weights_file)

        # Limpiar prefijos comunes en checkpoints HF
        cleaned = {}
        for k, v in state_dict.items():
            for prefix in ("model.", "backbone.", "encoder."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            cleaned[k] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            logger.debug(f"  Keys faltantes ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.debug(f"  Keys inesperadas ({len(unexpected)}): {unexpected[:5]}...")

        # Aplicar wrapper especial para Virchow2
        if reg.get("output_mode") == "virchow2":
            model = Virchow2Wrapper(model)

    # ── transformers AutoModel ────────────────────────────────────────────
    elif reg["hf_type"] == "transformers":
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=True
        )
        model = PhikonV2Wrapper(backbone)

    else:
        raise ValueError(f"hf_type desconocido: {reg['hf_type']}")

    model = model.eval().to(device)
    logger.info(f"  Modelo cargado correctamente en {device}")
    return model


def get_embedding_transform(model_name: str, patch_size: int = 224) -> T.Compose:
    """Retorna el transform correcto para cada modelo (normalización específica)."""
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
# HDF5 writer
# ─────────────────────────────────────────────

class HDF5Writer:
    """Thread-safe HDF5 writer for patches and embeddings."""

    def __init__(self, path: str, compression: str = "lzf"):
        self.path = path
        self.compression = compression if compression != "None" else None
        self._file = None

    def open(self, patch_size: int, n_channels: int = 3, embedding_dim: Optional[int] = None):
        self._file = h5py.File(self.path, "w")
        # Patches dataset (resizable)
        self._patches_ds = self._file.create_dataset(
            "patches",
            shape=(0, n_channels, patch_size, patch_size),
            maxshape=(None, n_channels, patch_size, patch_size),
            dtype=np.uint8,
            compression=self.compression,
            chunks=(16, n_channels, patch_size, patch_size)
        )
        # Coordinates dataset
        self._coords_ds = self._file.create_dataset(
            "coords",
            shape=(0, 2),
            maxshape=(None, 2),
            dtype=np.int32
        )
        if embedding_dim is not None:
            self._embeds_ds = self._file.create_dataset(
                "embeddings",
                shape=(0, embedding_dim),
                maxshape=(None, embedding_dim),
                dtype=np.float32,
                compression=self.compression,
                chunks=(64, embedding_dim)
            )
        else:
            self._embeds_ds = None

    def write_batch(
        self,
        patches: np.ndarray,      # (B, C, H, W) uint8
        coords: np.ndarray,       # (B, 2)
        embeddings: Optional[np.ndarray] = None  # (B, D)
    ):
        n = patches.shape[0]
        cur = self._patches_ds.shape[0]
        self._patches_ds.resize(cur + n, axis=0)
        self._patches_ds[cur:cur + n] = patches
        self._coords_ds.resize(cur + n, axis=0)
        self._coords_ds[cur:cur + n] = coords
        if embeddings is not None and self._embeds_ds is not None:
            self._embeds_ds.resize(cur + n, axis=0)
            self._embeds_ds[cur:cur + n] = embeddings

    def close(self):
        if self._file:
            self._file.close()

    def add_metadata(self, **kwargs):
        if self._file:
            for k, v in kwargs.items():
                self._file.attrs[k] = str(v)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ─────────────────────────────────────────────
# Individual image patch extractor
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
    dry_run: bool = False
) -> dict:
    """
    Extract patches from a single WSI file.
    Returns stats dict.
    """
    patch_size = ext_cfg["patch_size"]
    magnification = ext_cfg["magnification"]
    overlap = ext_cfg.get("overlap", 0.0)
    padding = ext_cfg.get("padding", True)

    tissue_thresh = filt_cfg.get("tissue_threshold", 0.5)
    filter_enabled = filt_cfg.get("enabled", True)
    sat_filter = filt_cfg.get("saturation_filter", True)
    sat_thresh = filt_cfg.get("saturation_threshold", 0.05)

    save_patches = out_cfg.get("save_patches", True)
    save_embeddings = out_cfg.get("save_embeddings", False)
    patches_fmt = out_cfg.get("patches_format", "hdf5")
    hdf5_compression = out_cfg.get("hdf5_compression", "lzf")
    embed_batch = out_cfg.get("embedding_batch_size", 64)
    jpeg_quality = out_cfg.get("jpeg_quality", 95)

    num_workers = comp_cfg.get("num_workers", 8)
    prefetch = comp_cfg.get("prefetch_factor", 2)
    pin_mem = comp_cfg.get("pin_memory", True)

    slide_name = Path(wsi_path).stem
    stats = {
        "slide": slide_name,
        "total_patches": 0,
        "kept_patches": 0,
        "filtered_patches": 0,
        "errors": 0,
        "time_s": 0.0
    }

    t0 = time.time()

    try:
        reader = WSIReader(
            wsi_path,
            target_magnification=magnification,
            native_magnification=ext_cfg.get("native_magnification", None)  # ← nuevo
        )
        slide_w, slide_h = reader.dimensions

        # ── Tissue mask ───────────────────────────────────────────────────
        if filter_enabled:
            thumb = reader.get_thumbnail((1024, 1024))
            mask_tensor = compute_tissue_mask_gpu(thumb, device, sat_filter, sat_thresh)
            mask_np = mask_tensor.numpy()
        else:
            mask_np = None

        # ── Generate patch coordinates ─────────────────────────────────────
        coords = generate_patch_coords(slide_w, slide_h, patch_size, overlap, padding)
        stats["total_patches"] = len(coords)

        if dry_run:
            reader.close()
            stats["time_s"] = time.time() - t0
            logger.info(
                f"[DRY-RUN] {slide_name}: {slide_w}x{slide_h}px @ target {magnification}x "
                f"→ {len(coords)} candidate patches"
            )
            return stats

        # ── Filter by tissue mask ─────────────────────────────────────────
        if filter_enabled and mask_np is not None:
            kept_coords = [
                (x, y) for (x, y) in coords
                if map_patch_to_mask(x, y, patch_size, patch_size,
                                     slide_w, slide_h, mask_np) >= tissue_thresh
            ]
        else:
            kept_coords = coords

        stats["kept_patches"] = len(kept_coords)
        stats["filtered_patches"] = stats["total_patches"] - stats["kept_patches"]

        if not kept_coords:
            logger.warning(f"[{slide_name}] No tissue patches found after filtering.")
            reader.close()
            return stats

        # ── Output setup ──────────────────────────────────────────────────
        if patches_fmt == "hdf5" and save_patches:
            hdf5_path = output_dir / f"{slide_name}.h5"
            writer = HDF5Writer(str(hdf5_path), compression=hdf5_compression)
            writer.open(patch_size, n_channels=3,
                        embedding_dim=embedding_dim if save_embeddings else None)
            writer.add_metadata(
                slide=slide_name,
                wsi_path=wsi_path,
                patch_size=patch_size,
                magnification=magnification,
                n_patches=len(kept_coords)
            )
        elif patches_fmt in ("png", "jpeg") and save_patches:
            img_dir = output_dir / slide_name
            img_dir.mkdir(parents=True, exist_ok=True)
            writer = None
        else:
            writer = None

        # ── Embedding-only output setup ────────────────────────────────────
        if save_embeddings and not save_patches:
            hdf5_path = output_dir / f"{slide_name}_embeddings.h5"
            writer = HDF5Writer(str(hdf5_path), compression=hdf5_compression)
            writer.open(patch_size, n_channels=3, embedding_dim=embedding_dim)
            writer.add_metadata(
                slide=slide_name,
                wsi_path=wsi_path,
                patch_size=patch_size,
                magnification=magnification,
                n_patches=len(kept_coords)
            )

        # ── Transform ────────────────────────────────────────────────────
        if save_embeddings and embedding_model is not None:
            model_name_key = out_cfg.get("embedding_model", "uni2")
            transform = get_embedding_transform(model_name_key, patch_size)
        else:
            transform = T.ToTensor()

        # ── DataLoader for efficient parallel reading ──────────────────────
        # Para slides PIL (BCNB .jpg), deshabilitamos persistent_workers:
        # con persistent_workers=True los workers viven entre slides y cada uno
        # mantiene su copia de la imagen en RAM (num_workers × tamaño_imagen).
        # Al deshabilitarlo, los workers se destruyen al terminar cada slide
        # y el GC libera la RAM antes de abrir la siguiente.
        is_pil_slide = Path(wsi_path).suffix.lower() in (".jpg", ".jpeg", ".png")
        use_persistent = num_workers > 0 and not is_pil_slide

        dataset = PatchDataset(
            wsi_path=wsi_path,
            coords=kept_coords,
            patch_size=patch_size,
            target_mag=magnification,
            native_mag=ext_cfg.get("native_magnification", None),  # BUG FIX: pasar native_mag a workers
            transform=transform
        )
        loader = DataLoader(
            dataset,
            batch_size=embed_batch,
            num_workers=num_workers,
            prefetch_factor=prefetch if num_workers > 0 else None,
            pin_memory=pin_mem and torch.cuda.is_available(),
            persistent_workers=use_persistent
        )

        # ── Main extraction loop ───────────────────────────────────────────
        with tqdm(
            total=len(kept_coords),
            desc=f"  {slide_name[:40]:40s}",
            unit="patch",
            leave=False,
            dynamic_ncols=True
        ) as pbar:
            for batch_tensors, batch_x, batch_y in loader:
                batch_size = batch_tensors.shape[0]
                coords_np = np.stack(
                    [batch_x.numpy(), batch_y.numpy()], axis=1
                ).astype(np.int32)

                # ── Embeddings ────────────────────────────────────────────
                if save_embeddings and embedding_model is not None:
                    with torch.no_grad():
                        gpu_tensors = batch_tensors.to(device, non_blocking=True)
                        embeds = embedding_model(gpu_tensors)
                        if isinstance(embeds, (list, tuple)):
                            embeds = embeds[0]
                        embeds_np = embeds.cpu().float().numpy()
                else:
                    embeds_np = None

                # ── Save patches ──────────────────────────────────────────
                if save_patches:
                    # Convert float tensor back to uint8 for saving
                    if batch_tensors.dtype == torch.float32:
                        patch_uint8 = (batch_tensors * 255).clamp(0, 255).byte().numpy()
                    else:
                        patch_uint8 = batch_tensors.numpy()

                    if patches_fmt == "hdf5" and writer is not None:
                        writer.write_batch(patch_uint8, coords_np, embeds_np)
                    elif patches_fmt in ("png", "jpeg"):
                        ext = "jpg" if patches_fmt == "jpeg" else "png"
                        for i in range(batch_size):
                            x_i, y_i = int(batch_x[i]), int(batch_y[i])
                            img_arr = np.transpose(patch_uint8[i], (1, 2, 0))
                            img = Image.fromarray(img_arr)
                            fname = img_dir / f"patch_{x_i:06d}_{y_i:06d}.{ext}"
                            if patches_fmt == "jpeg":
                                img.save(fname, quality=jpeg_quality)
                            else:
                                img.save(fname)
                elif save_embeddings and not save_patches and writer is not None:
                    # Dummy patches for HDF5 structure (only embeddings matter)
                    dummy = np.zeros(
                        (batch_size, 3, patch_size, patch_size), dtype=np.uint8
                    )
                    writer.write_batch(dummy, coords_np, embeds_np)

                pbar.update(batch_size)

        if writer is not None:
            writer.close()
        reader.close()

    except Exception as e:
        logger.error(f"[{Path(wsi_path).stem}] FAILED: {e}")
        logger.debug(traceback.format_exc())
        stats["errors"] += 1

    stats["time_s"] = time.time() - t0
    return stats


# ─────────────────────────────────────────────
# Dataset file discovery
# ─────────────────────────────────────────────

def discover_files(ds_cfg: dict) -> list:
    """Return list of WSI file paths based on dataset config."""
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
        # select por stem de archivo (ej. "00001")
        if select != "all" and isinstance(select, list):
            select_set = set(str(s) for s in select)
            files = [f for f in files if f.stem in select_set or f.name in select_set]

    elif structure == "case_folders":
        # case_xxxx/file.tiff — select filtra por nombre de CARPETA (ej. "case_0000")
        # BUG FIX: construir select_set una sola vez fuera del loop
        select_set = set(str(s) for s in select) if (select != "all" and isinstance(select, list)) else None
        for case_dir in sorted(base.iterdir()):
            if not case_dir.is_dir():
                continue
            # BUG FIX: excluir carpeta de output si está dentro del mismo directorio
            if case_dir.name == "patches":
                continue
            # Si hay select, solo incluir carpetas cuyo nombre esté en la lista
            if select_set is not None and case_dir.name not in select_set:
                continue
            for pat in patterns:
                files.extend(case_dir.glob(pat))
        files = sorted(files)

    elif structure == "split_folders":
        # SR1482_WSIs/file.czi — select filtra por stem de archivo
        for sub_dir in sorted(base.iterdir()):
            if sub_dir.is_dir():
                if sub_dir.name == "patches":  # BUG FIX: excluir carpeta de output
                    continue
                for pat in patterns:
                    files.extend(sub_dir.glob(pat))
        files = sorted(files)
        if select != "all" and isinstance(select, list):
            select_set = set(str(s) for s in select)
            files = [f for f in files if f.stem in select_set or f.name in select_set]

    return [str(f) for f in files]


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(config_path: str, dry_run: bool = False):
    cfg = load_config(config_path)

    setup_logging(
        log_level=cfg.get("log_level", "INFO"),
        log_file=cfg.get("log_file", None)
    )

    logger.info("=" * 70)
    logger.info("  WSI PATCH EXTRACTOR")
    logger.info(f"  Config: {config_path}")
    if dry_run:
        logger.info("  *** DRY RUN MODE - no files will be written ***")
    logger.info("=" * 70)

    # ── GPU setup ─────────────────────────────────────────────────────────
    comp_cfg = cfg.get("compute", {})
    gpu_ids = comp_cfg.get("gpu_ids", [0])
    if torch.cuda.is_available() and gpu_ids:
        device = torch.device(f"cuda:{gpu_ids[0]}")
        logger.info(f"GPUs available: {torch.cuda.device_count()}")
        for gid in gpu_ids:
            props = torch.cuda.get_device_properties(gid)
            logger.info(f"  GPU {gid}: {props.name} ({props.total_memory // 1024**3} GB)")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU found, falling back to CPU")

    # ── Extraction & filter configs ───────────────────────────────────────
    ext_cfg = cfg.get("extraction", {
        "patch_size": 224, "magnification": 20.0,
        "overlap": 0.0, "padding": True,
        "native_magnification": None  # BUG FIX: incluir en default para evitar KeyError
    })
    filt_cfg = cfg.get("filtering", {
        "enabled": True, "tissue_threshold": 0.5,
        "saturation_filter": True, "saturation_threshold": 0.05
    })
    out_cfg = cfg.get("output", {
        "base_dir": "./output", "save_patches": True,
        "patches_format": "hdf5", "save_embeddings": False,
        "hdf5_compression": "lzf", "embedding_batch_size": 64
    })

    output_base = Path(out_cfg.get("base_dir", "./output"))
    output_base.mkdir(parents=True, exist_ok=True)

    # ── Embedding model ───────────────────────────────────────────────────
    embedding_model = None
    embedding_dim = None
    if out_cfg.get("save_embeddings", False):
        model_name = out_cfg.get("embedding_model", "uni2")
        models_dir = out_cfg.get("models_dir", "/home/DIINF/vmieres/tesis/models")
        embedding_model = load_embedding_model(model_name, models_dir, device)
        if len(gpu_ids) > 1:
            logger.info(f"Using DataParallel across GPUs: {gpu_ids}")
            embedding_model = nn.DataParallel(embedding_model, device_ids=gpu_ids)
        embedding_model.eval()
        # Infer embedding dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, ext_cfg["patch_size"], ext_cfg["patch_size"]).to(device)
            out = embedding_model(dummy)
            if isinstance(out, (list, tuple)):
                out = out[0]
            embedding_dim = out.shape[-1]
        logger.info(f"Embedding dimension: {embedding_dim}")

    # ── Datasets ──────────────────────────────────────────────────────────
    datasets_cfg = cfg.get("datasets", [])
    if not datasets_cfg:
        logger.error("No datasets configured. Check your config file.")
        sys.exit(1)

    all_stats = []
    total_t0 = time.time()

    for ds in datasets_cfg:
        if not ds.get("enabled", True):
            logger.info(f"[{ds['name']}] SKIPPED (disabled)")
            continue

        logger.info(f"\n{'─'*60}")
        logger.info(f"Dataset: {ds['name']}  |  Format: {ds['format']}")
        logger.info(f"Path: {ds['path']}")

        files = discover_files(ds)
        if not files:
            logger.warning(f"[{ds['name']}] No files found!")
            continue

        logger.info(f"Found {len(files)} file(s)")

        ds_output = output_base / ds["name"]
        if not dry_run:
            ds_output.mkdir(parents=True, exist_ok=True)

        ds_stats = []
        with tqdm(
            total=len(files),
            desc=f"[{ds['name']}]",
            unit="slide",
            dynamic_ncols=True,
            position=0
        ) as slide_bar:
            for wsi_path in files:
                slide_bar.set_postfix_str(Path(wsi_path).name[:35])
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
                    dry_run=dry_run
                )
                ds_stats.append(stats)
                all_stats.append(stats)
                slide_bar.update(1)

        # Dataset summary
        total_kept = sum(s["kept_patches"] for s in ds_stats)
        total_cand = sum(s["total_patches"] for s in ds_stats)
        total_err  = sum(s["errors"] for s in ds_stats)
        total_time = sum(s["time_s"] for s in ds_stats)
        logger.info(
            f"\n[{ds['name']}] Summary: {len(files)} slides | "
            f"{total_kept:,}/{total_cand:,} patches kept "
            f"({100*total_kept/max(total_cand,1):.1f}%) | "
            f"Errors: {total_err} | "
            f"Time: {total_time/60:.1f} min"
        )

    # ── Global summary ────────────────────────────────────────────────────
    elapsed = time.time() - total_t0
    total_kept_all = sum(s["kept_patches"] for s in all_stats)
    total_cand_all = sum(s["total_patches"] for s in all_stats)
    total_err_all  = sum(s["errors"] for s in all_stats)

    logger.info(f"\n{'='*70}")
    logger.info("  GLOBAL SUMMARY")
    logger.info(f"  Total slides processed : {len(all_stats)}")
    logger.info(f"  Candidate patches      : {total_cand_all:,}")
    logger.info(f"  Kept patches           : {total_kept_all:,}")
    logger.info(
        f"  Filter rate            : "
        f"{100*(total_cand_all - total_kept_all)/max(total_cand_all,1):.1f}%"
    )
    logger.info(f"  Errors                 : {total_err_all}")
    logger.info(f"  Total time             : {elapsed/60:.1f} min")
    logger.info(f"  Output directory       : {output_base.resolve()}")
    logger.info("=" * 70)

    # ── Save stats CSV ────────────────────────────────────────────────────
    if not dry_run:
        import csv
        stats_path = output_base / "extraction_stats.csv"
        with open(stats_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=list(all_stats[0].keys()))
            writer_csv.writeheader()
            writer_csv.writerows(all_stats)
        logger.info(f"  Stats saved to         : {stats_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WSI Patch Extractor for Histology Datasets"
    )
    parser.add_argument(
        "config", type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan slides and count patches without extracting"
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    run_pipeline(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()