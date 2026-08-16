import argparse
import os
from pathlib import Path
from huggingface_hub import login, snapshot_download
from tqdm import tqdm

# ─── Configuración de modelos ───────────────────────────────────────────────
MODELS = {
    "virchow2": {
        "repo_id":    "paige-ai/Virchow2",
        "access":     "restricted",
        "arch":       "ViT-H/14",
        "size_aprox": "~2.5 GB",
    },
    "prov_gigapath": {
        "repo_id":    "prov-gigapath/prov-gigapath",
        "access":     "restricted",
        "arch":       "ViT-G/14",
        "size_aprox": "~3.5 GB",
    },
    "phikon_v2": {
        "repo_id":    "owkin/phikon-v2",
        "access":     "public",
        "arch":       "ViT-L/16",
        "size_aprox": "~1.2 GB",
    },
    "uni2": {
        "repo_id":    "MahmoodLab/UNI2-h",
        "access":     "restricted",
        "arch":       "ViT-H/14",
        "size_aprox": "~2.5 GB",
    },
}


def download_model(name: str, info: dict, cache_dir: Path):
    """Descarga un modelo completo con snapshot_download."""
    out_dir = cache_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Saltar si ya existe
    existing = list(out_dir.glob("*.safetensors")) + list(out_dir.glob("*.bin"))
    if existing:
        print(f"  [SKIP] {name} ya descargado en {out_dir}")
        return

    print(f"\n  Descargando {name} ({info['arch']}, {info['size_aprox']})...")
    print(f"  Repositorio: {info['repo_id']}")

    try:
        snapshot_download(
            repo_id=info["repo_id"],
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        # Verificar que se descargaron pesos
        weights = list(out_dir.glob("*.safetensors")) + list(out_dir.glob("*.bin"))
        if weights:
            total_mb = sum(f.stat().st_size for f in weights) / 1e6
            print(f"  [OK] {name} descargado → {out_dir}  ({total_mb:.0f} MB)")
        else:
            print(f"  [WARN] {name}: descarga completada pero no se encontraron pesos.")

    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        if info["access"] == "restricted":
            print(f"  → Asegúrate de haber solicitado acceso en:")
            print(f"    https://huggingface.co/{info['repo_id']}")


def main():
    parser = argparse.ArgumentParser(description="Descarga modelos FM de patología digital")
    parser.add_argument(
        "--token", default=None,
        help="Token de HuggingFace (hf_xxx). Opcional si ya hiciste `huggingface-cli login`"
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Modelos a descargar. Default: todos"
    )
    parser.add_argument(
        "--cache_dir", default="./models",
        help="Carpeta donde guardar los modelos. Default: ./models"
    )
    args = parser.parse_args()

    # Login
    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        print("[OK] Login HuggingFace exitoso.\n")
    else:
        print("[INFO] Sin token explícito, usando credenciales guardadas.")
        print("       Si falla, corre: huggingface-cli login\n")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Selección de modelos
    to_download = list(MODELS.keys()) if "all" in args.models else args.models

    print("=" * 55)
    print("  Modelos a descargar:")
    for name in to_download:
        info = MODELS[name]
        flag = "🔓 público" if info["access"] == "public" else "🔒 restringido"
        print(f"    • {name:<15} {info['arch']:<12} {info['size_aprox']:<10} {flag}")
    print(f"\n  Destino: {cache_dir.resolve()}")
    print("=" * 55)

    # Descargar
    for name in to_download:
        download_model(name, MODELS[name], cache_dir)

    print("\n" + "=" * 55)
    print("  Descarga completada.")
    print(f"  Modelos guardados en: {cache_dir.resolve()}")
    print("\n  Para cargar desde disco en tus scripts:")
    print(f"    MODEL_CACHE = '{cache_dir.resolve()}'")
    print("=" * 55)


if __name__ == "__main__":
    main()