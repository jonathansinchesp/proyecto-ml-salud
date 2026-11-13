import os
import shutil
import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuración de rutas
RAW_DATA_DIR = Path("chest_xray")
PROCESSED_DATA_DIR = Path("data_processed")
MODELS_DIR = Path("models")

# Semilla para reproducibilidad científica
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def find_base_dir(raw_dir: Path) -> Path:
    """Detecta si Kaggle creó una subcarpeta chest_xray/chest_xray."""
    if (raw_dir / "chest_xray" / "train").exists():
        return raw_dir / "chest_xray"
    if (raw_dir / "train").exists():
        return raw_dir
    raise FileNotFoundError("No se encontraron las carpetas train/test dentro de chest_xray.")

def collect_images(base_dir: Path):
    """Recopila todas las imágenes y las clasifica en Normal, Bacterial o Viral."""
    dataset = {"Normal": [], "Bacterial": [], "Viral": []}
    valid_extensions = {".jpeg", ".jpg", ".png"}

    for split in ["train", "test", "val"]:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue

        # 1. Imágenes Normales
        normal_dir = split_dir / "NORMAL"
        if normal_dir.exists():
            for img_path in normal_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    dataset["Normal"].append(img_path)

        # 2. Imágenes con Neumonía (separadas por nombre de archivo)
        pneumonia_dir = split_dir / "PNEUMONIA"
        if pneumonia_dir.exists():
            for img_path in pneumonia_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    filename = img_path.name.lower()
                    if "bacteria" in filename:
                        dataset["Bacterial"].append(img_path)
                    elif "virus" in filename:
                        dataset["Viral"].append(img_path)

    return dataset

def prepare_dataset():
    print("🔍 Buscando directorio base del dataset...")
    base_dir = find_base_dir(RAW_DATA_DIR)
    print(f"✅ Directorio encontrado: {base_dir}")

    print("📊 Clasificando imágenes en 3 categorías (Normal, Bacterial, Viral)...")
    dataset = collect_images(base_dir)

    for label, files in dataset.items():
        print(f"   - {label}: {len(files)} imágenes encontradas")

    # Limpiar carpeta data_processed si ya existía
    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)

    stats = {
        "total_images": 0,
        "classes": {},
        "splits": {"train": {}, "val": {}, "test": {}}
    }

    # División estratificada: 80% train, 10% val, 10% test
    for label, files in dataset.items():
        train_files, temp_files = train_test_split(
            files, test_size=0.20, random_state=RANDOM_SEED
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=0.50, random_state=RANDOM_SEED
        )

        splits_map = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }

        stats["classes"][label] = len(files)
        stats["total_images"] += len(files)

        for split_name, split_files in splits_map.items():
            dest_dir = PROCESSED_DATA_DIR / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            stats["splits"][split_name][label] = len(split_files)

            for idx, src_path in enumerate(split_files):
                dest_path = dest_dir / f"{label.lower()}_{idx:04d}{src_path.suffix.lower()}"
                shutil.copy2(src_path, dest_path)

    # Guardar estadísticas para el dashboard de Streamlit
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = MODELS_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print("\n🎉 ¡Preprocesamiento completado con éxito!")
    print(f"📁 Datos organizados en: {PROCESSED_DATA_DIR}/")
    print(f"📈 Estadísticas guardadas en: {stats_path}")

if __name__ == "__main__":
    prepare_dataset()