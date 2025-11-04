#!/usr/bin/env python3
"""
Скрипт для скачивания основных датасетов по заболеваниям растений
для проекта в Красноярском крае.
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
from zipfile import ZipFile
import tarfile

# Создаем директории для датасетов
DATASETS_DIR = Path("datasets")
DATASETS_DIR.mkdir(exist_ok=True)

def download_file(url, dest_path, description=""):
    """Скачать файл по URL"""
    print(f"\n📥 Скачивание {description}...")
    print(f"   URL: {url}")
    print(f"   Путь: {dest_path}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        os.makedirs(dest_path.parent, exist_ok=True)
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Прогресс: {percent:.1f}%", end='', flush=True)
        
        print(f"\n   ✅ Успешно скачан: {dest_path}")
        return True
        
    except Exception as e:
        print(f"\n   ❌ Ошибка: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Распаковать ZIP файл"""
    print(f"\n📦 Распаковка {zip_path}...")
    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"   ✅ Распакован в: {extract_to}")
        return True
    except Exception as e:
        print(f"   ❌ Ошибка распаковки: {e}")
        return False

def extract_tar(tar_path, extract_to):
    """Распаковать TAR файл"""
    print(f"\n📦 Распаковка {tar_path}...")
    try:
        with tarfile.open(tar_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
        print(f"   ✅ Распакован в: {extract_to}")
        return True
    except Exception as e:
        print(f"   ❌ Ошибка распаковки: {e}")
        return False

def download_plantdoc():
    """Скачать PlantDoc Dataset"""
    print("\n" + "="*60)
    print("🌿 PlantDoc Dataset")
    print("="*60)
    
    plantdoc_dir = DATASETS_DIR / "plantdoc"
    plantdoc_dir.mkdir(exist_ok=True)
    
    # PlantDoc доступен через GitHub зеркала или можно клонировать репозиторий
    # Попробуем скачать с альтернативных источников
    print("\n⚠️  PlantDoc Dataset лучше скачать вручную:")
    print("   1. Клонировать репозиторий: git clone https://github.com/pratikkayal/PlantDoc-Dataset.git")
    print("   2. Или скачать с Kaggle зеркала")
    
    # Проверяем, есть ли уже данные
    if (plantdoc_dir / "train").exists() or (plantdoc_dir / "PlantDoc-Dataset").exists():
        print("   ✅ PlantDoc уже присутствует")
        return True
    
    return False

def download_plantvillage():
    """Скачать PlantVillage Dataset"""
    print("\n" + "="*60)
    print("🌾 PlantVillage Dataset")
    print("="*60)
    
    plantvillage_dir = DATASETS_DIR / "plantvillage"
    plantvillage_dir.mkdir(exist_ok=True)
    
    # Проверяем, не скачан ли уже
    if (plantvillage_dir / "color").exists() or (plantvillage_dir / "segmented").exists():
        print("   ✅ PlantVillage уже присутствует")
        return True
    
    # PlantVillage можно скачать с нескольких источников
    # 1. С Kaggle (требует аутентификацию)
    # 2. С GitHub (через git clone)
    # 3. Прямые ссылки на зеркала
    
    print("\n📋 Варианты скачивания PlantVillage:")
    print("   1. Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print("   2. GitHub: git clone https://github.com/spMohanty/PlantVillage-Dataset.git")
    print("   3. Прямая ссылка (может быть медленной)")
    
    # Пробуем скачать через Kaggle зеркало или прямую ссылку
    kaggle_url = "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/download"
    
    print("\n⚠️  PlantVillage Dataset лучше скачать вручную:")
    print("   Команда: kaggle datasets download -d abdallahalidev/plantvillage-dataset")
    print("   Или: git clone https://github.com/spMohanty/PlantVillage-Dataset.git datasets/plantvillage/")
    
    return False

def download_plant_pathology_2021():
    """Скачать Plant Pathology 2021 - FGVC8 (Пшеница)"""
    print("\n" + "="*60)
    print("🌾 Plant Pathology 2021 - FGVC8 (Пшеница)")
    print("="*60)
    
    path_dir = DATASETS_DIR / "plant_pathology_2021"
    path_dir.mkdir(exist_ok=True)
    
    # Проверяем наличие
    if (path_dir / "train_images").exists() or (path_dir / "test_images").exists():
        print("   ✅ Plant Pathology 2021 уже присутствует")
        return True
    
    print("\n⚠️  Plant Pathology 2021 требует Kaggle API:")
    print("   1. Установите Kaggle: pip install kaggle")
    print("   2. Настройте credentials: ~/.kaggle/kaggle.json")
    print("   3. Команда: kaggle competitions download -c plant-pathology-2021-fgvc8")
    
    # Проверяем наличие Kaggle CLI
    try:
        result = subprocess.run(['kaggle', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\n   ✅ Kaggle CLI найден, пытаемся скачать...")
            
            # Пробуем скачать
            try:
                os.chdir(path_dir)
                result = subprocess.run(
                    ['kaggle', 'competitions', 'download', '-c', 'plant-pathology-2021-fgvc8'],
                    capture_output=True,
                    timeout=300
                )
                os.chdir("..")
                
                if result.returncode == 0:
                    print("   ✅ Успешно скачан через Kaggle CLI")
                    # Распаковываем если нужно
                    for file in path_dir.glob("*.zip"):
                        extract_zip(file, path_dir)
                    return True
                else:
                    print(f"   ⚠️  Ошибка Kaggle CLI: {result.stderr.decode()}")
            except subprocess.TimeoutExpired:
                print("   ⚠️  Превышено время ожидания")
            except Exception as e:
                print(f"   ⚠️  Ошибка: {e}")
        else:
            print("   ❌ Kaggle CLI не работает")
    except FileNotFoundError:
        print("   ❌ Kaggle CLI не установлен")
    
    return False

def create_download_script():
    """Создать bash скрипт для скачивания"""
    script_content = """#!/bin/bash
# Скрипт для скачивания датасетов

echo "🌿 Скачивание датасетов для проекта..."

# 1. PlantVillage Dataset (GitHub)
if [ ! -d "datasets/plantvillage/PlantVillage-Dataset" ]; then
    echo "📥 Скачивание PlantVillage..."
    git clone https://github.com/spMohanty/PlantVillage-Dataset.git datasets/plantvillage/PlantVillage-Dataset || \\
    echo "⚠️  Git clone не удался, используйте: kaggle datasets download -d abdallahalidev/plantvillage-dataset"
fi

# 2. PlantDoc Dataset
if [ ! -d "datasets/plantdoc/PlantDoc-Dataset" ]; then
    echo "📥 Скачивание PlantDoc..."
    git clone https://github.com/pratikkayal/PlantDoc-Dataset.git datasets/plantdoc/PlantDoc-Dataset || \\
    echo "⚠️  Git clone не удался, используйте альтернативные источники"
fi

# 3. Plant Pathology 2021 (требует Kaggle API)
if [ ! -d "datasets/plant_pathology_2021/train_images" ]; then
    echo "📥 Скачивание Plant Pathology 2021..."
    kaggle competitions download -c plant-pathology-2021-fgvc8 -p datasets/plant_pathology_2021/
    
    # Распаковка
    cd datasets/plant_pathology_2021/
    unzip -q *.zip 2>/dev/null || echo "Распакуйте ZIP файлы вручную"
    cd ../..
fi

echo "✅ Загрузка завершена!"
"""
    
    script_path = DATASETS_DIR / "download_datasets.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"\n📝 Создан bash скрипт: {script_path}")
    
    return script_path

def main():
    """Основная функция"""
    print("\n" + "="*60)
    print("🌾 Скачивание датасетов для определения заболеваний растений")
    print("="*60)
    
    results = {
        "PlantVillage": False,
        "PlantDoc": False,
        "Plant Pathology 2021": False
    }
    
    # Пробуем скачать датасеты
    results["PlantVillage"] = download_plantvillage()
    results["PlantDoc"] = download_plantdoc()
    results["Plant Pathology 2021"] = download_plant_pathology_2021()
    
    # Создаем bash скрипт для ручного скачивания
    script_path = create_download_script()
    
    # Итоги
    print("\n" + "="*60)
    print("📊 Итоги скачивания:")
    print("="*60)
    
    for name, success in results.items():
        status = "✅" if success else "⚠️  (требуется ручное скачивание)"
        print(f"   {name}: {status}")
    
    print("\n" + "="*60)
    print("📋 Инструкции для ручного скачивания:")
    print("="*60)
    print(f"\n1. Используйте bash скрипт: bash {script_path}")
    print("\n2. Или выполните команды вручную:")
    print("\n   🌾 PlantVillage:")
    print("      git clone https://github.com/spMohanty/PlantVillage-Dataset.git datasets/plantvillage/")
    print("      ИЛИ")
    print("      kaggle datasets download -d abdallahalidev/plantvillage-dataset -p datasets/plantvillage/")
    print("\n   🌿 PlantDoc:")
    print("      git clone https://github.com/pratikkayal/PlantDoc-Dataset.git datasets/plantdoc/")
    print("\n   🌾 Plant Pathology 2021 (Пшеница):")
    print("      kaggle competitions download -c plant-pathology-2021-fgvc8 -p datasets/plant_pathology_2021/")
    print("\n   После скачивания распакуйте ZIP файлы при необходимости.")
    
    print("\n✅ Готово!")

if __name__ == "__main__":
    main()

