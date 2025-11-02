"""
Адаптированный скрипт для проведения экспериментов с PlantVillage датасетом
Использует существующие функции из run_experiments.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from PIL import Image
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from collections import defaultdict

# Импортируем все функции из основного скрипта
sys.path.insert(0, os.path.dirname(__file__))
from run_experiments import (
    ImprovedCassavaModel,
    FocalLoss,
    LabelSmoothingCrossEntropy,
    get_standard_augmentations,
    get_aggressive_augmentations,
    get_test_augmentations,
    mixup_data,
    cutmix_data,
    tta_predict_batch,
    find_optimal_thresholds,
    evaluate_model,
    experiment_baseline,
    experiment_tta,
    experiment_threshold_tuning,
    experiment_tta_threshold,
    experiment_tta_3,
    experiment_tta_7,
    experiment_temperature_scaling,
    experiment_weighted_voting,
    experiment_temp_scaling_tta
)

# Настройки
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Путь к датасету PlantVillage
PLANTVILLAGE_PATH = "/home/ann/Загрузки/archive(1)/plantvillage/PlantVillage/"

# ============================================================================
# ДАТАСЕТ ДЛЯ PLANTVILLAGE
# ============================================================================

class PlantVillageDataset(Dataset):
    """Датасет для PlantVillage - загружает изображения из директорий классов"""
    
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        
        # Собираем все изображения с их метками
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Получаем все директории классов
        class_dirs = sorted([d for d in self.dataset_path.iterdir() if d.is_dir()])
        
        # Создаем маппинг классов
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            
            # Находим все изображения в директории класса
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG']:
                image_files.extend(list(class_dir.glob(f'*{ext}')))
            
            # Добавляем изображения с метками
            for img_path in image_files:
                self.images.append(str(img_path))
                self.labels.append(idx)
        
        print(f"📊 Загружено {len(self.images)} изображений из {len(class_dirs)} классов")
        print(f"   Классы: {list(self.class_to_idx.keys())[:5]}...")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Загружаем изображение
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"⚠️  Ошибка загрузки изображения {img_path}: {e}")
            # Возвращаем черное изображение как fallback
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Применяем трансформации
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return {
            'image': image,
            'label': label
        }
    
    def get_class_names(self):
        """Возвращает список имен классов"""
        return [self.idx_to_class[i] for i in range(len(self.idx_to_class))]


# ============================================================================
# ЗАГРУЗКА МОДЕЛИ И ДАННЫХ
# ============================================================================

def load_model_and_data_plantvillage():
    """Загружает модель и PlantVillage данные"""
    # Загрузка датасета
    print("📥 Загрузка PlantVillage датасета...")
    full_dataset = PlantVillageDataset(PLANTVILLAGE_PATH, transform=None)
    
    # Получаем имена классов
    class_names = full_dataset.get_class_names()
    num_classes = len(class_names)
    
    print(f"✅ Найдено классов: {num_classes}")
    print(f"   Всего изображений: {len(full_dataset)}")
    
    # Разделяем на train/val/test (80/10/10)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Применяем трансформации
    test_transform = get_test_augmentations()
    
    # Создаем тестовый датасет с трансформациями
    class TestDatasetWrapper(Dataset):
        def __init__(self, subset_dataset, transform):
            self.subset_dataset = subset_dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.subset_dataset)
        
        def __getitem__(self, idx):
            data = self.subset_dataset[idx]
            image = data['image']
            label = data['label']
            
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            
            return {'image': image, 'label': label}
    
    test_dataset_transformed = TestDatasetWrapper(test_dataset, test_transform)
    val_dataset_transformed = TestDatasetWrapper(val_dataset, test_transform)
    
    test_loader = DataLoader(test_dataset_transformed, batch_size=32, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset_transformed, batch_size=32, shuffle=False, num_workers=4)
    
    # Загрузка или создание модели
    print(f"\n📦 Создание модели для {num_classes} классов...")
    model = ImprovedCassavaModel(num_classes=num_classes, model_name='efficientnet_b0', pretrained=True)
    
    # Если есть предобученная модель, можно загрузить её и адаптировать последний слой
    # Но для PlantVillage нужно обучать с нуля или использовать transfer learning
    model.to(device)
    
    print(f"✅ Модель создана: EfficientNet-B0 с {num_classes} классами")
    
    return model, test_loader, val_loader, class_names


# ============================================================================
# БЫСТРОЕ ОБУЧЕНИЕ МОДЕЛИ (Transfer Learning)
# ============================================================================

def train_model_quick(model, train_loader, val_loader, num_epochs=5, num_classes=15):
    """Быстрое обучение модели для получения baseline"""
    print(f"\n🎓 Обучение модели ({num_epochs} эпох)...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_bar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{train_loss/(train_bar.n+1):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in val_bar:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': f'{val_loss/(val_bar.n+1):.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        scheduler.step()
        print(f'Epoch {epoch+1}: Train Acc={100.*train_correct/train_total:.2f}%, '
              f'Val Acc={val_acc:.2f}%, Best Val Acc={best_val_acc:.2f}%')
    
    # Загружаем лучшую модель
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"✅ Обучение завершено. Лучшая точность на валидации: {best_val_acc:.2f}%")
    return model


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главная функция для проведения всех экспериментов с PlantVillage"""
    print("="*60)
    print("ЭКСПЕРИМЕНТЫ С ДАТАСЕТОМ PLANTVILLAGE")
    print("="*60)
    
    # Загружаем модель и данные
    model, test_loader, val_loader, class_names = load_model_and_data_plantvillage()
    num_classes = len(class_names)
    
    # Если модель не обучена, проводим быстрое обучение
    # Для демонстрации можно пропустить обучение и использовать случайную модель
    # Или загрузить предобученную модель и адаптировать последний слой
    
    # Проверяем, нужно ли обучать модель
    print("\n⚠️  ВНИМАНИЕ: Модель не обучена на PlantVillage!")
    print("   Для получения реальных результатов нужно обучить модель.")
    print("   Для демонстрации экспериментов будем использовать предобученную модель.")
    
    # Если хотим обучить быстро, раскомментируйте:
    # train_dataset = PlantVillageDataset(PLANTVILLAGE_PATH, transform=get_standard_augmentations())
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # model = train_model_quick(model, train_loader, val_loader, num_epochs=5, num_classes=num_classes)
    
    results = []
    
    # Эксперимент 0: Baseline
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 0: БАЗОВАЯ МОДЕЛЬ")
    print("="*60)
    baseline_result = experiment_baseline(model, test_loader, class_names, device)
    results.append(baseline_result)
    
    # Эксперимент 1: TTA
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 1: БАЗОВАЯ МОДЕЛЬ + TTA")
    print("="*60)
    tta_result = experiment_tta(model, test_loader, class_names, device)
    results.append(tta_result)
    
    # Эксперимент 2: Threshold Tuning
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 2: БАЗОВАЯ МОДЕЛЬ + THRESHOLD TUNING")
    print("="*60)
    threshold_result = experiment_threshold_tuning(model, test_loader, val_loader, class_names, device)
    results.append(threshold_result)
    
    # Эксперимент 3: TTA + Threshold
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 3: БАЗОВАЯ МОДЕЛЬ + TTA + THRESHOLD TUNING")
    print("="*60)
    tta_threshold_result = experiment_tta_threshold(model, test_loader, val_loader, class_names, device)
    results.append(tta_threshold_result)
    
    # Эксперимент 4: TTA с 3 аугментациями
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 4: БАЗОВАЯ МОДЕЛЬ + TTA (3 аугментации)")
    print("="*60)
    tta_3_result = experiment_tta_3(model, test_loader, class_names, device)
    results.append(tta_3_result)
    
    # Эксперимент 5: TTA с 7 аугментациями
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 5: БАЗОВАЯ МОДЕЛЬ + TTA (7 аугментаций)")
    print("="*60)
    tta_7_result = experiment_tta_7(model, test_loader, class_names, device)
    results.append(tta_7_result)
    
    # Эксперимент 6: Temperature Scaling
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 6: БАЗОВАЯ МОДЕЛЬ + TEMPERATURE SCALING")
    print("="*60)
    temp_scaling_result = experiment_temperature_scaling(model, test_loader, val_loader, class_names, device)
    results.append(temp_scaling_result)
    
    # Эксперимент 7: Weighted Voting
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 7: БАЗОВАЯ МОДЕЛЬ + WEIGHTED VOTING")
    print("="*60)
    weighted_voting_result = experiment_weighted_voting(model, test_loader, class_names, device)
    results.append(weighted_voting_result)
    
    # Эксперимент 8: Temperature Scaling + TTA
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 8: БАЗОВАЯ МОДЕЛЬ + TEMP SCALING + TTA")
    print("="*60)
    temp_tta_result = experiment_temp_scaling_tta(model, test_loader, val_loader, class_names, device)
    results.append(temp_tta_result)
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ВСЕХ ЭКСПЕРИМЕНТОВ")
    print("="*60)
    
    # Создаем DataFrame для сводной таблицы
    summary_data = []
    for result in results:
        summary_data.append({
            'Эксперимент': result['experiment_name'],
            'Описание': result['description'],
            'Accuracy': f"{result['accuracy']*100:.2f}%",
            'Precision (Macro)': f"{result['precision_macro']*100:.2f}%",
            'Recall (Macro)': f"{result['recall_macro']*100:.2f}%",
            'F1-Score (Macro)': f"{result['f1_macro']*100:.2f}%",
            'Precision (Weighted)': f"{result['precision_weighted']*100:.2f}%",
            'Recall (Weighted)': f"{result['recall_weighted']*100:.2f}%",
            'F1-Score (Weighted)': f"{result['f1_weighted']*100:.2f}%",
        })
    
    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"experiments/plantvillage_experiments_{timestamp}.json"
    os.makedirs("experiments", exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'dataset': 'PlantVillage',
            'num_classes': num_classes,
            'class_names': class_names,
            'timestamp': timestamp,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Результаты сохранены: {results_file}")
    
    # Создаем сводную таблицу
    summary_file = f"experiments/plantvillage_summary_{timestamp}.csv"
    df.to_csv(summary_file, index=False, encoding='utf-8')
    print(f"✅ Сводная таблица сохранена: {summary_file}")
    
    print("\n" + "="*60)
    print("✅ ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print("="*60)


if __name__ == "__main__":
    main()

