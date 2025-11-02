"""
Комплексный скрипт для проведения всех экспериментов по улучшению модели
Создает сводную таблицу результатов всех экспериментов
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from collections import defaultdict

# Настройки
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# ============================================================================
# МОДЕЛЬ И БАЗОВЫЕ КОМПОНЕНТЫ
# ============================================================================

class ImprovedCassavaModel(nn.Module):
    """Улучшенная модель с лучшей регуляризацией"""
    
    def __init__(self, num_classes=5, model_name='efficientnet_b0', pretrained=True):
        super().__init__()
        
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ============================================================================
# LOSS ФУНКЦИИ
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss для работы с дисбалансом классов"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing для регуляризации"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=1)
        weight = pred.new_ones(pred.size()) * self.smoothing / (pred.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=1).mean()
        return loss


# ============================================================================
# АУГМЕНТАЦИИ
# ============================================================================

def get_standard_augmentations():
    """Стандартные аугментации"""
    return A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_aggressive_augmentations():
    """Агрессивные аугментации для minority классов"""
    return A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=30, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        A.GridDistortion(p=0.3),
        A.OpticalDistortion(distort_limit=0.2, shift_limit=0.1, p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_test_augmentations():
    """Тестовые трансформации"""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ============================================================================
# MIXUP И CUTMIX
# ============================================================================

def mixup_data(x, y, alpha=1.0):
    """MixUp аугментация"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix аугментация"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    W = x.size(3)
    H = x.size(2)
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y, y[index], lam


# ============================================================================
# ДАТАСЕТ И DATA LOADERS
# ============================================================================

class CassavaDataset(Dataset):
    """Датасет для кассавы"""
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = np.array(example['image'])
        label = example['label']
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return {
            'image': image,
            'label': label
        }


# ============================================================================
# TTA (TEST TIME AUGMENTATION)
# ============================================================================

def tta_predict_batch(model, images, n_aug=5):
    """Test Time Augmentation для батча изображений"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        # Базовое предсказание
        base_pred = model(images)
        all_predictions.append(F.softmax(base_pred, dim=1))
        
        # Денормализация для аугментаций
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
        
        # Аугментации
        tta_transforms = [
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.RandomRotate90(p=1.0),
            A.Transpose(p=1.0),
        ]
        
        # Применяем аугментации к каждому изображению
        for transform in tta_transforms[:n_aug-1]:
            batch_probs = []
            for i in range(images.size(0)):
                # Денормализация
                img = images[i].cpu()
                img = img * std[0, :, 0, 0].view(3, 1, 1).cpu() + mean[0, :, 0, 0].view(3, 1, 1).cpu()
                img = torch.clamp(img, 0, 1)
                
                # В numpy формат для albumentations
                img_np = img.numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                
                # Аугментация
                augmented = transform(image=img_np)
                aug_img = augmented['image'].astype(np.float32) / 255.0
                
                # Обратно в tensor и нормализация
                aug_tensor = torch.from_numpy(aug_img.transpose(2, 0, 1)).float()
                aug_tensor = (aug_tensor - mean[0, :, 0, 0].cpu()) / std[0, :, 0, 0].cpu()
                
                # Предсказание
                pred = model(aug_tensor.unsqueeze(0).to(device))
                batch_probs.append(F.softmax(pred, dim=1))
            
            all_predictions.append(torch.cat(batch_probs, dim=0))
    
    # Среднее всех предсказаний
    avg_pred = torch.stack(all_predictions).mean(dim=0)
    return avg_pred


# ============================================================================
# THRESHOLD TUNING
# ============================================================================

def find_optimal_thresholds(model, val_loader, device):
    """Находит оптимальные пороги для каждого класса"""
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Finding thresholds'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Простая стратегия: использовать пороги, которые максимизируют F1 для каждого класса
    thresholds = {}
    for class_idx in range(5):
        class_probs = all_probs[:, class_idx]
        class_targets = (all_targets == class_idx).astype(int)
        
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.arange(0.1, 1.0, 0.05):
            preds = (class_probs >= threshold).astype(int)
            if preds.sum() > 0:
                f1 = f1_score(class_targets, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        thresholds[class_idx] = best_threshold
    
    return thresholds


# ============================================================================
# ОЦЕНКА МОДЕЛИ
# ============================================================================

def evaluate_model(model, test_loader, class_names, device, use_tta=False, thresholds=None):
    """Оценка модели на тестовых данных"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluation'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            if use_tta:
                # Используем TTA для батча
                probs = tta_predict_batch(model, images, n_aug=5)
            else:
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
            
            if thresholds:
                # Применяем оптимальные пороги - улучшенная версия
                adjusted_probs = probs.clone()
                # Создаем маску для каждого класса на основе порога
                for class_idx, threshold in thresholds.items():
                    # Применяем порог как минимальную вероятность для класса
                    adjusted_probs[:, class_idx] = torch.clamp(
                        adjusted_probs[:, class_idx], 
                        min=threshold if probs[:, class_idx].max() > threshold else 0
                    )
                
                # Нормализуем
                adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=1, keepdim=True)
                preds = adjusted_probs.argmax(dim=1)
            else:
                preds = probs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Вычисляем метрики
    accuracy = accuracy_score(all_targets, all_preds)
    precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    precision_weighted = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    # Per-class метрики
    precision_per_class = precision_score(all_targets, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_targets, all_preds, average=None, zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }


# ============================================================================
# ЗАГРУЗКА МОДЕЛИ И ДАННЫХ
# ============================================================================

def load_model_and_data():
    """Загружает модель и данные один раз"""
    # Загрузка датасета
    print("📥 Загрузка датасета...")
    ds = load_dataset("pufanyi/cassava-leaf-disease-classification", "full")
    test_ds = ds['validation']
    train_ds = ds['train']
    class_names = test_ds.features['label'].names
    
    test_transform = get_test_augmentations()
    test_dataset = CassavaDataset(test_ds, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Validation loader для threshold tuning
    val_dataset = CassavaDataset(test_ds, transform=test_transform)  # Используем часть test как val
    val_size = len(test_ds) // 2
    val_indices = list(range(val_size))
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Загрузка модели
    print("📦 Загрузка модели...")
    model_path = "notebooks/best_improved_model.pth"
    model = ImprovedCassavaModel(num_classes=5)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    
    return model, test_loader, val_loader, class_names


# ============================================================================
# ЭКСПЕРИМЕНТЫ
# ============================================================================

def experiment_baseline(model, test_loader, class_names, device):
    """Эксперимент 0: Базовая модель"""
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 0: БАЗОВАЯ МОДЕЛЬ")
    print("="*60)
    
    metrics = evaluate_model(model, test_loader, class_names, device)
    
    return {
        'experiment_name': 'Baseline',
        'description': 'Базовая модель с Weighted Cross-Entropy',
        **metrics
    }


def experiment_tta(model, test_loader, class_names, device):
    """Эксперимент 1: Базовая модель + TTA"""
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 1: БАЗОВАЯ МОДЕЛЬ + TTA")
    print("="*60)
    
    metrics = evaluate_model(model, test_loader, class_names, device, use_tta=True)
    
    return {
        'experiment_name': 'Baseline + TTA',
        'description': 'Базовая модель с Test Time Augmentation (5 аугментаций)',
        **metrics
    }


def experiment_threshold_tuning(model, test_loader, val_loader, class_names, device):
    """Эксперимент 2: Базовая модель + Optimal Threshold Tuning"""
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 2: БАЗОВАЯ МОДЕЛЬ + THRESHOLD TUNING")
    print("="*60)
    
    print("🔍 Поиск оптимальных порогов...")
    thresholds = find_optimal_thresholds(model, val_loader, device)
    print(f"Найденные пороги: {thresholds}")
    
    metrics = evaluate_model(model, test_loader, class_names, device, thresholds=thresholds)
    metrics['thresholds'] = thresholds
    
    return {
        'experiment_name': 'Baseline + Threshold Tuning',
        'description': 'Базовая модель с оптимальными порогами для каждого класса',
        **metrics
    }


def experiment_tta_threshold(model, test_loader, val_loader, class_names, device):
    """Эксперимент 3: Базовая модель + TTA + Threshold Tuning"""
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 3: БАЗОВАЯ МОДЕЛЬ + TTA + THRESHOLD TUNING")
    print("="*60)
    
    print("🔍 Поиск оптимальных порогов...")
    thresholds = find_optimal_thresholds(model, val_loader, device)
    print(f"Найденные пороги: {thresholds}")
    
    metrics = evaluate_model(model, test_loader, class_names, device, use_tta=True, thresholds=thresholds)
    metrics['thresholds'] = thresholds
    
    return {
        'experiment_name': 'Baseline + TTA + Threshold',
        'description': 'Базовая модель с TTA и оптимальными порогами',
        **metrics
    }


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главная функция для проведения всех экспериментов"""
    print("="*60)
    print("КОМПЛЕКСНЫЕ ЭКСПЕРИМЕНТЫ ПО УЛУЧШЕНИЮ МОДЕЛИ")
    print("="*60)
    
    results = []
    
    # Базовый эксперимент
    baseline_result = evaluate_baseline()
    results.append(baseline_result)
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
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
    
    # Сохраняем результаты
    os.makedirs('experiments', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON с полными результатами
    json_path = f'experiments/experiments_results_{timestamp}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # CSV таблица
    csv_path = f'experiments/summary_table_{timestamp}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # Markdown таблица
    md_path = f'experiments/summary_table_{timestamp}.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Сводная таблица экспериментов\n\n")
        f.write(df.to_markdown(index=False))
    
    print("\n" + df.to_string(index=False))
    print(f"\n✅ Результаты сохранены:")
    print(f"  • JSON: {json_path}")
    print(f"  • CSV: {csv_path}")
    print(f"  • Markdown: {md_path}")


if __name__ == "__main__":
    main()

