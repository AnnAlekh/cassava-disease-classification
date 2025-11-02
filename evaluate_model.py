"""
Скрипт для оценки модели на тестовых данных
Вычисляет метрики точности и сохраняет результаты
"""

import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Настройки
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Класс модели (должен совпадать с обученной моделью)
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
        
        # Улучшенный классификатор с регуляризацией
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


def load_model(model_path, device):
    """Загрузка обученной модели"""
    print(f"Загрузка модели из {model_path}...")
    
    model = ImprovedCassavaModel(num_classes=5)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Проверяем формат checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Модель загружена из checkpoint (эпоха {checkpoint.get('epoch', 'unknown')})")
        print(f"Лучший Val F1: {checkpoint.get('val_f1', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Модель загружена напрямую")
    
    model.to(device)
    model.eval()
    
    return model


def evaluate_model(model, test_loader, class_names, device):
    """Оценка модели на тестовых данных"""
    print("\n" + "="*60)
    print("ОЦЕНКА МОДЕЛИ НА ТЕСТОВЫХ ДАННЫХ")
    print("="*60)
    
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluation'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
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
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Classification report
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    
    return {
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs,
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
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def print_metrics(metrics, class_names):
    """Вывод метрик в консоль"""
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("="*60)
    
    print("\n📊 ОБЩИЕ МЕТРИКИ:")
    print(f"  • Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  • Precision (Macro):  {metrics['precision_macro']:.4f} ({metrics['precision_macro']*100:.2f}%)")
    print(f"  • Recall (Macro):     {metrics['recall_macro']:.4f} ({metrics['recall_macro']*100:.2f}%)")
    print(f"  • F1-Score (Macro):   {metrics['f1_macro']:.4f} ({metrics['f1_macro']*100:.2f}%)")
    print(f"  • Precision (Weighted): {metrics['precision_weighted']:.4f} ({metrics['precision_weighted']*100:.2f}%)")
    print(f"  • Recall (Weighted):    {metrics['recall_weighted']:.4f} ({metrics['recall_weighted']*100:.2f}%)")
    print(f"  • F1-Score (Weighted): {metrics['f1_weighted']:.4f} ({metrics['f1_weighted']*100:.2f}%)")
    
    print("\n🎯 МЕТРИКИ ПО КЛАССАМ:")
    print(f"{'Класс':<50} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 86)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<50} {metrics['precision_per_class'][i]:<12.4f} "
              f"{metrics['recall_per_class'][i]:<12.4f} {metrics['f1_per_class'][i]:<12.4f}")
    
    print("\n📋 ДЕТАЛЬНЫЙ ОТЧЕТ:")
    print(classification_report(
        metrics['targets'], 
        metrics['predictions'], 
        target_names=class_names, 
        digits=4
    ))


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """Визуализация confusion matrix"""
    # Создаем директорию если её нет
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar_kws={'label': 'Количество'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Предсказанные классы', fontsize=12)
    plt.ylabel('Реальные классы', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix сохранена: {save_path}")
    plt.close()


def save_results(metrics, model_path, test_size, class_names, output_dir='results'):
    """Сохранение результатов в файлы"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON файл с метриками
    results = {
        'timestamp': timestamp,
        'model_path': model_path,
        'test_size': test_size,
        'device': str(device),
        'metrics': {
            'accuracy': metrics['accuracy'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
            'f1_macro': metrics['f1_macro'],
            'precision_weighted': metrics['precision_weighted'],
            'recall_weighted': metrics['recall_weighted'],
            'f1_weighted': metrics['f1_weighted'],
            'precision_per_class': metrics['precision_per_class'],
            'recall_per_class': metrics['recall_per_class'],
            'f1_per_class': metrics['f1_per_class'],
            'confusion_matrix': metrics['confusion_matrix'],
            'classification_report': metrics['classification_report']
        }
    }
    
    json_path = os.path.join(output_dir, f'metrics_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Метрики сохранены: {json_path}")
    
    # Текстовый отчет
    report_path = os.path.join(output_dir, f'report_{timestamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ОТЧЕТ ОБ ОЦЕНКЕ МОДЕЛИ\n")
        f.write("="*60 + "\n\n")
        f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Модель: {model_path}\n")
        f.write(f"Тестовых примеров: {test_size}\n")
        f.write(f"Устройство: {device}\n\n")
        
        f.write("="*60 + "\n")
        f.write("ОБЩИЕ МЕТРИКИ\n")
        f.write("="*60 + "\n")
        f.write(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision (Macro):  {metrics['precision_macro']:.4f} ({metrics['precision_macro']*100:.2f}%)\n")
        f.write(f"Recall (Macro):     {metrics['recall_macro']:.4f} ({metrics['recall_macro']*100:.2f}%)\n")
        f.write(f"F1-Score (Macro):   {metrics['f1_macro']:.4f} ({metrics['f1_macro']*100:.2f}%)\n")
        f.write(f"Precision (Weighted): {metrics['precision_weighted']:.4f} ({metrics['precision_weighted']*100:.2f}%)\n")
        f.write(f"Recall (Weighted):    {metrics['recall_weighted']:.4f} ({metrics['recall_weighted']*100:.2f}%)\n")
        f.write(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f} ({metrics['f1_weighted']*100:.2f}%)\n\n")
        
        f.write("="*60 + "\n")
        f.write("МЕТРИКИ ПО КЛАССАМ\n")
        f.write("="*60 + "\n")
        f.write(f"{'Класс':<50} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 86 + "\n")
        
        # Используем оригинальные имена классов
        for class_name in class_names:
            if class_name in metrics['classification_report']:
                report = metrics['classification_report'][class_name]
                f.write(f"{class_name:<50} {report['precision']:<12.4f} "
                       f"{report['recall']:<12.4f} {report['f1-score']:<12.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("ДЕТАЛЬНЫЙ ОТЧЕТ\n")
        f.write("="*60 + "\n")
        f.write(classification_report(
            metrics['targets'], 
            metrics['predictions'], 
            target_names=class_names,
            digits=4
        ))
    
    print(f"✅ Текстовый отчет сохранен: {report_path}")
    
    return json_path, report_path


def main():
    """Главная функция"""
    print("="*60)
    print("ОЦЕНКА МОДЕЛИ CASSAVA LEAF DISEASE CLASSIFIER")
    print("="*60)
    
    # Пути
    model_path = "notebooks/best_improved_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return
    
    # Загрузка датасета
    print("\n📥 Загрузка датасета...")
    ds = load_dataset("pufanyi/cassava-leaf-disease-classification", "full")
    test_ds = ds['validation']  # Используем validation как test
    
    class_names = test_ds.features['label'].names
    print(f"✅ Датасет загружен: {len(test_ds)} тестовых примеров")
    print(f"Классы: {class_names}")
    
    # Трансформации для теста
    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Создание датасета и загрузчика
    test_dataset = CassavaDataset(test_ds, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Загрузка модели
    model = load_model(model_path, device)
    
    # Оценка модели
    metrics = evaluate_model(model, test_loader, class_names, device)
    
    # Вывод метрик
    print_metrics(metrics, class_names)
    
    # Визуализация confusion matrix
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']), 
        class_names,
        save_path='results/confusion_matrix.png'
    )
    
    # Сохранение результатов
    json_path, report_path = save_results(
        metrics, 
        model_path, 
        len(test_ds),
        class_names,
        output_dir='results'
    )
    
    print("\n" + "="*60)
    print("✅ ОЦЕНКА ЗАВЕРШЕНА")
    print("="*60)
    print(f"\nРезультаты сохранены:")
    print(f"  • JSON: {json_path}")
    print(f"  • Отчет: {report_path}")
    print(f"  • Confusion Matrix: results/confusion_matrix.png")


if __name__ == "__main__":
    main()

