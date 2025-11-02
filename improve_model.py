"""
Скрипт с примерами техник для улучшения модели
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# 1. FOCAL LOSS для лучшей работы с дисбалансом
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss для фокусировки на сложных примерах"""
    
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


# ============================================================
# 2. LABEL SMOOTHING для регуляризации
# ============================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing для предотвращения переобучения"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=1)
        weight = pred.new_ones(pred.size()) * self.smoothing / (pred.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=1).mean()
        return loss


# ============================================================
# 3. MIXUP АУГМЕНТАЦИЯ
# ============================================================

def mixup_data(x, y, alpha=1.0):
    """MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss для MixUp"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# 4. CUTMIX АУГМЕНТАЦИЯ
# ============================================================

def rand_bbox(size, lam):
    """Генерация случайного bounding box для CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    
    # Случайная точка
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


def cutmix(x, y, alpha=1.0):
    """CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Адаптируем lambda к фактическому соотношению пикселей
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y, y[index], lam


# ============================================================
# 5. УЛУЧШЕННЫЕ АУГМЕНТАЦИИ ДЛЯ MINORITY КЛАССОВ
# ============================================================

class EnhancedAugmentationPipeline:
    """Усиленные аугментации для minority классов"""
    
    def __init__(self, image_size=224):
        self.image_size = image_size
    
    def get_aggressive_augmentations(self):
        """Агрессивные аугментации для minority классов"""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.8),
            A.VerticalFlip(p=0.6),
            A.RandomRotate90(p=0.6),
            A.ShiftScaleRotate(
                shift_limit=0.25,
                scale_limit=0.4,
                rotate_limit=60,
                p=0.8
            ),
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                alpha_affine=120 * 0.03,
                p=0.3
            ),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=0.3),
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=30,
                    sat_shift_limit=40,
                    val_shift_limit=30,
                    p=1.0
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.9),
            A.OneOf([
                A.GaussianBlur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.4),
            A.GaussNoise(var_limit=(10.0, 100.0), p=0.4),
            A.RandomGridShuffle(grid=(4, 4), p=0.2),
            A.CoarseDropout(
                max_holes=16,
                max_height=32,
                max_width=32,
                min_holes=4,
                min_height=16,
                min_width=16,
                p=0.4
            ),
            A.RandomErasing(p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


# ============================================================
# 6. TEST TIME AUGMENTATION (TTA)
# ============================================================

class TTA_Predictor:
    """Test Time Augmentation для улучшения предсказаний"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def predict_with_tta(self, image, n_aug=5):
        """Предсказание с применением TTA"""
        predictions = []
        
        # Оригинальное изображение
        with torch.no_grad():
            pred = self.model(image.to(self.device))
            predictions.append(torch.softmax(pred, dim=1))
        
        # Аугментации
        tta_transforms = [
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.RandomRotate90(p=1.0),
            A.Transpose(p=1.0),
        ]
        
        for transform in tta_transforms[:n_aug-1]:
            augmented = transform(image=image.cpu().numpy())
            aug_tensor = torch.tensor(augmented['image']).unsqueeze(0)
            
            with torch.no_grad():
                pred = self.model(aug_tensor.to(self.device))
                predictions.append(torch.softmax(pred, dim=1))
        
        # Среднее предсказаний
        avg_pred = torch.stack(predictions).mean(dim=0)
        return avg_pred


# ============================================================
# 7. OPTIMAL THRESHOLD FINDING
# ============================================================

def find_optimal_thresholds(model, val_loader, device):
    """Нахождение оптимальных порогов для каждого класса"""
    from sklearn.metrics import precision_recall_curve
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    optimal_thresholds = {}
    for class_idx in range(all_probs.shape[1]):
        precision, recall, thresholds = precision_recall_curve(
            (all_labels == class_idx).astype(int),
            all_probs[:, class_idx]
        )
        
        # Находим threshold, максимизирующий F1
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_thresholds[class_idx] = thresholds[optimal_idx]
    
    return optimal_thresholds


# ============================================================
# 8. ENSEMBLE PREDICTIONS
# ============================================================

def ensemble_predict(models, data_loader, device, method='average'):
    """Ансамблирование предсказаний нескольких моделей"""
    all_predictions = []
    
    for model in models:
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                predictions.append(probs)
        
        all_predictions.append(torch.cat(predictions, dim=0))
    
    # Объединение предсказаний
    stacked_preds = torch.stack(all_predictions)
    
    if method == 'average':
        avg_preds = stacked_preds.mean(dim=0)
    elif method == 'max':
        avg_preds = stacked_preds.max(dim=0)[0]
    elif method == 'weighted':
        weights = torch.softmax(torch.ones(len(models)), dim=0)
        avg_preds = (stacked_preds * weights.view(-1, 1, 1)).sum(dim=0)
    
    return avg_preds.argmax(dim=1)


# ============================================================
# 9. УЛУЧШЕННЫЙ ТРЕЙНЕР С НОВЫМИ ТЕХНИКАМИ
# ============================================================

class AdvancedTrainer:
    """Улучшенный тренер с новыми техниками"""
    
    def __init__(self, model, train_loader, val_loader, device, use_mixup=True, use_cutmix=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        
        # Используем Focal Loss вместо обычного
        class_weights = torch.tensor([2.0, 1.5, 1.2, 0.3, 1.0]).to(device)
        self.criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        
        # Оптимизатор
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        # OneCycleLR scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            epochs=30,
            steps_per_epoch=len(train_loader)
        )
    
    def train_epoch(self):
        """Одна эпоха обучения с MixUp/CutMix"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Применяем MixUp или CutMix
            use_mixup = self.use_mixup and np.random.rand() < 0.5
            use_cutmix = self.use_cutmix and not use_mixup and np.random.rand() < 0.5
            
            if use_mixup:
                mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=1.0)
                self.optimizer.zero_grad()
                outputs = self.model(mixed_images)
                loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
            
            elif use_cutmix:
                cutmix_images, labels_a, labels_b, lam = cutmix(images, labels, alpha=1.0)
                self.optimizer.zero_grad()
                outputs = self.model(cutmix_images)
                loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
            
            else:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss, all_preds, all_targets


# ============================================================
# 10. ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ПРИМЕРЫ ТЕХНИК ДЛЯ УЛУЧШЕНИЯ МОДЕЛИ")
    print("=" * 60)
    
    print("\n1. Focal Loss:")
    print("   - Лучше работает с дисбалансом классов")
    print("   - Фокусируется на сложных примерах")
    print("   - Ожидаемое улучшение: +3-5% F1-score")
    
    print("\n2. MixUp/CutMix:")
    print("   - Создает синтетические примеры")
    print("   - Улучшает обобщение модели")
    print("   - Ожидаемое улучшение: +1-2% F1-score")
    
    print("\n3. TTA (Test Time Augmentation):")
    print("   - Улучшает точность во время инференса")
    print("   - Ожидаемое улучшение: +1-3% accuracy")
    
    print("\n4. Ensemble:")
    print("   - Объединение нескольких моделей")
    print("   - Ожидаемое улучшение: +2-4% accuracy")
    
    print("\n" + "=" * 60)
    print("Для реализации см. соответствующие функции в коде")
    print("=" * 60)

