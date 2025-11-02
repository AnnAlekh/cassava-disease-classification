# Рекомендации по улучшению результатов модели

## Текущее состояние модели

### Метрики на тестовых данных:
- **Accuracy**: 78.50%
- **Precision (Macro)**: 66.39%
- **Recall (Macro)**: 75.73%
- **F1-Score (Macro)**: 69.25%

### Основные проблемы:

1. **Низкая точность (Precision) для minority классов:**
   - CBB: 46.77% (модель переклассифицирует)
   - CGM: 56.00%
   - Healthy: 51.55%

2. **Дисбаланс классов:**
   - CMD: 61.38% (доминирующий класс)
   - CBB: 5.13% (самый редкий класс)
   - Соотношение: ~12:1

---

## Рекомендации по улучшению

### 1. Улучшенная обработка дисбаланса классов

#### 1.1. Использование Focal Loss вместо Weighted Cross-Entropy
Focal Loss лучше работает с дисбалансом, фокусируясь на сложных примерах:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # веса классов
        self.gamma = gamma  # фокусирующий параметр
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
```

**Ожидаемое улучшение**: +3-5% в macro F1-score

#### 1.2. SMOTE для oversampling minority классов
Использование Synthetic Minority Oversampling Technique:

```python
from imblearn.over_sampling import SMOTE

# Применить SMOTE к minority классам (CBB, CBSD)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**Ожидаемое улучшение**: +2-4% в precision для minority классов

#### 1.3. Комбинация Oversampling + Undersampling
- Oversampling для CBB и CBSD (меньшинство)
- Undersampling для CMD (большинство)

---

### 2. Улучшенные аугментации данных

#### 2.1. MixUp аугментация
Создание синтетических примеров путем смешивания двух изображений:

```python
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

**Ожидаемое улучшение**: +1-2% в общем F1-score

#### 2.2. CutMix аугментация
Более эффективная версия MixUp:

```python
def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y, y[index], lam
```

#### 2.3. Более агрессивные аугментации для minority классов
- Elastic Transform
- Grid Distortion
- Optical Distortion
- Random Erasing

---

### 3. Улучшение архитектуры модели

#### 3.1. Использование более мощных моделей
- **EfficientNet-B3** или **EfficientNet-B4** вместо B0
- **Vision Transformer (ViT)**
- **ConvNeXt**

#### 3.2. Transfer Learning с разными предобученными моделями
Обучить несколько моделей и использовать ансамбль:
- EfficientNet-B0
- ResNet-50
- DenseNet-121
- RegNet

#### 3.3. Добавление Attention механизмов
- SE-Net (Squeeze-and-Excitation)
- CBAM (Convolutional Block Attention Module)

---

### 4. Ансамблирование моделей

#### 4.1. Voting Ensemble
Объединение предсказаний нескольких моделей:

```python
def ensemble_predict(models, data_loader):
    all_predictions = []
    for model in models:
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                outputs = model(batch['image'])
                preds = torch.softmax(outputs, dim=1)
                predictions.append(preds)
        all_predictions.append(torch.cat(predictions, dim=0))
    
    # Среднее предсказаний
    avg_predictions = torch.stack(all_predictions).mean(dim=0)
    return avg_predictions.argmax(dim=1)
```

**Ожидаемое улучшение**: +2-4% в accuracy

#### 4.2. Stacking Ensemble
Использование мета-модели для комбинирования предсказаний

---

### 5. Оптимизация процесса обучения

#### 5.1. Использование Learning Rate Finder
Автоматический поиск оптимального learning rate:

```python
from torch_lr_finder import LRFinder

lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
lr_finder.plot()
lr_finder.reset()
```

#### 5.2. Использование OneCycleLR
Более эффективный шедулер learning rate:

```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer, 
    max_lr=1e-3,
    epochs=30,
    steps_per_epoch=len(train_loader)
)
```

#### 5.3. Label Smoothing
Регуляризация для предотвращения переобучения:

```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=1)
        weight = pred.new_ones(pred.size()) * self.smoothing / (pred.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=1).mean()
        return loss
```

**Ожидаемое улучшение**: +1-2% в общих метриках

---

### 6. Метрики и мониторинг

#### 6.1. Фокус на macro F1-score вместо accuracy
Для несбалансированных данных macro F1 более информативен

#### 6.2. Использование confusion matrix для анализа ошибок
Идентификация паттернов неправильной классификации

#### 6.3. Per-class metrics monitoring
Отдельный мониторинг метрик для каждого класса

---

### 7. Техники для улучшения precision

#### 7.1. Threshold Tuning
Настройка порога вероятности для каждого класса:

```python
def find_optimal_thresholds(model, val_loader, class_names):
    thresholds = {}
    for class_idx in range(len(class_names)):
        # Найти оптимальный threshold для класса
        # используя ROC curve или precision-recall curve
        thresholds[class_idx] = optimal_threshold
    return thresholds
```

#### 7.2. Cost-Sensitive Learning
Использование матрицы стоимости ошибок:

```python
cost_matrix = torch.tensor([
    [0, 5, 3, 2, 4],  # CBB
    [5, 0, 4, 3, 2],  # CBSD
    [3, 4, 0, 2, 3],  # CGM
    [2, 3, 2, 0, 1],  # CMD
    [4, 2, 3, 1, 0]   # Healthy
])
```

---

### 8. Дополнительные техники

#### 8.1. Тестовое время аугментация (TTA)
Применение аугментаций во время инференса:

```python
def tta_predict(model, image, n_aug=5):
    predictions = []
    for _ in range(n_aug):
        augmented = augment(image)
        pred = model(augmented)
        predictions.append(pred)
    return torch.stack(predictions).mean(dim=0)
```

#### 8.2. Knowledge Distillation
Обучение меньшей модели от большей:

```python
def distillation_loss(student_outputs, teacher_outputs, labels, alpha=0.7, T=3):
    soft_loss = F.kl_div(
        F.log_softmax(student_outputs / T, dim=1),
        F.softmax(teacher_outputs / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)
    hard_loss = F.cross_entropy(student_outputs, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

---

## Приоритетность рекомендаций

### Высокий приоритет (быстрое улучшение):
1. ✅ Использование Focal Loss
2. ✅ Threshold Tuning для улучшения precision
3. ✅ Улучшенные аугментации для minority классов
4. ✅ Тестовое время аугментация (TTA)

### Средний приоритет (значительное улучшение):
5. ✅ SMOTE oversampling
6. ✅ MixUp/CutMix аугментация
7. ✅ Обучение нескольких моделей для ансамбля
8. ✅ OneCycleLR шедулер

### Низкий приоритет (долгосрочные улучшения):
9. ✅ Использование более мощных архитектур (EfficientNet-B3/B4)
10. ✅ Vision Transformer
11. ✅ Knowledge Distillation

---

## Ожидаемые результаты

Применение рекомендаций высокого и среднего приоритета должно дать:

- **Accuracy**: 78.50% → **82-85%**
- **Precision (Macro)**: 66.39% → **72-75%**
- **Recall (Macro)**: 75.73% → **78-80%**
- **F1-Score (Macro)**: 69.25% → **75-78%**

Особенно заметное улучшение ожидается для:
- CBB: Precision 46.77% → **55-60%**
- CGM: Precision 56.00% → **65-70%**
- Healthy: Precision 51.55% → **60-65%**

---

## Примеры кода для реализации

См. файл `improve_model.py` с реализацией ключевых техник.

