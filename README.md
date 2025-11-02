# 🍃 Cassava Disease Classification Project

Система для автоматической классификации болезней листьев кассавы с использованием глубокого обучения. Проект включает веб-API на FastAPI и мобильное приложение на Flutter.

## 📋 Описание

Проект разработан для помощи фермерам в раннем обнаружении и классификации заболеваний листьев кассавы. Система использует сверточные нейронные сети на основе EfficientNet-B0 для распознавания 5 классов:

- **CBB** - Cassava Bacterial Blight (Бактериальная пятнистость)
- **CBSD** - Cassava Brown Streak Disease (Коричневая полосатость)
- **CGM** - Cassava Green Mottle (Зеленая крапчатость)
- **CMD** - Cassava Mosaic Disease (Мозаичная болезнь)
- **Healthy** - Здоровый лист

## 🎯 Текущие результаты модели

### Метрики на тестовых данных:
- **Accuracy**: 78.50%
- **Precision (Macro)**: 66.39%
- **Recall (Macro)**: 75.73%
- **F1-Score (Macro)**: 69.25%

## 🏗️ Архитектура проекта

```
cassava_project/
├── app.py                      # FastAPI приложение
├── requirements.txt            # Python зависимости
├── Dockerfile                  # Docker конфигурация для API
├── docker-compose.yml          # Docker Compose конфигурация
├── evaluate_model.py           # Скрипт для оценки модели
├── improve_model.py            # Скрипт для улучшения модели
├── notebooks/                  # Jupyter ноутбуки для обучения
│   ├── experiment.ipynb        # Эксперименты и обучение
│   └── *.pth                   # Обученные модели
├── mobile_app/                 # Flutter мобильное приложение
│   ├── lib/
│   │   ├── main.dart
│   │   ├── models/             # Модели данных
│   │   ├── screens/            # Экраны приложения
│   │   └── services/           # Сервисы (API, БД)
│   └── pubspec.yaml
└── results/                    # Результаты тестирования
    ├── metrics_*.json
    ├── report_*.txt
    └── confusion_matrix.png
```

## 🚀 Быстрый старт

### Предварительные требования

- Python 3.8+
- PyTorch 2.0+
- Flutter SDK 3.0+ (для мобильного приложения)
- Docker и Docker Compose (опционально)

### Установка и запуск API

1. **Клонируйте репозиторий**
```bash
git clone <repository-url>
cd cassava_project
```

2. **Создайте виртуальное окружение**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

3. **Установите зависимости**
```bash
pip install -r requirements.txt
```

4. **Разместите обученную модель**
```bash
# Поместите файл модели в notebooks/ или укажите путь в переменной окружения
export MODEL_DIR=/path/to/models
```

5. **Запустите API сервер**
```bash
python app.py
# или
uvicorn app:app --host 0.0.0.0 --port 8000
```

API будет доступен по адресу: `http://localhost:8000`
- Документация API: `http://localhost:8000/docs`
- Проверка здоровья: `http://localhost:8000/health`

### Запуск через Docker

```bash
docker-compose up -d
```

### Установка и запуск мобильного приложения

1. **Перейдите в директорию мобильного приложения**
```bash
cd mobile_app
```

2. **Установите зависимости Flutter**
```bash
flutter pub get
```

3. **Запустите приложение**
```bash
flutter run
```

## 📡 API Эндпоинты

### `GET /`
Корневой эндпоинт с информацией об API

### `GET /health`
Проверка состояния API и модели
```json
{
  "status": "healthy",
  "model_loaded": true,
  "memory_usage": 45.2,
  "timestamp": "2024-01-01T12:00:00"
}
```

### `GET /model/info`
Информация о загруженной модели

### `POST /predict`
Предсказание класса болезни по изображению
- **Body**: multipart/form-data с файлом изображения
- **Response**: JSON с предсказаниями, уверенностью и временем инференса

### `POST /predict/batch`
Пакетное предсказание для нескольких изображений (до 10 файлов)

## 🔧 Технологический стек

### Backend
- **FastAPI** - Современный веб-фреймворк для Python
- **PyTorch** - Глубокое обучение
- **EfficientNet-B0** - Архитектура CNN
- **Albumentations** - Аугментация изображений
- **Uvicorn** - ASGI сервер

### Mobile
- **Flutter** - Кроссплатформенная разработка
- **Dio** - HTTP клиент
- **SQFlite** - Локальная база данных
- **Image Picker** - Работа с камерой

### DevOps
- **Docker** - Контейнеризация
- **Docker Compose** - Оркестрация контейнеров

## 📊 Модель машинного обучения

### Архитектура
- **Backbone**: EfficientNet-B0 (предобученная на ImageNet)
- **Classifier**: Полносвязная сеть с регуляризацией
  - Dropout (0.3, 0.4, 0.2)
  - BatchNorm
  - ReLU активации

### Поддерживаемые форматы
- PyTorch (.pth)
- TorchScript (.pt)
- ONNX (.onnx) - в разработке

## 📈 Улучшение модели

См. файл `IMPROVEMENT_RECOMMENDATIONS.md` для детальных рекомендаций по улучшению результатов модели.

Основные направления:
- Использование Focal Loss для дисбаланса классов
- SMOTE oversampling для minority классов
- MixUp/CutMix аугментации
- Ансамблирование моделей
- Test-Time Augmentation (TTA)

## 🤝 Вклад в проект

Мы приветствуем вклад в проект! Пожалуйста:

1. Форкните репозиторий
2. Создайте ветку для новой функции (`git checkout -b feature/AmazingFeature`)
3. Зафиксируйте изменения (`git commit -m 'Add some AmazingFeature'`)
4. Отправьте в ветку (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

## 📝 Лицензия

Этот проект распространяется под лицензией MIT. См. файл `LICENSE` для подробностей.

## 👥 Авторы

- **Команда разработки** - Изначальная работа

## 🙏 Благодарности

- Kaggle Cassava Leaf Disease Classification Challenge
- PyTorch и FastAPI сообщества
- Все контрибьюторы проекта

## 📞 Контакты

По вопросам и предложениям создавайте Issues в репозитории.

---

⭐ Если проект полезен, поставьте звезду!

