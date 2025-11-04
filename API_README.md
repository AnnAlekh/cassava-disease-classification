# Cassava Leaf Disease Classification API

FastAPI приложение с браузерным интерфейсом для классификации болезней листьев кассавы.

## Особенности

- 🚀 **FastAPI** - современный и быстрый веб-фреймворк
- 🎨 **Красивый веб-интерфейс** - интуитивный браузерный интерфейс с визуализацией
- 🔬 **Test Time Augmentation (TTA)** - улучшение точности предсказаний
- 📊 **Визуализация результатов** - графики вероятностей и статистика
- 🖼️ **Drag & Drop** - удобная загрузка изображений
- 🔄 **Поддержка разных моделей** - EfficientNet, ConvNeXt, ViT и др.

## Установка

Все зависимости уже должны быть установлены из `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Запуск

### Запуск сервера

```bash
python app.py
```

Или с помощью uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Доступ к интерфейсу

После запуска откройте браузер и перейдите по адресу:

```
http://localhost:8000
```

### API документация

Swagger UI доступен по адресу:

```
http://localhost:8000/docs
```

ReDoc доступен по адресу:

```
http://localhost:8000/redoc
```

## API Endpoints

### GET `/`
Главная страница с веб-интерфейсом.

### GET `/health`
Проверка состояния API и модели.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "memory_usage": 45.2,
  "timestamp": "2024-01-01T12:00:00",
  "model_name": "efficientnet_b4"
}
```

### GET `/model/info`
Информация о загруженной модели.

**Response:**
```json
{
  "name": "Cassava Leaf Disease Classifier",
  "version": "2.0.0",
  "format": "pytorch",
  "input_size": [1, 3, 224, 224],
  "num_classes": 5,
  "architecture": "efficientnet_b4"
}
```

### POST `/predict`
Классификация изображения листа кассавы.

**Parameters:**
- `file` (file): Изображение для классификации
- `use_tta` (bool, optional): Использовать TTA (по умолчанию: true)

**Response:**
```json
{
  "predictions": [
    {
      "class_name": "Cassava Mosaic Disease (CMD)",
      "class_id": 3,
      "confidence": 0.85
    },
    ...
  ],
  "inference_time": 0.234,
  "timestamp": "2024-01-01T12:00:00",
  "model_name": "efficientnet_b4"
}
```

### POST `/predict/batch`
Пакетная классификация (до 10 изображений).

**Parameters:**
- `files` (list of files): Изображения для классификации
- `use_tta` (bool, optional): Использовать TTA (по умолчанию: true)

## Модели

Приложение автоматически ищет лучшую доступную модель в следующем порядке:

1. `models/efficientnet_b4_smoothing/best_model_epoch_18.pth`
2. `models/efficientnet_b4_smoothing/best_model_epoch_14.pth`
3. `models/efficientnet_b4_smoothing/best_model_epoch_9.pth`
4. `notebooks/best_model.pth`
5. `notebooks/best_improved_model.pth`

## Классы болезней

1. **Cassava Bacterial Blight (CBB)** - Бактериальный ожог
2. **Cassava Brown Streak Disease (CBSD)** - Бурая полосатость
3. **Cassava Green Mottle (CGM)** - Зеленая крапчатость
4. **Cassava Mosaic Disease (CMD)** - Мозаичная болезнь
5. **Healthy** - Здоровое растение

## Использование через curl

```bash
# Классификация с TTA
curl -X POST "http://localhost:8000/predict?use_tta=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"

# Классификация без TTA
curl -X POST "http://localhost:8000/predict?use_tta=false" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

## Использование через Python

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("image.jpg", "rb")}
params = {"use_tta": True}

response = requests.post(url, files=files, params=params)
data = response.json()

print(f"Top prediction: {data['predictions'][0]['class_name']}")
print(f"Confidence: {data['predictions'][0]['confidence']:.2%}")
print(f"Inference time: {data['inference_time']:.3f}s")
```

## Docker

Если у вас есть Dockerfile, вы можете запустить приложение в контейнере:

```bash
docker build -t cassava-api .
docker run -p 8000:8000 cassava-api
```

## Troubleshooting

### Модель не загружается

Убедитесь, что файл модели существует в одной из ожидаемых директорий. Проверьте логи при запуске.

### Ошибка CUDA

Если у вас нет GPU, приложение автоматически использует CPU. Это может быть медленнее, но будет работать.

### Порт занят

Измените порт в `app.py` или используйте флаг `--port` с uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

## Производительность

- **CPU**: ~200-500ms на изображение (без TTA)
- **GPU**: ~50-100ms на изображение (без TTA)
- **TTA**: увеличивает время обработки в 5 раз, но улучшает точность

## Лицензия

MIT

