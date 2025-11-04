import os
import json
import io
import time
import logging
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import psutil
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.jit as jit

# Импорт функции создания модели
try:
    from train_advanced_models import create_model
except ImportError:
    # Fallback: создаем функцию create_model локально
    def create_model(model_name='efficientnet_b0', num_classes=5, pretrained=True):
        """Создание модели с заданной архитектурой (fallback)"""
        if model_name.startswith('efficientnet_b'):
            size = model_name.split('_')[1].upper()
            if size == 'B0':
                backbone = models.efficientnet_b0(pretrained=pretrained)
                in_features = backbone.classifier[1].in_features
            elif size == 'B4':
                backbone = models.efficientnet_b4(pretrained=pretrained)
                in_features = backbone.classifier[1].in_features
            elif size == 'B7':
                backbone = models.efficientnet_b7(pretrained=pretrained)
                in_features = backbone.classifier[1].in_features
            else:
                raise ValueError(f"Unknown EfficientNet size: {size}")
            backbone.classifier = nn.Identity()
        else:
            # Fallback на EfficientNet-B0
            backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        
        classifier = nn.Sequential(
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
        
        class CustomModel(nn.Module):
            def __init__(self, backbone, classifier):
                super().__init__()
                self.backbone = backbone
                self.classifier = classifier
            
            def forward(self, x):
                features = self.backbone(x)
                return self.classifier(features)
        
        return CustomModel(backbone, classifier)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Модели данных для API
class PredictionResult(BaseModel):
    class_name: str
    class_id: int
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    inference_time: float
    timestamp: str
    model_name: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    memory_usage: float
    timestamp: str
    model_name: Optional[str] = None

class ModelInfo(BaseModel):
    name: str
    version: str
    format: str
    input_size: List[int]
    num_classes: int
    architecture: str

# Классы болезней кассавы
CLASS_NAMES = [
    'Cassava Bacterial Blight (CBB)',
    'Cassava Brown Streak Disease (CBSD)', 
    'Cassava Green Mottle (CGM)',
    'Cassava Mosaic Disease (CMD)',
    'Healthy'
]

# Класс для инференса с поддержкой TTA
class CassavaPredictor:
    """Класс для инференса с поддержкой TTA и разных моделей"""
    
    def __init__(self, model_path=None, model_name='efficientnet_b0', device='cpu', use_tta=False, tta_n_aug=5):
        self.device = torch.device(device)
        self.model_name = model_name
        self.use_tta = use_tta
        self.tta_n_aug = tta_n_aug
        self.class_names = CLASS_NAMES
        
        # Загрузка трансформаций
        self.transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # TTA трансформации
        self.tta_transforms = [
            A.Compose([
                A.Resize(height=224, width=224),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.Resize(height=224, width=224),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.Resize(height=224, width=224),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.Resize(height=224, width=224),
                A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.Resize(height=224, width=224),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
        ]
        
        # Загрузка модели
        if model_path:
            self.load_model(model_path)
        else:
            self.model = None
    
    def load_model(self, model_path):
        """Загрузка модели"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Создаем модель
        self.model = create_model(model_name=self.model_name, num_classes=5, pretrained=False)
        
        # Загружаем веса
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Проверяем формат checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded: {model_path} ({self.model_name})")
    
    def preprocess_image(self, image):
        """Предобработка изображения"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert('RGB')
        
        image_np = np.array(image)
        transformed = self.transform(image=image_np)
        return transformed['image'].unsqueeze(0)
    
    def tta_predict(self, image):
        """Предсказание с TTA"""
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("Model not loaded")
        
        predictions = []
        
        # Оригинальное изображение
        input_tensor = self.preprocess_image(image).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            predictions.append(outputs)
        
        # TTA аугментации
        image_np = np.array(image) if isinstance(image, Image.Image) else np.array(Image.open(io.BytesIO(image)).convert('RGB'))
        
        for i in range(min(self.tta_n_aug - 1, len(self.tta_transforms))):
            transform = self.tta_transforms[i]
            augmented = transform(image=image_np)
            input_tensor = augmented['image'].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                predictions.append(outputs)
        
        # Усреднение предсказаний
        avg_predictions = torch.stack(predictions).mean(dim=0)
        return avg_predictions
    
    def predict(self, image, top_k=5):
        """Предсказание для одного изображения"""
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Предобработка
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert('RGB')
        
        # Инференс
        with torch.no_grad():
            if self.use_tta:
                outputs = self.tta_predict(image)
            else:
                input_tensor = self.preprocess_image(image).to(self.device)
                outputs = self.model(input_tensor)
            
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Форматирование результатов
        results = []
        for i in range(top_k):
            class_idx = top_indices[0][i].item()
            confidence = top_probs[0][i].item()
            results.append({
                'class_name': self.class_names[class_idx],
                'class_id': class_idx,
                'confidence': confidence
            })
        
        return results

# Создаем FastAPI приложение
app = FastAPI(
    title="Cassava Leaf Disease Classification API",
    description="API для классификации болезней листьев кассавы с поддержкой TTA",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
model_predictor = None
model_info = None
executor = ThreadPoolExecutor(max_workers=4)

# Создаем директорию для статических файлов
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

def find_best_model():
    """Поиск лучшей доступной модели"""
    # Приоритет поиска моделей
    model_paths = [
        "models/efficientnet_b4_smoothing/best_model_epoch_18.pth",
        "models/efficientnet_b4_smoothing/best_model_epoch_14.pth",
        "models/efficientnet_b4_smoothing/best_model_epoch_9.pth",
        "notebooks/best_model.pth",
        "notebooks/best_improved_model.pth",
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            # Определяем архитектуру по пути
            if "efficientnet_b4" in model_path:
                return model_path, "efficientnet_b4"
            elif "efficientnet" in model_path:
                return model_path, "efficientnet_b0"
            else:
                return model_path, "efficientnet_b0"
    
    return None, None

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    global model_predictor, model_info
    
    logger.info("🚀 Запуск Cassava Disease Classification API...")
    
    try:
        # Определяем устройство
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"📊 Устройство: {device}")
        
        # Ищем модель
        model_path, model_name = find_best_model()
        
        if not model_path:
            logger.warning("⚠️ Модель не найдена, API будет работать без модели")
            model_predictor = None
            model_info = None
            return
        
        # Загружаем модель
        model_predictor = CassavaPredictor(
            model_path=model_path,
            model_name=model_name,
            device=device,
            use_tta=True,  # По умолчанию используем TTA
            tta_n_aug=5
        )
        
        # Создаем информацию о модели
        model_info = ModelInfo(
            name="Cassava Leaf Disease Classifier",
            version="2.0.0",
            format="pytorch",
            input_size=[1, 3, 224, 224],
            num_classes=5,
            architecture=model_name
        )
        
        logger.info(f"✅ Модель загружена: {model_path} ({model_name})")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        model_predictor = None
        model_info = None

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при завершении"""
    logger.info("🛑 Завершение работы API...")
    executor.shutdown(wait=True)

# Эндпоинты API
@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница с интерфейсом"""
    html_file = static_dir / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding='utf-8')
    else:
        return """
        <html>
            <head><title>Cassava Disease Classifier</title></head>
            <body>
                <h1>Cassava Leaf Disease Classification API</h1>
                <p>API работает. Загрузите файл index.html в папку static/</p>
                <p><a href="/docs">API Documentation</a></p>
            </body>
        </html>
        """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья API"""
    memory_usage = psutil.virtual_memory().percent
    
    return HealthResponse(
        status="healthy" if model_predictor else "unhealthy",
        model_loaded=model_predictor is not None,
        memory_usage=memory_usage,
        timestamp=datetime.now().isoformat(),
        model_name=model_info.architecture if model_info else None
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Информация о загруженной модели"""
    if not model_info:
        raise HTTPException(status_code=503, detail="Model information not available")
    
    return model_info

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), use_tta: bool = Query(True, description="Use Test Time Augmentation")):
    """Предсказание класса болезни по изображению"""
    start_time = time.time()
    
    if not model_predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Проверяем тип файла
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Читаем изображение
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Временно меняем режим TTA
        old_tta = model_predictor.use_tta
        model_predictor.use_tta = use_tta
        
        # Выполняем предсказание в отдельном потоке
        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(
            executor, 
            model_predictor.predict, 
            image
        )
        
        # Восстанавливаем режим TTA
        model_predictor.use_tta = old_tta
        
        inference_time = time.time() - start_time
        
        # Форматируем ответ
        prediction_results = [
            PredictionResult(
                class_name=pred['class_name'],
                class_id=pred['class_id'],
                confidence=pred['confidence']
            )
            for pred in predictions
        ]
        
        logger.info(f"✅ Prediction completed in {inference_time:.3f}s - Top class: {predictions[0]['class_name']}")
        
        return PredictionResponse(
            predictions=prediction_results,
            inference_time=inference_time,
            timestamp=datetime.now().isoformat(),
            model_name=model_info.architecture if model_info else "unknown"
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...), use_tta: bool = Query(True, description="Use Test Time Augmentation")):
    """Пакетное предсказание для нескольких изображений"""
    if not model_predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    total_start_time = time.time()
    
    try:
        loop = asyncio.get_event_loop()
        old_tta = model_predictor.use_tta
        model_predictor.use_tta = use_tta
        
        for file in files:
            if not file.content_type or not file.content_type.startswith('image/'):
                continue
            
            file_start_time = time.time()
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            # Предсказание для каждого файла
            predictions = await loop.run_in_executor(
                executor, 
                model_predictor.predict, 
                image
            )
            
            file_inference_time = time.time() - file_start_time
            
            results.append({
                "filename": file.filename,
                "predictions": predictions,
                "inference_time": file_inference_time
            })
        
        model_predictor.use_tta = old_tta
        total_time = time.time() - total_start_time
        
        return {
            "results": results,
            "total_files": len(results),
            "total_inference_time": total_time,
            "average_time_per_file": total_time / len(results) if results else 0
        }
        
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
