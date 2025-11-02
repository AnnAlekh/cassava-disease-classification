import os
import json
import io
import time
import logging
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import psutil
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.jit as jit

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

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    memory_usage: float
    timestamp: str

class ModelInfo(BaseModel):
    name: str
    version: str
    format: str
    input_size: List[int]
    num_classes: int

# Класс модели
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

# Класс для инференса
class OptimizedCassavaPredictor:
    """Оптимизированный класс для инференса"""
    
    def __init__(self, model_path=None, model_format='pytorch', device='cpu'):
        self.device = torch.device(device)
        self.model_format = model_format
        self.class_names = [
            'Cassava Bacterial Blight (CBB)',
            'Cassava Brown Streak Disease (CBSD)', 
            'Cassava Green Mottle (CGM)',
            'Cassava Mosaic Disease (CMD)',
            'Healthy'
        ]
        
        # Загрузка трансформаций
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Загрузка модели
        if model_format == 'pytorch':
            self.load_pytorch_model(model_path)
        elif model_format == 'torchscript':
            self.load_torchscript_model(model_path)
        else:
            raise ValueError(f"Unsupported format: {model_format}")
    
    def load_pytorch_model(self, model_path):
        """Загрузка PyTorch модели"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = ImprovedCassavaModel(num_classes=5)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Проверяем формат checkpoint
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"PyTorch model loaded: {model_path}")
    
    def load_torchscript_model(self, model_path):
        """Загрузка TorchScript модели"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = jit.load(model_path, map_location=self.device)
        self.model.eval()
        logger.info(f"TorchScript model loaded: {model_path}")
    
    def preprocess_image(self, image):
        """Предобработка изображения"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        image_np = np.array(image)
        transformed = self.transform(image=image_np)
        return transformed['image'].unsqueeze(0)
    
    def predict(self, image, top_k=3):
        """Предсказание для одного изображения"""
        if not hasattr(self, 'model'):
            raise RuntimeError("Model not loaded")
            
        # Предобработка
        input_tensor = self.preprocess_image(image)
        
        # Инференс
        with torch.no_grad():
            if self.model_format == 'torchscript':
                outputs = self.model(input_tensor)
            else:
                outputs = self.model(input_tensor.to(self.device))
            
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
    description="API для классификации болезней листьев кассавы",
    version="1.0.0"
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

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    global model_predictor, model_info
    
    logger.info("🚀 Запуск Cassava Disease Classification API...")
    
    try:
        # Определяем путь к модели и формат
        model_dir = os.getenv("MODEL_DIR", "/app/models")
        model_path = None
        model_format = 'pytorch'
        
        # Приоритет: TorchScript > PyTorch
        torchscript_path = os.path.join(model_dir, "model_scripted.pt")
        pytorch_path = os.path.join(model_dir, "best_improved_model.pth")
        
        if os.path.exists(torchscript_path):
            model_path = torchscript_path
            model_format = 'torchscript'
        elif os.path.exists(pytorch_path):
            model_path = pytorch_path
            model_format = 'pytorch'
        else:
            # Попробуем найти в текущей директории
            if os.path.exists("model_scripted.pt"):
                model_path = "model_scripted.pt"
                model_format = 'torchscript'
            elif os.path.exists("best_improved_model.pth"):
                model_path = "best_improved_model.pth"
                model_format = 'pytorch'
            else:
                raise FileNotFoundError("No model file found")
        
        # Определяем устройство
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Загружаем модель
        model_predictor = OptimizedCassavaPredictor(
            model_path=model_path,
            model_format=model_format,
            device=device
        )
        
        # Создаем информацию о модели
        model_info = ModelInfo(
            name="Cassava Leaf Disease Classifier",
            version="1.0.0",
            format=model_format,
            input_size=[1, 3, 224, 224],
            num_classes=5
        )
        
        logger.info(f"✅ Модель загружена: {model_path} ({model_format})")
        logger.info(f"📊 Устройство: {device}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        model_predictor = None

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при завершении"""
    logger.info("🛑 Завершение работы API...")
    executor.shutdown(wait=True)

# Эндпоинты API
@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Cassava Leaf Disease Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья API"""
    memory_usage = psutil.virtual_memory().percent
    
    return HealthResponse(
        status="healthy" if model_predictor else "unhealthy",
        model_loaded=model_predictor is not None,
        memory_usage=memory_usage,
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Информация о загруженной модели"""
    if not model_info:
        raise HTTPException(status_code=503, detail="Model information not available")
    
    return model_info

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Предсказание класса болезни по изображению"""
    start_time = time.time()
    
    if not model_predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Проверяем тип файла
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Читаем и декодируем изображение
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Выполняем предсказание в отдельном потоке
        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(
            executor, 
            model_predictor.predict, 
            image
        )
        
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
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Пакетное предсказание для нескольких изображений"""
    if not model_predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    total_start_time = time.time()
    
    try:
        loop = asyncio.get_event_loop()
        
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


