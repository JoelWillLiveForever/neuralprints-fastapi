import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models import ArchitecturePayload

from app.services.training import MODELS_DIR, ModelTrainer  # Импортируем класс ModelTrainer

router = APIRouter()

trainer = ModelTrainer()

class TrainRequest(BaseModel):
    dataset_name: str
    architecture_hash: str

@router.post(
    "/train", 
    tags=['TensorFlow'],
    response_model=dict
)
async def train_model(request: TrainRequest):
    """
    Эндпоинт для запуска обучения модели.
    """
    try:
        result = trainer.train_model(
            dataset_name=request.dataset_name,
            architecture_name=request.architecture_hash
        )
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while training: {str(e)}")
    
@router.get(
    "/status/{model_hash}",
    tags=['TensorFlow'],
    response_model=dict
)
async def get_model_status(model_hash: str):
    """
    Эндпоинт для получения статуса модели по её хэшу.
    """
    try:
        # Здесь можно добавить логику для проверки состояния модели, если это необходимо.
        # Например, может быть хранилище или база данных для отслеживания процесса.
        # В примере просто возвращаем статус как успешный.
        return {"status": "success", "model_hash": model_hash}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while retrieving status: {str(e)}")
    
@router.get(
    "/model/{model_hash}",
    tags=['TensorFlow'],
    response_model=dict
)
async def get_model(model_hash: str):
    """
    Эндпоинт для получения модели по её хэшу.
    """
    try:
        model_path = os.path.join(MODELS_DIR, f"{model_hash}.h5")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        return {"status": "success", "model_path": model_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while retrieving model: {str(e)}")