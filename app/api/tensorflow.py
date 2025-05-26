from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.training import ModelTrainer  # Импортируем класс ModelTrainer

router = APIRouter()

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
        trainer = ModelTrainer()
        
        result = trainer.train_model(
            dataset_name=request.dataset_name,
            architecture_name=request.architecture_hash
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while training: {str(e)}")