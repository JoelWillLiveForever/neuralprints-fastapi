import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import logging

# Уникальный ключ логгера на этот файл
LOGGER_KEY = "models.py"
MODELS_DIR = "./saved_models"

# Получение глобального логгера
logger = logging.getLogger(LOGGER_KEY)

router = APIRouter()
    
@router.get(
    "/{model_hash}/download/h5",
    tags=['Models'],
)
async def download_model_h5(model_hash: str):
    logger.info(f"model_hash = {model_hash}")
    
    file_path = os.path.join(MODELS_DIR, f"{model_hash}.h5")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model not found")

    return FileResponse(
        path=file_path,
        filename=f"{model_hash}.h5",
        media_type="application/octet-stream"
    )

# @router.get(
#     "/executable/python/{model_hash}",
#     tags=['TensorFlow'],
#     response_model=dict
# )
# async def get_model(model_hash: str):
#     pass

# @router.get(
#     "/status/{model_hash}",
#     tags=['TensorFlow'],
#     response_model=dict
# )
# async def get_model_status(model_hash: str):
#     """
#     Эндпоинт для получения статуса модели по её хэшу.
#     """
#     try:
#         # Здесь можно добавить логику для проверки состояния модели, если это необходимо.
#         # Например, может быть хранилище или база данных для отслеживания процесса.
#         # В примере просто возвращаем статус как успешный.
#         return {"status": "success", "model_hash": model_hash}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error while retrieving status: {str(e)}")
    
# @router.get(
#     "/model/{model_hash}",
#     tags=['TensorFlow'],
#     response_model=dict
# )
# async def get_model(model_hash: str):
#     """
#     Эндпоинт для получения модели по её хэшу.
#     """
#     try:
#         model_path = os.path.join(MODELS_DIR, f"{model_hash}.h5")
#         if not os.path.exists(model_path):
#             raise HTTPException(status_code=404, detail="Model not found")
#         return {"status": "success", "model_path": model_path}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error while retrieving model: {str(e)}")